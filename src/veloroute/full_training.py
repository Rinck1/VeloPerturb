"""Staged full-model training: A encoders, B static warmup, C delayed EM, D gate.

Velocity is a reference for the dynamics encoder and a directional compatibility
cost, never the expression-flow regression target. Teachers are training-only.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .artifacts import object_hash
from .contracts import FitScope
from .coupling import sinkhorn_log
from .full_components import corrupt_source, mode_regularization
from .full_model import FullVeloRoute
from .mixture_energy import mixture_energy


@dataclass(frozen=True)
class FullTrainConfig:
    implementation_revision: str = 'gradient_audit_20260921_v2'
    stage_a_steps: int = 1000
    stage_b_steps: int = 2000
    stage_c_steps: int = 12000
    batch_size: int = 64
    e_interval: int = 300
    teacher_lag: int = 3
    activation_interval: int = 300
    reference_interval: int = 1000
    usage_interval: int = 1000
    corruption_interval: int = 4
    checkpoint_interval: int = 2000
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 5e-5
    gate_learning_rate: float = 1e-3
    lambda_dynamic: float = .1
    lambda_field: float = .01
    teacher_weight: float = .1
    mode_regularization_weight: float = 1.
    sinkhorn_epsilon: float = .1
    mode_temperature: float = .5
    sinkhorn_iterations: int = 300
    time_start: float = 4.
    time_end: float = 5.
    mask_ratio: float = .3
    seed: int = 0
    gfg_native_weight: float = .1
    velocity_background: str = ''
    coupling_mode: str = 'ot'
    dist_entropy_weight: float = .01
    dist_source_batch: int = 64
    dist_target_batch: int = 128

    def __post_init__(self):
        if self.implementation_revision != 'gradient_audit_20260921_v2':
            raise ValueError('Unsupported training semantics; start an audited run')
        for name in ('stage_a_steps', 'stage_b_steps', 'stage_c_steps', 'batch_size', 'e_interval', 'teacher_lag',
                     'activation_interval', 'reference_interval', 'usage_interval', 'corruption_interval',
                     'checkpoint_interval', 'sinkhorn_iterations', 'dist_source_batch', 'dist_target_batch'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'Need a positive bounded training count: {name}')
        for name in ('learning_rate', 'encoder_learning_rate', 'gate_learning_rate', 'sinkhorn_epsilon', 'mode_temperature'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'Invalid optimizer/transport parameter: {name}')
        for name in ('lambda_dynamic', 'lambda_field', 'teacher_weight', 'mode_regularization_weight', 'gfg_native_weight', 'dist_entropy_weight'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'Invalid loss weight: {name}')
        if not math.isfinite(self.time_start+self.time_end) or self.time_end <= self.time_start or not 0 < self.mask_ratio < 1:
            raise ValueError('Invalid physical interval or mask ratio')
        if self.coupling_mode not in {'ot', 'distribution_matching'}:
            raise ValueError('Unknown coupling_mode')
        if self.velocity_background and self.coupling_mode != 'distribution_matching':
            raise ValueError('Velocity background is supported only for distribution_matching')


class DelayedTeacher(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.router_hidden_dim
        dim = 2*config.state_dim+config.representation_dim+config.condition_dim
        self.network = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                     nn.Linear(width, width), nn.SiLU(), nn.Linear(width, config.max_experts))

    def forward(self, z, representation, condition, future_encoding):
        return self.network(torch.cat((z.detach(), representation.detach(), condition.detach(), future_encoding.detach()), -1))


class DelayedLabels:
    def __init__(self, lag):
        if lag < 1:
            raise ValueError('Teacher needs a strictly positive E-round lag')
        self.lag, self.queue = lag, deque()

    def push(self, record):
        old = self.queue.popleft() if len(self.queue) >= self.lag else None
        self.queue.append({k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in record.items()})
        return old


def base_rate(field, z, time, condition):
    t = torch.as_tensor(time, device=z.device, dtype=z.dtype)
    if t.ndim == 0:
        t = t.expand(len(z))
    h = field.blocks(field.input(torch.cat((z, field.time(t), condition), -1)))
    return field.base(h)


def direction_cost(values, reference):
    valid = (values.norm(dim=-1) > 1e-8) & (reference.norm(dim=-1) > 1e-8)
    return torch.where(valid, 1-F.cosine_similarity(values, reference, dim=-1), torch.zeros_like(valid, dtype=values.dtype))


@torch.no_grad()
def full_responsibilities(model, context, target, config, *, teacher=None, target_encoder=None):
    """One-condition no-grad E step. Direction terms include a k-dependent term."""
    z, e = context.z0, context.condition.detach()
    if not torch.allclose(e, e[:1].expand_as(e), atol=1e-6):
        raise ValueError('E step must use a single consistent perturbation condition')
    dt = config.time_end-config.time_start
    fields = model.field(z, config.time_start, e)
    intrinsic = model.intrinsic(z, z, context.velocity) if model.config.use_intrinsic else torch.zeros_like(z)
    predicted = z[:, None, :]+dt*(fields+intrinsic[:, None, :])
    expression = (predicted[:, None, :, :]-target[None, :, None, :]).square().mean(-1)
    cost = expression/expression.mean().clamp_min(1e-8)
    dynamic = torch.zeros_like(cost)
    if config.lambda_dynamic and model.config.use_dynamics:
        rate = (target[None, :, :]-z[:, None, :])/dt
        for fraction in (1/3, 1/2, 2/3):
            points = (1-fraction)*z[:, None, :] + fraction*target[None, :, :]
            reference, _ = model.reference(points.reshape(-1, z.shape[-1]))
            reference = reference.reshape_as(points)
            pair = direction_cost(rate, reference)
            flat_e = e[:, None, :].expand(-1, len(target), -1).reshape(-1, e.shape[-1])
            candidates = model.field(points.reshape(-1, z.shape[-1]), config.time_start+fraction*dt, flat_e)
            candidates = candidates.reshape(len(z), len(target), model.config.max_experts, -1)
            mode = direction_cost(candidates+intrinsic[:, None, None, :], reference[:, :, None, :].expand_as(candidates))
            dynamic += .5*(pair[..., None]+mode)/3
        cost += config.lambda_dynamic*dynamic
    future_h = target_encoder(target) if target_encoder is not None else target
    pair_inputs = {
        'z': z[:, None, :].expand(-1, len(target), -1).reshape(-1, z.shape[-1]),
        'r': context.representation[:, None, :].expand(-1, len(target), -1).reshape(-1, context.representation.shape[-1]),
        'e': e[:, None, :].expand(-1, len(target), -1).reshape(-1, e.shape[-1]),
        'h': future_h[None, :, :].expand(len(z), -1, -1).reshape(-1, future_h.shape[-1]),
    }
    if teacher is not None and config.teacher_weight:
        logits = teacher(*[pair_inputs[key] for key in ('z', 'r', 'e', 'h')])
        prior = logits.masked_fill(~model.active_modes[None, :], -torch.inf).log_softmax(-1)
        cost -= config.mode_temperature*config.teacher_weight*prior.reshape_as(cost)
    cost = cost.masked_fill(~model.active_modes[None, None, :], torch.inf)
    active_k = int(model.active_modes.sum())
    pair_cost = -config.mode_temperature*(torch.logsumexp(-cost/config.mode_temperature, -1)-math.log(active_k))
    plan = sinkhorn_log(pair_cost, epsilon=config.sinkhorn_epsilon, iterations=config.sinkhorn_iterations)
    row_error = float((plan.sum(1)-1/len(z)).abs().max())
    col_error = float((plan.sum(0)-1/len(target)).abs().max())
    if max(row_error, col_error) > 1e-3:
        raise RuntimeError('Full Sinkhorn failed marginal convergence; no pseudo-labels emitted')
    pair_labels = (-cost/config.mode_temperature).softmax(-1)
    gamma = plan[..., None]*pair_labels
    labels = gamma.sum(1)/gamma.sum((1, 2))[:, None].clamp_min(1e-12)
    record = {**pair_inputs, 'labels': pair_labels.reshape(-1, model.config.max_experts),
              'weight': plan.flatten(), 'active': model.active_modes.clone()}
    return {'gamma': gamma, 'labels': labels, 'record': record, 'dynamic_cost': dynamic,
            'row_error': row_error, 'column_error': col_error}


def field_consistency_loss(model, source, target, condition, config):
    """Directional along-path cost trains the shared backbone ONLY, not a v target."""
    losses = []
    for fraction in (1/3, 1/2, 2/3):
        z = ((1-fraction)*source+fraction*target).detach()
        ref, _ = model.reference(z)
        field = base_rate(model.field, z, config.time_start+fraction*(config.time_end-config.time_start), condition.detach())
        losses.append(direction_cost(field, ref.detach()).mean())
    return torch.stack(losses).mean()


class FullTrainer:
    """All task losses have separate optimizers and explicit detach boundaries."""
    def __init__(self, model: FullVeloRoute, config: FullTrainConfig):
        if (model.config.support_start, model.config.support_end) != (config.time_start, config.time_end):
            raise ValueError('Gate time support must match the actual training interval; a planned interval is not observed support')
        self.model, self.config = model, config
        device = next(model.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(config.seed)
        self.teacher = DelayedTeacher(model.config).to(device)
        self.target_encoder = copy.deepcopy(model.state_encoder).requires_grad_(False)
        self.history = DelayedLabels(config.teacher_lag)
        self.encoder_optimizer = torch.optim.AdamW(list(model.state_encoder.parameters())+list(model.dynamics.parameters()),
                                                   lr=config.encoder_learning_rate)
        self.field_parameters = list(model.field.parameters())+list(model.condition_encoder.parameters())+list(model.intrinsic.parameters())
        self.field_optimizer = torch.optim.AdamW(self.field_parameters, lr=config.learning_rate)
        self.router_parameters = list(model.router.parameters())
        groups = [{'params': list(self.router_parameters), 'lr': config.learning_rate, 'base_lr': config.learning_rate}]
        if model.config.joint_dynamics:
            dynamics_parameters = [p for p in model.dynamics.parameters() if p.requires_grad]
            self.router_parameters += dynamics_parameters
            groups.append({'params': dynamics_parameters, 'lr': config.encoder_learning_rate,
                           'base_lr': config.encoder_learning_rate})
        self.router_optimizer = torch.optim.AdamW(groups)
        self.gate_optimizer = torch.optim.AdamW(model.gate.parameters(), lr=config.gate_learning_rate)
        self.teacher_optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=config.learning_rate)
        self.noise_optimizer = torch.optim.AdamW([model.noise_raw], lr=config.learning_rate)
        self.optimizers = {name: getattr(self, name+'_optimizer') for name in ('encoder', 'field', 'router', 'gate', 'teacher', 'noise')}
        self.stage_a_done = self.stage_b_done = False
        self.completed_c_steps = self.e_round = self.teacher_updates = self.corruptions = 0
        self.cache, self.trace = {}, []
        self.data_hash = None
        self.velocity_background = None
        if getattr(config, 'velocity_background', ''):
            from .velocity_background import load_background
            bg = load_background(config.velocity_background, 'train')
            self.velocity_background = {key: torch.as_tensor(bg[key], dtype=torch.float32, device=device)
                                       for key in ('r', 'v')}
            self.velocity_background['cell_ids'] = bg['cell_ids']
        self.progress_callback = None

    def _draw(self, indices):
        select = torch.randint(len(indices), (self.config.batch_size,), generator=self.generator, device=indices.device)
        return indices[select]

    def _step(self, optimizer, loss, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite full-model loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        parameters = list(parameters)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Nonfinite full-model gradient')
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()

    def _log(self, stage, step, **values):
        row = dict(stage=stage, step=step, encoder_loss=0., field_loss=0., along_path_loss=0., router_loss=0.,
                   teacher_loss=0., noise_loss=0., gate_loss=0., mean_gate=0., mode_entropy=0., energy=0.,
                   active_modes=int(self.model.active_modes.sum()), e_round=self.e_round, teacher_updates=self.teacher_updates,
                   gfg_native_loss=0., gfg_task_gradient_norm=0.)
        row.update({k: float(v) for k, v in values.items()})
        self.trace.append(row)
        if self.progress_callback:
            self.progress_callback(row)

    @torch.no_grad()
    def _seed_experts(self, source, target, tokens, mask, source_groups, target_groups):
        # Pure expression OT on training cells; centres are not fate labels.
        from sklearn.cluster import KMeans
        z = self.model.encode_state(source)
        rates = []
        for label in sorted(source_groups):
            si, ti = self._draw(source_groups[label]), self._draw(target_groups[label])
            cost = torch.cdist(z[si], target[ti]).square()
            plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=self.config.sinkhorn_epsilon)
            draw = torch.multinomial(plan.flatten(), self.config.batch_size, replacement=True, generator=self.generator)
            rates.append((target[ti][draw % len(ti)]-z[si][draw//len(ti)])/(self.config.time_end-self.config.time_start))
        all_rates = torch.cat(rates)
        if len(all_rates) < self.model.config.max_experts:
            raise ValueError('Not enough training displacement samples to seed all experts')
        centres = torch.as_tensor(KMeans(n_clusters=self.model.config.max_experts, n_init=10,
                    random_state=self.config.seed).fit(all_rates.cpu().numpy()).cluster_centers_, device=z.device, dtype=z.dtype)
        ix = torch.arange(min(256, len(z)), device=z.device)
        e = self.model.condition_encoder(tokens[ix], mask[ix])
        field = self.model.field
        t = z.new_full((len(ix),), self.config.time_start)
        h = field.blocks(field.input(torch.cat((z[ix], field.time(t), e), -1)))
        base = field.base(h).mean(0)
        for k, expert in enumerate(field.experts):
            activation = F.gelu(expert.down(h)).mean(0)
            expert.up.weight.copy_(torch.outer(centres[k]-base, activation)/activation.square().sum().clamp_min(1e-8))

    def _distribution_step(self, step, spliced, unspliced, target, tokens, mask, sg, tg, labels, model, device):
        c = self.config
        label = labels[step % len(labels)]
        n_src = min(len(sg[label]), c.dist_source_batch)
        n_tgt = min(len(tg[label]), c.dist_target_batch)
        si = sg[label][torch.randint(len(sg[label]), (n_src,), generator=self.generator, device=device)]
        ti = tg[label][torch.randint(len(tg[label]), (n_tgt,), generator=self.generator, device=device)]
        background = None
        if self.velocity_background is not None:
            background = (self.velocity_background['r'][si], self.velocity_background['v'][si])
        context = model.make_context(spliced[si], unspliced[si], tokens[si], mask[si],
                                     background=background, time_start=c.time_start)
        # Capture the source-native loss before any calibration forward pass.
        native_loss = (model.dynamics.last_native_loss if model.config.joint_dynamics
                       else context.z0.new_zeros(()))
        p = context.probabilities
        active = [k for k in range(model.config.max_experts) if bool(model.active_modes[k])]
        endpoints = []
        for k in active:
            modes = torch.full((len(si),), k, dtype=torch.long, device=device)
            pred = model.integrate_context(context, t0=c.time_start, t1=c.time_end, max_step=.25,
                modes=modes, stochastic=model.config.use_noise, generator=self.generator)
            endpoints.append(pred.endpoint)
        # Score the weighted distribution of expert endpoints, not their barycentre.
        energy = mixture_energy(torch.stack(endpoints, 1), p[:, active], target[ti], detach_candidates=False)
        entropy = -(p*p.clamp_min(1e-8).log()).sum(-1).mean()
        loss = energy - c.dist_entropy_weight*entropy+c.gfg_native_weight*native_loss
        groups = [(self.router_optimizer, self.router_parameters),
                  (self.field_optimizer, self.field_parameters)]
        if model.config.use_noise:
            groups.append((self.noise_optimizer, [model.noise_raw]))
        task_gradient = 0.
        if model.config.joint_dynamics and model.config.use_dynamics:
            parameter = model.dynamics.core.velocity_encoder.net[-1].weight
            gradient = torch.autograd.grad(energy, parameter, retain_graph=True, allow_unused=True)[0]
            task_gradient = 0. if gradient is None else float(gradient.detach().norm())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite distribution loss')
        for opt, _ in groups:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt, params in groups:
            parameters = list(params)
            if any(pp.grad is not None and not torch.isfinite(pp.grad).all() for pp in parameters):
                raise FloatingPointError('Nonfinite full-model gradient')
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            opt.step()
        gate_loss = self._calibrate_gate(step, spliced[si], unspliced[si], tokens[si], mask[si], background=background)
        entropy_log = entropy.detach()
        self._log('C-dist', step, energy=energy.detach(), mode_entropy=entropy_log,
                  gate_loss=gate_loss, gfg_native_loss=native_loss.detach(), gfg_task_gradient_norm=task_gradient,
                  mean_gate=model.gate_value(context, c.time_start).mean().detach())
        self.completed_c_steps = step+1
        return None

    def _calibrate_gate(self, step, spliced, unspliced, tokens, mask, *, background=None):
        model, c = self.model, self.config
        if not (model.config.use_gate and model.config.use_dynamics and (step+1) % c.corruption_interval == 0):
            return 0.
        method = ('shift', 'gaussian', 'zero_u')[self.corruptions % 3]
        with torch.no_grad():
            clean = model.make_context(spliced, unspliced, tokens, mask, background=background, time_start=c.time_start)
            if model.config.dynamics_backend == 'gfg':
                bad_u, bad_v = model.dynamics.corrupt(unspliced, method, self.generator), None
            else:
                bad_u, bad_v = corrupt_source(spliced, unspliced, clean.velocity, method, generator=self.generator)
            bad = model.make_context(spliced, bad_u, tokens, mask, velocity_override=bad_v,
                                     background=background, time_start=c.time_start)
        features = torch.cat((clean.gate_features, bad.gate_features))
        trusted = torch.cat((clean.rho.new_ones(len(spliced)), clean.rho.new_zeros(len(spliced))))
        loss = model.gate.calibration_loss(features, trusted)
        self._step(self.gate_optimizer, loss, model.gate.parameters())
        self.corruptions += 1
        return float(loss.detach())

    @torch.no_grad()
    def _review_modes(self, step, spliced, unspliced, tokens, mask, source_conditions, labels):
        model, c = self.model, self.config
        if not (model.config.use_adaptive_modes and (step+1) % c.usage_interval == 0):
            return
        probabilities = []
        batch = getattr(model.dynamics, 'reference_batch_size', 256)
        for start in range(0, len(spliced), batch):
            ix = slice(start, start+batch)
            bg = None if self.velocity_background is None else (
                self.velocity_background['r'][ix], self.velocity_background['v'][ix])
            probabilities.append(model.make_context(spliced[ix], unspliced[ix], tokens[ix], mask[ix],
                                  background=bg, time_start=c.time_start).probabilities)
        if model.mode_controller.review(torch.cat(probabilities), source_conditions, labels):
            model.freeze_retired_experts()
            self.cache.clear()

    def fit(self, spliced, unspliced, reference_velocity, target, tokens, mask, *, source_conditions,
            target_conditions, source_ids, target_ids, scope, checkpoint_callback=None, max_c_steps=None):
        c, model = self.config, self.model
        scope.validate(source_ids)
        scope.validate(target_ids)
        if set(source_ids) & set(target_ids):
            raise ValueError('Training source and target identities overlap')
        if set(source_conditions) != set(target_conditions):
            raise ValueError('Full training source/target condition coverage differs')
        if len(source_ids) != len(spliced) or len(target_ids) != len(target) or len(source_conditions) != len(spliced):
            raise ValueError('Full training metadata/count alignment mismatch')
        if self.velocity_background is not None and list(source_ids) != self.velocity_background['cell_ids'].tolist():
            raise ValueError('Velocity background/source cell order mismatch')
        binding = object_hash({'source': sorted(source_ids), 'target': sorted(target_ids), 'source_conditions': list(source_conditions),
                               'target_conditions': list(target_conditions)})
        if self.data_hash is not None and self.data_hash != binding:
            raise ValueError('Cannot resume trainer on different cell IDs/conditions')
        self.data_hash = binding
        device = spliced.device
        labels = sorted(set(source_conditions))
        sg = {label: torch.tensor([i for i, x in enumerate(source_conditions) if x == label], device=device) for label in labels}
        tg = {label: torch.tensor([i for i, x in enumerate(target_conditions) if x == label], device=device) for label in labels}
        all_source = torch.arange(len(spliced), device=device)
        if not self.stage_a_done:
            for step in range(c.stage_a_steps):
                si = self._draw(all_source)
                loss = model.state_encoder.reconstruction_loss(spliced[si], mask_ratio=c.mask_ratio, generator=self.generator)
                # A static arm can retain the GFG module for checkpoint shape
                # matching, but it must not train or read U/S through it.
                if model.config.use_dynamics:
                    loss = loss+model.dynamics.reference_loss(spliced[si], unspliced[si], reference_velocity[si])
                self._step(self.encoder_optimizer, loss, list(model.state_encoder.parameters())+list(model.dynamics.parameters()))
                self._log('A', step, encoder_loss=loss.detach())
            model.freeze_encoders()
            self.target_encoder.load_state_dict(model.state_encoder.state_dict())
            self.stage_a_done = True
        if not self.stage_b_done:
            for step in range(c.stage_b_steps):
                label = labels[step % len(labels)]
                si, ti = self._draw(sg[label]), self._draw(tg[label])
                with torch.no_grad():
                    z = model.encode_state(spliced[si])
                    cost = torch.cdist(z, target[ti]).square()
                    plan = sinkhorn_log(cost/cost.mean().clamp_min(1e-8), epsilon=c.sinkhorn_epsilon)
                    draw = torch.multinomial(plan.flatten(), c.batch_size, replacement=True, generator=self.generator)
                    ii, jj = draw//len(ti), draw % len(ti)
                    fraction = torch.rand((len(ii), 1), device=device, generator=self.generator)
                    points = (1-fraction)*z[ii]+fraction*target[ti][jj]
                    rate = (target[ti][jj]-z[ii])/(c.time_end-c.time_start)
                e = model.condition_encoder(tokens[si][ii], mask[si][ii])
                prediction = base_rate(model.field, points, c.time_start+fraction[:, 0]*(c.time_end-c.time_start), e)
                loss = (prediction-rate).square().mean()
                self._step(self.field_optimizer, loss, self.field_parameters)
                self._log('B', step, field_loss=loss.detach())
            model.snapshot_static()
            self._seed_experts(spliced, target, tokens, mask, sg, tg)
            self.stage_b_done = True
        if not len(model.reference.positions):
            model.refresh_reference(spliced, unspliced, tokens, mask, cell_ids=source_ids, scope=scope)
        stop = c.stage_c_steps if max_c_steps is None else min(c.stage_c_steps, self.completed_c_steps+max_c_steps)
        for step in range(self.completed_c_steps, stop):
            # Cosine learning-rate decay only follows the fixed declared budget.
            multiplier = .1+.9*(1+math.cos(math.pi*step/c.stage_c_steps))/2
            for name, optimizer in self.optimizers.items():
                if name not in {'encoder', 'gate'}:
                    for group in optimizer.param_groups:
                        group['lr'] = group.get('base_lr', c.learning_rate)*multiplier
            if model.config.use_adaptive_modes and step % c.activation_interval == 0:
                if model.mode_controller.activate_next() is not None:
                    self.cache.clear()
            if step and step % c.reference_interval == 0:
                model.refresh_reference(spliced, unspliced, tokens, mask, cell_ids=source_ids, scope=scope)
            if c.coupling_mode == 'distribution_matching':
                self._distribution_step(step, spliced, unspliced, target, tokens, mask, sg, tg, labels, model, device)
                self._review_modes(step, spliced, unspliced, tokens, mask, source_conditions, labels)
                if checkpoint_callback and self.completed_c_steps % c.checkpoint_interval == 0:
                    checkpoint_callback(self)
                continue
            label = labels[step % len(labels)]
            teacher_loss = 0.
            if label not in self.cache or step-self.cache[label]['step'] >= c.e_interval:
                si, ti = self._draw(sg[label]), self._draw(tg[label])
                with torch.no_grad():
                    context = model.make_context(spliced[si], unspliced[si], tokens[si], mask[si], time_start=c.time_start)
                    result = full_responsibilities(model, context, target[ti], c,
                        teacher=self.teacher if self.teacher_updates else None, target_encoder=self.target_encoder)
                old = self.history.push(result['record'])
                if old is not None:
                    logits = self.teacher(*[old[key] for key in ('z', 'r', 'e', 'h')])
                    logp = logits[:, old['active']].log_softmax(-1)
                    per_pair = -(old['labels'][:, old['active']]*logp).sum(-1)
                    teacher_loss = (per_pair*old['weight']).sum()/old['weight'].sum().clamp_min(1e-12)
                    self._step(self.teacher_optimizer, teacher_loss, self.teacher.parameters())
                    teacher_loss = float(teacher_loss.detach())
                    self.teacher_updates += 1
                self.e_round += 1
                self.cache[label] = {'step': step, 'si': si, 'ti': ti, 'gamma': result['gamma'], 'labels': result['labels']}
            cached = self.cache[label]
            si, ti, gamma = cached['si'], cached['ti'], cached['gamma']
            context = model.make_context(spliced[si], unspliced[si], tokens[si], mask[si], time_start=c.time_start)
            draw = torch.multinomial(gamma.flatten(), c.batch_size, replacement=True, generator=self.generator)
            kk = draw % model.config.max_experts
            jj = (draw//model.config.max_experts) % len(ti)
            ii = draw//(model.config.max_experts*len(ti))
            z0, z1 = context.z0[ii], target[ti][jj].detach()
            fraction = torch.rand((len(ii), 1), device=device, generator=self.generator)
            points = ((1-fraction)*z0+fraction*z1).detach()
            time = c.time_start+fraction[:, 0]*(c.time_end-c.time_start)
            true_rate = ((z1-z0)/(c.time_end-c.time_start)).detach()
            prediction = model.field.selected(points, time, context.condition[ii], kk)
            if model.config.use_intrinsic:
                prediction = prediction+model.intrinsic(points, z0, context.velocity[ii])
            fm = (prediction-true_rate).square().mean()
            along = field_consistency_loss(model, z0, z1, context.condition[ii], c) if c.lambda_field else fm.detach()*0
            self._step(self.field_optimizer, fm+c.lambda_field*along, self.field_parameters)
            logits = model.router.logits(context.z0, context.representation, context.condition.detach())
            active = model.active_modes
            ce = -(cached['labels'][:, active]*logits[:, active].log_softmax(-1)).sum(-1).mean()
            regularizer = mode_regularization(logits, active)
            route_loss = ce+c.mode_regularization_weight*regularizer
            native, task_gradient = 0., 0.
            if model.config.joint_dynamics:
                native_loss = model.dynamics.last_native_loss
                if model.config.use_dynamics:
                    parameter = model.dynamics.core.velocity_encoder.net[-1].weight
                    gradient = torch.autograd.grad(ce, parameter, retain_graph=True, allow_unused=True)[0]
                    task_gradient = 0. if gradient is None else float(gradient.detach().norm())
                route_loss = route_loss+c.gfg_native_weight*native_loss
                native = float(native_loss.detach())
            self._step(self.router_optimizer, route_loss, self.router_parameters)
            noise_loss = 0.
            if model.config.use_noise:
                sigma = model.noise_scale()[kk]
                residual = (true_rate-prediction).detach()
                noise_loss = (.5*(residual/sigma).square()+sigma.log()).mean()
                self._step(self.noise_optimizer, noise_loss, [model.noise_raw])
                noise_loss = float(noise_loss.detach())
            gate_loss = self._calibrate_gate(step, spliced[si], unspliced[si], tokens[si], mask[si])
            self._review_modes(step, spliced, unspliced, tokens, mask, source_conditions, labels)
            entropy = -(context.probabilities*context.probabilities.clamp_min(1e-8).log()).sum(-1).mean()
            self._log('C+D', step, field_loss=fm.detach(), along_path_loss=along.detach(), router_loss=ce.detach(),
                      teacher_loss=teacher_loss, noise_loss=noise_loss, gate_loss=gate_loss,
                      mean_gate=model.gate_value(context, c.time_start).mean().detach(), mode_entropy=entropy.detach(),
                      gfg_native_loss=native, gfg_task_gradient_norm=task_gradient)
            self.completed_c_steps = step+1
            if checkpoint_callback and self.completed_c_steps % c.checkpoint_interval == 0:
                checkpoint_callback(self)
        return self.trace

    def state_dict(self):
        return {'format': 'veloroute_full_training_v1', 'config': asdict(self.config), 'data_hash': self.data_hash,
                'teacher': self.teacher.state_dict(), 'target_encoder': self.target_encoder.state_dict(),
                'history': list(self.history.queue), 'cache': self.cache, 'trace': self.trace,
                'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
                'generator_state': self.generator.get_state(), 'stage_a_done': self.stage_a_done,
                'stage_b_done': self.stage_b_done, 'completed_c_steps': self.completed_c_steps,
                'e_round': self.e_round, 'teacher_updates': self.teacher_updates, 'corruptions': self.corruptions}

    def load_state_dict(self, state):
        if state.get('format') != 'veloroute_full_training_v1' or state['config'] != asdict(self.config):
            raise ValueError('Resume training config/format mismatch')
        self.teacher.load_state_dict(state['teacher'])
        self.target_encoder.load_state_dict(state['target_encoder'])
        self.history.queue = deque(state['history'])
        self.cache, self.trace, self.data_hash = state['cache'], state['trace'], state['data_hash']
        self.generator.set_state(state['generator_state'].cpu())
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state['optimizers'][name])
        for name in ('stage_a_done', 'stage_b_done', 'completed_c_steps', 'e_round', 'teacher_updates', 'corruptions'):
            setattr(self, name, state[name])
        if self.stage_a_done:
            self.model.freeze_encoders()
