"""Training-only source/target/expert responsibilities; never treated as fate truth."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def sinkhorn_log(cost, *, epsilon=.1, iterations=200):
    if cost.ndim != 2 or min(cost.shape) == 0 or not torch.isfinite(cost).all() or epsilon <= 0 or iterations < 1:
        raise ValueError("Invalid Sinkhorn problem")
    n, m = cost.shape
    log_kernel = -cost / epsilon
    log_a = cost.new_full((n,), -math.log(n))
    log_b = cost.new_full((m,), -math.log(m))
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iterations):
        u = log_a - torch.logsumexp(log_kernel+v[None, :], dim=1)
        v = log_b - torch.logsumexp(log_kernel+u[:, None], dim=0)
    plan = (log_kernel+u[:, None]+v[None, :]).exp()
    return plan


def source_mode_cost(velocity, fields, *, eps=1e-8):
    """k-dependent direction compatibility; dimensionless, not a velocity-rate target."""
    if fields.ndim != 3 or velocity.shape != (fields.shape[0], fields.shape[2]):
        raise ValueError("Source velocity and candidate fields must share an audited latent coordinate system")
    if not torch.isfinite(velocity).all() or not torch.isfinite(fields).all():
        raise ValueError("Nonfinite compatibility inputs")
    valid = (velocity.norm(dim=-1) > eps)[:, None] & (fields.norm(dim=-1) > eps)
    cos = (F.normalize(velocity, dim=-1, eps=eps)[:, None, :]*F.normalize(fields, dim=-1, eps=eps)).sum(-1)
    return torch.where(valid, 1-cos.clamp(-1, 1), torch.zeros_like(cos))


@torch.no_grad()
def build_responsibilities(model, z0, z1, velocity0, condition, *, source_conditions, target_conditions,
                           t0=4., t1=5., lambda_dynamic=0., lambda_pair_dynamic=0., temperature=.5, epsilon=.1, iterations=200):
    if len(source_conditions) != len(z0) or len(target_conditions) != len(z1):
        raise ValueError("Condition metadata dimensions mismatch")
    if len(set(source_conditions)) != 1 or set(source_conditions) != set(target_conditions):
        raise ValueError("OT batches must contain one shared condition; cross-condition pairing forbidden")
    if condition.ndim != 2 or len(condition) != len(z0) or not torch.allclose(condition, condition[:1].expand_as(condition)):
        raise ValueError("One-condition OT batch must use one consistent condition embedding")
    if t1 <= t0 or temperature <= 0 or lambda_dynamic < 0 or lambda_pair_dynamic < 0:
        raise ValueError("Invalid coupling time/temperature/weight")
    if model.config.state_dim != model.config.velocity_dim and (lambda_dynamic or lambda_pair_dynamic):
        raise ValueError("Directional costs need explicit equal-coordinate state/velocity projection")
    fields = model.field(z0, t0, condition)
    predicted = z0[:, None, :] + (t1-t0)*fields
    # i,j,k: Euler endpoint errors are training costs, not held-out evaluation.
    expression_cost = (predicted[:, None, :, :]-z1[None, :, None, :]).square().mean(-1)
    scale = expression_cost.mean().clamp_min(1e-8)
    cost = expression_cost / scale
    if lambda_dynamic:
        cost = cost + lambda_dynamic*source_mode_cost(velocity0, fields)[:, None, :]
    if lambda_pair_dynamic:
        # Unlike an i-only term (which cancels in balanced Sinkhorn for K=1),
        # this source-velocity/displacement compatibility genuinely varies with j.
        rates = (z1[None, :, :]-z0[:, None, :])/(t1-t0)
        cost = cost + lambda_pair_dynamic*source_mode_cost(velocity0, rates)[:, :, None]
    pair_cost = -temperature*(torch.logsumexp(-cost/temperature, -1)-math.log(model.config.n_experts))
    plan = sinkhorn_log(pair_cost, epsilon=epsilon, iterations=iterations)
    row_error = (plan.sum(1)-1/len(z0)).abs().max()
    col_error = (plan.sum(0)-1/len(z1)).abs().max()
    if max(float(row_error), float(col_error)) > 1e-3:
        raise RuntimeError("Sinkhorn did not converge; no responsibility labels emitted")
    gamma = plan[:, :, None]*(-cost/temperature).softmax(-1)
    source_weights = gamma.sum(1)
    source_weights /= source_weights.sum(-1, keepdim=True).clamp_min(1e-12)
    return {"gamma": gamma, "source_responsibilities": source_weights, "cost": cost,
            "row_marginal_error": float(row_error), "column_marginal_error": float(col_error)}
