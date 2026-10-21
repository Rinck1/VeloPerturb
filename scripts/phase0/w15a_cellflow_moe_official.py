"""Unmodified official CellFlow hidden features for the L3 mixture wrapper.

Only the final shared projection is removed via a project-local subclass. All
official encoders, conditioning and decoder implementations remain inherited.
This module is an integration/sanity layer, not permission to train a real grid
before the L2 cf_static hyperparameters have been frozen.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import hashlib
from pathlib import Path

from w15a_cellflow_adapter import add_cellflow_paths

add_cellflow_paths()

from cellflow.networks import ConditionalVelocityField
import flax.linen as nn
import jax
import jax.numpy as jnp

from w15a_cellflow_moe import (KineticMoE, MoEConfig, align_ot_indices_jit,
                              paired_flow_matching_loss, state_condition_only)


ALLOWED_STATE_CONDITIONS = frozenset(("perturbation", "bio_time", "bio_delta"))


class HiddenFeatureField(ConditionalVelocityField):
    """Return the native decoder's hidden features instead of output_layer(h).

    This is the only overridden official computation. The final output Dense is
    still declared by inherited setup, but absent from initialized parameters
    because it is never called. L3 replaces that projection with expert heads.
    """

    def _combine_and_decode(self, t_encoded, x_encoded, cond_embedding, squeeze,
                            train, x_0_encoded=None):
        concat_inputs, conditioning_vec = self._conditioning_signals(
            t_encoded, x_encoded, cond_embedding, x_0_encoded)
        if self.conditioning == "concatenation":
            hidden = jnp.concatenate(concat_inputs, axis=-1)
        elif self.conditioning == "film":
            hidden = self.film_block(x_encoded, conditioning_vec)
        elif self.conditioning == "resnet":
            hidden = self.resnet_block(x_encoded, conditioning_vec)
        else:
            raise ValueError(f"Unknown conditioning mode: {self.conditioning}")
        return self.decoder(hidden, training=train)


def hidden_field_from_native(native_field):
    """Copy configuration, NOT weights, from the actual official native module."""
    if type(native_field) is not ConditionalVelocityField:
        raise TypeError("L3 integration currently accepts plain official ConditionalVelocityField only")
    if native_field.condition_mode != "deterministic":
        raise ValueError("L3 wrapper currently requires deterministic CellFlow conditions")
    dropout_fields = ("cond_output_dropout", "condition_dropout_prob", "time_encoder_dropout",
                      "hidden_dropout", "decoder_dropout")
    if any(float(getattr(native_field, name)) != 0 for name in dropout_fields):
        raise ValueError("Nonzero dropout requires an explicitly audited L3 train/inference RNG interface")
    kwargs = {field.name: getattr(native_field, field.name)
              for field in fields(ConditionalVelocityField)
              if field.init and field.name not in ("parent", "name")}
    return HiddenFeatureField(**kwargs)


class OfficialCellFlowBackbone(nn.Module):
    """Official hidden field behind the S-only L3 backbone contract.

    L2 cf_static has two trainable token MLP slots with constant zero input. Those
    same zero slots are inserted INTERNALLY to inherit its condition architecture;
    callers cannot supply velocity/U through them. This is not kinetic injection.
    """

    field: HiddenFeatureField
    zero_condition_slots: tuple[str, ...] = ("population_token", "cell_token")

    def state_conditions(self, x, condition):
        state_condition_only(condition)
        if not isinstance(condition, Mapping) or set(condition) != ALLOWED_STATE_CONDITIONS:
            raise ValueError("L3 official state conditions must be exactly perturbation/bio_time/bio_delta")
        batch = x.shape[0]
        result = {}
        for key in ("perturbation", "bio_time", "bio_delta"):
            value = jnp.asarray(condition[key])
            if value.ndim != 3 or value.shape[0] != batch:
                raise ValueError("Official condition leaves must be [batch, set_size, features]")
            result[key] = value
        set_size = result["perturbation"].shape[1]
        for key in self.zero_condition_slots:
            result[key] = jnp.zeros((batch, set_size, self.field.output_dim), dtype=x.dtype)
        return result

    def __call__(self, x, tau, condition):
        conditions = self.state_conditions(x, condition)
        noise = jnp.zeros((x.shape[0], self.field.condition_embedding_dim), dtype=x.dtype)
        hidden, _, _ = self.field(t=tau, x_t=x, cond=conditions,
                                  encoder_noise=noise, train=False)
        return hidden


def official_moe_from_native(native_field, *, config=MoEConfig()):
    """Use L2's instantiated architecture, not a guessed parallel implementation."""
    if config.dimensions != native_field.output_dim:
        raise ValueError("L3 state dimensions must equal the frozen L2 CellFlow dimensions")
    hidden = hidden_field_from_native(native_field)
    layers = native_field.layers_before_pool
    slots = tuple(key for key in ("population_token", "cell_token")
                  if isinstance(layers, Mapping) and key in layers)
    backbone = OfficialCellFlowBackbone(hidden, zero_condition_slots=slots)
    return KineticMoE(backbone, config)


def make_official_ot_train_step(model, match_fn):
    """Compiled sanity/training primitive using OFFICIAL state-only OT sampling.

    No real-data schedule or hyperparameters are chosen here. Conditions and
    kappa follow the exact source indices drawn from the official coupling. The
    match function sees only source/target S-state, never model parameters.
    """
    from ott.solvers import utils as solver_utils

    @jax.jit
    def step(state, rng, source_x, target_x, kinetic_input, condition):
        key_pair, key_tau = jax.random.split(rng)
        coupling = match_fn(source_x, target_x)
        si, ti = solver_utils.sample_joint(key_pair, coupling)
        paired = align_ot_indices_jit(source_x, kinetic_input, target_x, si, ti,
                                      condition=condition)
        tau = jax.random.uniform(key_tau, (len(si), 1), minval=0., maxval=1.)

        def objective(params):
            return paired_flow_matching_loss(model, dict(params=params), paired, tau)

        (loss, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=gradients)
        gradient_norm = lambda tree: jnp.sqrt(sum(jnp.square(value).sum()
            for value in jax.tree_util.tree_leaves(tree)))
        return new_state, dict(metrics, loss=loss,
            kinetic_gradient_norm=gradient_norm(gradients["kinetic_encoder"]),
            router_gradient_norm=gradient_norm(gradients["router"]),
            source_indices=si, target_indices=ti)

    return step


def official_source_record():
    import cellflow.networks._velocity_field as source
    path = Path(source.__file__).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        override="project HiddenFeatureField._combine_and_decode: remove only final output_layer",
        upstream_source_modified=False, original_output_layer_in_shared_backbone=False)
