"""Source-only L3 Tsit5 candidates on exact frozen W13 common IDs.

All-source q is computed before the canonical evaluation subsample. Experts see
only S-state conditions, so cohort/uniform/U interventions change mixture weights
and reuse the exact same per-expert trajectories (not a mixed-field shortcut).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from w4_artifacts import sha256
from w15a_cellflow_adapter import add_cellflow_paths, time_schedule
add_cellflow_paths()

import jax
import jax.numpy as jnp
from flax import serialization

from w15a_cellflow_common_evaluation import (DEFAULT_REFERENCE, KEYS, align_source_ids,
                                            reference_sources, save_common_predictions)
from w15a_cellflow_train import load_npz
from w15a_cellflow_moe_real_train import (create_model, read_feature_cohort, read_feature_normalizer,
    require_complete_synthetic_gate, sidecar_metadata, source_state_conditions, velocity_input)


DIAGNOSTICS = ("cohort_q", "uniform_q", "Umean", "Uzero")


def make_candidate_integrator(model):
    """Fixed expert trajectories, native Tsit5/PID tolerances inherited from L2."""
    import diffrax
    solvers = []
    for expert in range(model.config.experts):
        def one(params, source, condition, tau_end, expert=expert):
            condition = {key:value[None] for key,value in condition.items()}
            def field(tau, state, parameters):
                return model.apply(dict(params=parameters), state[None], tau, condition,
                                    method=model.velocities)[0, expert]
            return diffrax.diffeqsolve(diffrax.ODETerm(field), solver=diffrax.Tsit5(),
                t0=0., t1=tau_end, dt0=None, y0=source, args=params,
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5)).ys[0]
        solvers.append(jax.jit(jax.vmap(one, in_axes=(None,0,0,None))))
    def integrate(params, source, condition, tau_end):
        if tau_end == 0:
            result = jnp.repeat(source[:,None], model.config.experts, axis=1)
        else:
            result = jnp.stack([solve(params,source,condition,jnp.asarray(tau_end)) for solve in solvers], axis=1)
        value = np.asarray(result)
        if not np.isfinite(value).all():
            raise FloatingPointError("Nonfinite independent L3 expert trajectory")
        return value
    return integrate


def canonical_full_source(reference_root, start):
    """Exact W13 state for EVERY source cell, not a float-reprojected replacement."""
    study = json.loads((Path(reference_root)/"study_manifest.json").read_text())
    pm = json.loads((Path(study["pack"])/"manifest.json").read_text())
    matches = [row for row in pm["prediction_groups"] if int(row["source_hour"]) == start]
    if len(matches) != 1 or sha256(matches[0]["path"]) != matches[0]["sha256"]:
        raise ValueError("W13 full source state identity/hash mismatch")
    data = load_npz(matches[0]["path"])
    return np.asarray(data["cell_ids"],dtype=str), data["z"]


def _embedding_cosine(original, changed):
    original, changed = [np.asarray(value,dtype="float64") for value in (original,changed)]
    denominator = np.linalg.norm(original,axis=1)*np.linalg.norm(changed,axis=1)
    valid = denominator > 0
    if not valid.any():
        return None, 0
    cosine = np.sum(original[valid]*changed[valid],axis=1)/denominator[valid]
    return float(np.median(cosine)), int(valid.sum())


def forecasts_from_parameters(model, params, *, sidecar, variant, reference_root,
                              include_diagnostics=True):
    """Pure source-side forecasting; no target cache is opened by this function."""
    reference = reference_sources(reference_root)
    manifest = sidecar_metadata(sidecar)
    normalizer = read_feature_normalizer(sidecar)
    integrate = make_candidate_integrator(model)
    contexts, audits = {}, []
    for start in (5,10,15):
        matches = [row for row in manifest["development_sources"] if float(row["time_hours"]) == start]
        if len(matches) != 1:
            raise ValueError("Exactly one complete source sidecar is required per W13 origin")
        row = matches[0]
        pack, _ = read_feature_cohort(sidecar,row["source_key"])
        common = reference["sources"][start]
        if set(pack["cell_ids"].tolist()) != set(common["all_cell_ids"].tolist()):
            raise ValueError("L3 full source cohort differs from the W13 canonical cohort")
        ids, all_z = canonical_full_source(reference_root,start)
        full_order = align_source_ids(pack["cell_ids"], ids)
        full_z = np.asarray(all_z[full_order],dtype="float32")
        np.testing.assert_allclose(pack["z"],full_z,rtol=1e-6,atol=2e-5)
        selected = align_source_ids(common["cell_ids"],pack["cell_ids"])
        np.testing.assert_array_equal(full_z[selected],np.asarray(common["z"],dtype="float32"))
        kinetics = velocity_input(pack,normalizer,variant,row["source_key"])
        q = np.asarray(model.apply(dict(params=params),jnp.asarray(full_z),jnp.asarray(kinetics),method=model.route))
        weights = dict(main=q)
        if variant == "moe_kin" and include_diagnostics:
            # mean over ALL cells, not over the terminal common-ID sample.
            weights["cohort_q"] = np.broadcast_to(q.mean(axis=0),q.shape)
            weights["uniform_q"] = np.full_like(q,1/q.shape[1])
            original_embedding = model.apply(dict(params=params),jnp.asarray(kinetics),method=model.encode_kinetics)
            for intervention in ("Umean","Uzero"):
                modified = velocity_input(pack,normalizer,variant,row["source_key"],intervention=intervention)
                changed_q = np.asarray(model.apply(dict(params=params),jnp.asarray(full_z),jnp.asarray(modified),method=model.route))
                weights[intervention] = changed_q
                embedding = model.apply(dict(params=params),jnp.asarray(modified),method=model.encode_kinetics)
                cosine, valid = _embedding_cosine(original_embedding,embedding)
                tv = float(np.abs(q-changed_q).sum(axis=1).mean()/2)
                audits.append(dict(source_hours=start,intervention=intervention,full_source_cells=len(q),
                    median_embedding_cosine=cosine,valid_embedding_cosine_cells=valid,
                    full_source_mean_q_TV=tv,weak_sensitivity_to_this_current_U_intervention=tv<.01,
                    only_current_source_U_replaced=True,original_observed_reference_fixed=True))
        contexts[start] = dict(x=np.asarray(common["z"],dtype="float32"),source_ids=common["cell_ids"],
            selected=selected,q_by_intervention={name:weight[selected] for name,weight in weights.items()},
            full_source_population_q=q.mean(axis=0).tolist(),full_source_cells=len(q))
    forecasts = {"main":{}}
    if variant == "moe_kin" and include_diagnostics:
        forecasts.update({name:{} for name in DIAGNOSTICS})
    for key,(start,stop,clock) in KEYS.items():
        context = contexts[start]
        schedule = time_schedule(stop-start,kernel=clock)
        condition = source_state_conditions(len(context["x"]),start,schedule["condition_delta_hours"])
        candidates = integrate(params,jnp.asarray(context["x"]),condition,schedule["tau_end"])
        for intervention in forecasts:
            forecasts[intervention][key] = dict(x=candidates,
                q=context["q_by_intervention"][intervention],source_ids=context["source_ids"])
    return forecasts,dict(U_sensitivity=audits,
        source_contexts={str(start):{key:value for key,value in context.items()
            if key in ("full_source_cells","full_source_population_q")} for start,context in contexts.items()},
        source_routing_fixed=True,cohort_q_uses_full_source_before_common_ID_selection=True,
        interventions_reuse_identical_S_only_expert_trajectories=True,target_caches_read=False,
        global_U_independence_established=False,inference_without_U_established=False)


def predict_real(training, output, *, reference_root=DEFAULT_REFERENCE,
                  engineering_fixture=False, include_diagnostics=True):
    training, output = Path(training),Path(output)
    summary = json.loads((training/"summary.json").read_text())
    if engineering_fixture:
        if summary.get("engineering_fixture") is not True or summary.get("status") != "CPU_SYNTHETIC_L3_FIXTURE_TRAINED":
            raise ValueError("Only explicit CPU synthetic training fixtures may bypass the real-data gate")
        if any(device.platform!="cpu" for device in jax.devices()):
            raise ValueError("Engineering fixture prediction is CPU-only")
    else:
        if summary.get("engineering_fixture") or summary.get("status") != "L3_REAL_DATA_TRAINED":
            raise ValueError("A gated real-data L3 checkpoint is required")
        checked = require_complete_synthetic_gate(summary["synthetic_gate"]["directory"])
        if checked != summary["synthetic_gate"]:
            raise ValueError("Synthetic training prerequisite changed after the real fit")
    sidecar = Path(summary["sidecar_directory"])
    if sha256(training/"model.msgpack")!=summary["model_sha256"]:
        raise ValueError("Frozen L3 checkpoint changed")
    if (sha256(sidecar/"manifest.json")!=summary["sidecar_manifest_sha256"] or
            sha256(sidecar/"normalizer.npz")!=summary["sidecar_normalizer_sha256"]):
        raise ValueError("Frozen source features or training-only normalization changed")
    model = create_model(summary["config"],summary["dimensions"],summary["seed"])
    dummy = jnp.zeros((1,summary["dimensions"]))
    template = model.init(jax.random.PRNGKey(summary["seed"]),dummy,dummy,dummy,.5,source_state_conditions(1,0.))["params"]
    params = serialization.from_bytes(template,(training/"model.msgpack").read_bytes())
    forecasts,diagnostics = forecasts_from_parameters(model,params,sidecar=sidecar,variant=summary["variant"],
        reference_root=reference_root,include_diagnostics=include_diagnostics)
    exports = []
    for intervention,forecast in forecasts.items():
        path = output if intervention=="main" else output/intervention
        arm = summary["variant"] if intervention=="main" else f"moe_kin_{intervention}"
        result = save_common_predictions(path,forecast,reference_root=reference_root,arm=arm,seed=summary["seed"],
            model_family="CellFlow_L3",extra=dict(training=str(training.resolve()),
                model_sha256=summary["model_sha256"],sidecar_manifest_sha256=summary["sidecar_manifest_sha256"],
                intervention=intervention,fixed_source_routing=True,source_context="all_source_before_common_ID_selection",
                engineering_fixture=engineering_fixture))
        exports.append(dict(arm=arm,path=str(path.resolve()),predictions_sha256=result["predictions_sha256"]))
    (output/"l3_inference_diagnostics.json").write_text(json.dumps(diagnostics,indent=2,allow_nan=False)+"\n")
    result = dict(status="L3_COMMON_PREDICTIONS_FROZEN",variant=summary["variant"],seed=summary["seed"],
        exports=exports,windows_per_export=12,real_data_training_started_by_this_runner=False,
        target_read=False,engineering_fixture=engineering_fixture)
    (output/"l3_exports.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--training",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--reference",default=str(DEFAULT_REFERENCE))
    args=parser.parse_args()
    print(json.dumps(predict_real(args.training,args.output,reference_root=args.reference),indent=2))
