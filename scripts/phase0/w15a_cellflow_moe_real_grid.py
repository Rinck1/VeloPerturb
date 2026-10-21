"""Bounded deferred L3 grid: full synthetic gate + free GPU0, then 18 fits.

Only an approved controller launches workers. A failed synthetic gate terminates
without training or retuning. GPU0 trains/predicts/freezes; separate CPU evaluation
then consumes all 30 arm/seed manifests on the immutable W13 common targets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from w4_artifacts import Artifacts,sha256
from w15a_cellflow_adapter import ROOT
from w15a_cellflow_common_evaluation import freeze_common_grid,evaluate_common
from w15a_cellflow_moe_real_train import (VARIANTS,SEEDS,registered_jobs,
    require_complete_synthetic_gate,train_real)
from w15a_cellflow_moe_real_predict import DIAGNOSTICS,predict_real


PREDICTION_ARMS=VARIANTS+tuple(f"moe_kin_{name}" for name in DIAGNOSTICS)
HERE=Path(__file__).resolve().parent
RUNTIME=Path("/data/yuchang/veloroute_w15a/runtime")


def _read(path):
    return json.loads(Path(path).read_text())


def dependency_state(synthetic,l2_grid,*,synthetic_job=None,l2_job=None):
    """Read completion state only; never inspect partial training scores."""
    for name,path in (("synthetic",synthetic_job),("L2",l2_job)):
        if path is not None and (Path(path)/"status.json").exists():
            state=_read(Path(path)/"status.json")
            if state.get("status")=="failed":
                raise RuntimeError(f"Prerequisite {name} job failed; no automatic retry")
    summary=Path(synthetic)/"summary.json"
    if not summary.exists():
        return dict(status="waiting",reason="complete_120_fit_synthetic_summary_not_available")
    syn=_read(summary)
    if (syn.get("status")!="L3_POPULATION_SYNTHETIC_GRID_COMPLETE" or
            syn.get("completed_fits")!=120 or not syn.get("full_positive_null_grid_completed")):
        raise ValueError("Prerequisite summary is not the full registered synthetic experiment")
    if syn.get("population_composition_learning_gate_passed") is not True:
        return dict(status="do_not_train",reason="complete_synthetic_engineering_gate_did_not_pass",
                    synthetic_summary_sha256=sha256(summary))
    gate=require_complete_synthetic_gate(synthetic)
    l2_path=Path(l2_grid)/"summary.json"
    if not l2_path.exists():
        return dict(status="waiting",reason="L2_GPU0_registered_grid_not_complete")
    l2=_read(l2_path)
    if l2.get("status")!="REGISTERED_L2_DEVELOPMENT_GRID_COMPLETE" or l2.get("models")!=18:
        raise ValueError("L2 predecessor did not complete its fixed eighteen-model queue")
    if l2_job is not None:
        job=_read(Path(l2_job)/"status.json")
        if job.get("status")!="complete":
            return dict(status="waiting",reason="L2_GPU0_process_has_not_released_device")
    return dict(status="ready",synthetic_gate=gate,L2_summary_sha256=sha256(l2_path))


def launch_phase(script,arguments,job_directory,*,gpu=None):
    """Use the existing detached launcher; only physical GPU0 is supported here."""
    if gpu not in (None,0):
        raise ValueError("Deferred L3 controller owns GPU0 only")
    env=os.environ.copy()
    env.update(JAX_PLATFORMS="cpu" if gpu is None else "cuda",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",TMPDIR=str(RUNTIME/"tmp"),
        JAX_COMPILATION_CACHE_DIR=str(RUNTIME/("jax_l3_real_cpu_cache" if gpu is None else "jax_l3_real_gpu_cache")),
        XDG_CACHE_HOME=str(RUNTIME/"cache_l3_real"),
        LD_LIBRARY_PATH=str(RUNTIME/"cuda_deps/nvidia/cudnn/lib"),
        PYTHONPATH=str(ROOT/"src"))
    command=[sys.executable,str(HERE/"launch_gpu_job.py")]
    command+=(['--cpu'] if gpu is None else ['--gpu',str(gpu)])
    command+=['--script',str(script),'--output',str(job_directory),'--',*arguments]
    return json.loads(subprocess.check_output(command,cwd=ROOT,env=env,text=True))


def _wait_job(job_directory,deadline,poll_seconds):
    while True:
        path=Path(job_directory)/"status.json"
        if path.exists():
            state=_read(path)
            if state.get("status")=="complete":return state
            if state.get("status")=="failed":raise RuntimeError(f"L3 child job failed: {job_directory}")
        if time.monotonic()>deadline:raise TimeoutError("Bounded deferred L3 controller deadline exceeded")
        time.sleep(poll_seconds)


def worker_grid(configuration,freeze,sidecar,parent_pack,synthetic,reference,output,*,require_gpu=True):
    gate=require_complete_synthetic_gate(synthetic)
    if require_gpu:
        from veloroute.gpu_policy import enforce_gpu_policy
        enforce_gpu_policy("cuda")
    run=Artifacts(output,seed=list(SEEDS),inputs=[Path(configuration),Path(freeze),
        Path(sidecar)/"manifest.json",Path(reference)/"study_manifest.json",Path(reference)/"source_samples.json",
        Path(__file__),HERE/"w15a_cellflow_moe_real_train.py",HERE/"w15a_cellflow_moe_real_predict.py"])
    spec=dict(status="L3_FIXED_REAL_GRID_BOUND",jobs=registered_jobs(),models=18,seeds=list(SEEDS),
        training_variants=list(VARIANTS),prediction_arms=list(PREDICTION_ARMS),
        main_prediction_manifests=18,diagnostic_prediction_manifests=12,expected_manifests=30,
        expected_windows=360,synthetic_gate=gate,configuration=str(Path(configuration).resolve()),
        freeze_record=str(Path(freeze).resolve()),sidecar=str(Path(sidecar).resolve()),
        parent_pack=str(Path(parent_pack).resolve()),reference=str(Path(reference).resolve()),
        target_caches_opened=False,automatic_loss_or_hyperparameter_changes=False)
    run.json("study_manifest.json",spec)
    completed=[];paths=[]
    for job in spec["jobs"]:
        variant,seed=job["variant"],job["seed"]
        directory=run.directory/variant/f"seed{seed}"
        run.json("status.json",dict(status="training",variant=variant,current_seed=seed,completed_models=len(completed)))
        fitted=train_real(sidecar,parent_pack,configuration,freeze,synthetic,directory/"train",variant=variant,seed=seed)
        run.json("status.json",dict(status="source_only_prediction",variant=variant,current_seed=seed,completed_models=len(completed)))
        predicted=predict_real(directory/"train",directory/"common_predict",reference_root=reference)
        paths.extend(Path(item["path"]) for item in predicted["exports"])
        completed.append(dict(**job,training=str((directory/"train").resolve()),
            model_sha256=fitted["model_sha256"],exports=predicted["exports"]))
        run.json("completed_models.json",dict(models=completed,completed=len(completed),total=18))
    _,frozen=freeze_common_grid(paths,reference_root=reference,expected_seeds=SEEDS,expected_arms=PREDICTION_ARMS)
    run.json("whole_grid_frozen.json",dict(predictions=frozen,manifests=len(frozen),windows=360,
        all_expected_arms_and_seeds_frozen=True,target_caches_opened=False))
    result=dict(status="L3_REAL_GRID_PREDICTIONS_FROZEN",models=18,prediction_manifests=len(frozen),
        windows=360,prediction_directories=[str(path.resolve()) for path in paths],
        expected_arms=list(PREDICTION_ARMS),expected_seeds=list(SEEDS),
        GPU_training_and_prediction_complete=True,target_caches_opened=False,
        scientific_scores_computed=False,automatic_real_training_beyond_registered_grid=False)
    run.json("summary.json",result);run.json("status.json",dict(status="complete"))
    return result


def evaluate_grid(grid,output):
    grid=Path(grid);summary=_read(grid/"summary.json");spec=_read(grid/"study_manifest.json")
    if (summary.get("status")!="L3_REAL_GRID_PREDICTIONS_FROZEN" or summary.get("models")!=18 or
            summary.get("prediction_manifests")!=30 or not (grid/"whole_grid_frozen.json").is_file()):
        raise ValueError("Complete frozen 18-model/30-manifest L3 grid is required before target scoring")
    parent=Path(spec["parent_pack"]);manifest=_read(parent/"manifest.json")
    centers=parent/"training_target_mode_centers.npy"
    if sha256(centers)!=manifest["training_target_centers_sha256"]:
        raise ValueError("Frozen training-only mode centers changed")
    return evaluate_common(summary["prediction_directories"],output,reference_root=spec["reference"],
        expected_seeds=SEEDS,expected_arms=PREDICTION_ARMS,mode_centers=centers)


def program(configuration,freeze,sidecar,parent_pack,synthetic,l2_grid,reference,output,jobs_output,*,
            approved=False,synthetic_job=None,l2_job=None,deadline_hours=24.,poll_seconds=30.):
    if not approved:raise ValueError("Root approval is required before starting the deferred L3 controller")
    if l2_job is None:
        raise ValueError("The L2 launcher job directory is required to prove GPU0 process release")
    if not 0<deadline_hours<=48 or not 0<poll_seconds<=60:
        raise ValueError("Controller deadline/poll must be bounded")
    output,jobs_output=Path(output),Path(jobs_output)
    run=Artifacts(output,seed=list(SEEDS),inputs=[Path(configuration),Path(freeze),Path(__file__),
        HERE/"w15a_cellflow_moe_real_train.py",HERE/"w15a_cellflow_moe_real_predict.py"])
    run.json("controller_spec.json",dict(jobs=registered_jobs(),models=18,expected_manifests=30,windows=360,
        GPU_policy="physical_GPU0_only_after_L2_process_complete",synthetic=str(Path(synthetic).resolve()),
        L2_grid=str(Path(l2_grid).resolve()),deadline_hours=deadline_hours,poll_seconds=poll_seconds,
        no_automatic_retuning=True))
    deadline=time.monotonic()+deadline_hours*3600
    try:
        while True:
            state=dependency_state(synthetic,l2_grid,synthetic_job=synthetic_job,l2_job=l2_job)
            run.json("status.json",state)
            if state["status"]=="do_not_train":
                result=dict(status="L3_REAL_NOT_STARTED_SYNTHETIC_GATE_FAILED",models_trained=0,
                    reason=state["reason"],automatic_retry=False)
                run.json("summary.json",result)
                run.json("status.json",dict(status="complete",outcome="not_started_synthetic_gate_failed"))
                return result
            if state["status"]=="ready":break
            if time.monotonic()>deadline:raise TimeoutError("L3 prerequisites did not complete before deadline")
            time.sleep(poll_seconds)
        run.json("prerequisites.json",state)
        arguments=["worker","--configuration",str(configuration),"--freeze",str(freeze),
            "--sidecar",str(sidecar),"--parent-pack",str(parent_pack),"--synthetic",str(synthetic),
            "--reference",str(reference),"--output",str(output/"grid")]
        gpu=launch_phase(Path(__file__),arguments,jobs_output/"training_gpu0",gpu=0)
        run.json("GPU_job.json",gpu);run.json("status.json",dict(status="waiting_for_GPU0_L3_training_prediction"))
        _wait_job(jobs_output/"training_gpu0",deadline,poll_seconds)
        cpu=launch_phase(Path(__file__),["evaluate","--grid",str(output/"grid"),"--output",str(output/"evaluation")],
                         jobs_output/"evaluation_cpu",gpu=None)
        run.json("CPU_job.json",cpu);run.json("status.json",dict(status="GPU0_released_waiting_for_CPU_common_scoring"))
        _wait_job(jobs_output/"evaluation_cpu",deadline,poll_seconds)
        result=dict(status="L3_REGISTERED_REAL_PIPELINE_COMPLETE",models=18,manifests=30,windows=360,
            evaluation=str((output/"evaluation"/"summary.json").resolve()),automatic_retuning=False)
        run.json("summary.json",result);run.json("status.json",dict(status="complete"));return result
    except Exception as error:
        run.json("status.json",dict(status="failed",error=repr(error),automatic_retry=False))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("program","worker","evaluate"))
    for name in ("configuration","freeze","sidecar","parent-pack","synthetic","l2-grid","reference","jobs-output",
                 "synthetic-job","l2-job","grid"):
        parser.add_argument("--"+name)
    parser.add_argument("--output",required=True);parser.add_argument("--approved",action="store_true")
    parser.add_argument("--deadline-hours",type=float,default=24.)
    args=parser.parse_args()
    if args.action=="program":
        result=program(args.configuration,args.freeze,args.sidecar,args.parent_pack,args.synthetic,args.l2_grid,
            args.reference,args.output,args.jobs_output,approved=args.approved,synthetic_job=args.synthetic_job,
            l2_job=args.l2_job,deadline_hours=args.deadline_hours)
    elif args.action=="worker":
        result=worker_grid(args.configuration,args.freeze,args.sidecar,args.parent_pack,args.synthetic,
            args.reference,args.output)
    else:result=evaluate_grid(args.grid,args.output)
    print(json.dumps(result,indent=2))
