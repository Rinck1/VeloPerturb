"""CPU-only orchestration simulations; no real model/GPU queue is launched."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"/"phase0"))
import w15a_cellflow_moe_real_grid as grid


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def test_grid_has_eighteen_fits_and_thirty_prediction_manifests():
    assert len(grid.registered_jobs())==18
    assert len(grid.PREDICTION_ARMS)==10
    assert len(set(grid.PREDICTION_ARMS))==10
    assert len(grid.PREDICTION_ARMS)*len(grid.SEEDS)==30


def test_pending_dependencies_do_not_read_partial_scores(tmp_path,monkeypatch):
    synthetic=tmp_path/"running";synthetic.mkdir()
    (synthetic/"all_repetitions.csv").write_text("incomplete score file must not be opened")
    monkeypatch.setattr(grid,"require_complete_synthetic_gate",lambda p:pytest.fail("Partial gate inspected"))
    assert grid.dependency_state(synthetic,tmp_path/"L2")["status"]=="waiting"


def test_failed_complete_gate_stops_before_gpu_or_l2_scoring(tmp_path,monkeypatch):
    synthetic=tmp_path/"synthetic"
    write_json(synthetic/"summary.json",dict(status="L3_POPULATION_SYNTHETIC_GRID_COMPLETE",completed_fits=120,
        full_positive_null_grid_completed=True,population_composition_learning_gate_passed=False))
    monkeypatch.setattr(grid,"require_complete_synthetic_gate",lambda p:pytest.fail("Failed gate should stop"))
    state=grid.dependency_state(synthetic,tmp_path/"L2")
    assert state["status"]=="do_not_train"


def test_l2_summary_not_enough_until_gpu_process_released(tmp_path,monkeypatch):
    synthetic=tmp_path/"synthetic";l2=tmp_path/"L2";job=tmp_path/"L2_job"
    write_json(synthetic/"summary.json",dict(status="L3_POPULATION_SYNTHETIC_GRID_COMPLETE",completed_fits=120,
        full_positive_null_grid_completed=True,population_composition_learning_gate_passed=True))
    write_json(l2/"summary.json",dict(status="REGISTERED_L2_DEVELOPMENT_GRID_COMPLETE",models=18))
    write_json(job/"status.json",dict(status="running"))
    monkeypatch.setattr(grid,"require_complete_synthetic_gate",lambda p:dict(engineering_gate_passed=True))
    assert grid.dependency_state(synthetic,l2,l2_job=job)["status"]=="waiting"
    write_json(job/"status.json",dict(status="complete"))
    assert grid.dependency_state(synthetic,l2,l2_job=job)["status"]=="ready"


def test_failed_predecessor_is_not_retried(tmp_path):
    job=tmp_path/"job";write_json(job/"status.json",dict(status="failed"))
    with pytest.raises(RuntimeError,match="no automatic retry"):
        grid.dependency_state(tmp_path/"syn",tmp_path/"L2",synthetic_job=job)


def test_controller_gate_failure_writes_stop_without_launch(tmp_path,monkeypatch):
    config=tmp_path/"configuration.json";freeze=tmp_path/"freeze.json"
    write_json(config,{});write_json(freeze,{})
    monkeypatch.setattr(grid,"dependency_state",lambda *a,**kw:dict(status="do_not_train",reason="synthetic_failed"))
    monkeypatch.setattr(grid,"launch_phase",lambda *a,**kw:pytest.fail("GPU launch after failed gate"))
    result=grid.program(config,freeze,"source","parent","synthetic","L2","ref",tmp_path/"controller",
        tmp_path/"job",approved=True,l2_job=tmp_path/"L2_job",poll_seconds=.01)
    assert result["models_trained"]==0
    assert result["status"]=="L3_REAL_NOT_STARTED_SYNTHETIC_GATE_FAILED"
    assert (tmp_path/"controller"/"summary.json").is_file()
    assert json.loads((tmp_path/"controller"/"status.json").read_text())["outcome"]=="not_started_synthetic_gate_failed"


def test_program_requires_l2_launcher_release_evidence(tmp_path):
    with pytest.raises(ValueError,match="L2 launcher job directory"):
        grid.program("config","freeze","source","parent","synthetic","L2","ref",tmp_path/"controller",
                     tmp_path/"jobs",approved=True)


def test_successful_worker_runs_fixed_eighteen_jobs_and_freezes_all_thirty_before_scoring(tmp_path,monkeypatch):
    config=tmp_path/"configuration.json";freeze=tmp_path/"freeze.json"
    sidecar=tmp_path/"sidecar";reference=tmp_path/"reference"
    for path in (config,freeze,sidecar/"manifest.json",reference/"study_manifest.json",reference/"source_samples.json"):
        write_json(path,{})
    monkeypatch.setattr(grid,"require_complete_synthetic_gate",lambda p:dict(engineering_gate_passed=True))
    fitted=[];predicted=[];freeze_calls=[]
    def train(*args,variant,seed,**kwargs):
        fitted.append((variant,seed));return dict(model_sha256=f"model_{variant}_{seed}")
    def predict(training,output,**kwargs):
        variant=training.parent.parent.name;seed=int(training.parent.name[4:])
        predicted.append((variant,seed))
        exports=[dict(path=str(output),arm=variant,seed=seed)]
        if variant=="moe_kin":
            exports.extend(dict(path=str(output/name),arm=f"moe_kin_{name}",seed=seed) for name in grid.DIAGNOSTICS)
        return dict(exports=exports)
    def freeze_all(paths,**kwargs):
        assert len(fitted)==len(predicted)==18
        assert len(paths)==30
        freeze_calls.append(kwargs)
        return {},[dict(path=str(path)) for path in paths]
    monkeypatch.setattr(grid,"train_real",train);monkeypatch.setattr(grid,"predict_real",predict)
    monkeypatch.setattr(grid,"freeze_common_grid",freeze_all)
    monkeypatch.setattr(grid,"evaluate_common",lambda *a,**kw:pytest.fail("GPU worker scored targets"))
    result=grid.worker_grid(config,freeze,sidecar,"parent","synthetic",reference,tmp_path/"worker",require_gpu=False)
    assert set(fitted)=={(v,s) for v in grid.VARIANTS for s in grid.SEEDS}
    assert freeze_calls[0]["expected_arms"]==grid.PREDICTION_ARMS
    assert result["prediction_manifests"]==30 and result["windows"]==360
    assert result["scientific_scores_computed"] is False
    assert json.loads((tmp_path/"worker"/"status.json").read_text())["status"]=="complete"
    assert (tmp_path/"worker"/"whole_grid_frozen.json").is_file()


def test_successful_program_releases_gpu_worker_before_cpu_evaluation(tmp_path,monkeypatch):
    config=tmp_path/"configuration.json";freeze=tmp_path/"freeze.json"
    write_json(config,{});write_json(freeze,{})
    monkeypatch.setattr(grid,"dependency_state",lambda *a,**kw:dict(status="ready",synthetic_gate=dict(passed=True)))
    calls=[]
    def launch(script,arguments,directory,*,gpu):
        calls.append(("launch",arguments[0],gpu));return dict(status="LAUNCHED",pid=len(calls))
    def wait(directory,*args):
        calls.append(("complete",Path(directory).name));return dict(status="complete")
    monkeypatch.setattr(grid,"launch_phase",launch);monkeypatch.setattr(grid,"_wait_job",wait)
    result=grid.program(config,freeze,"source","parent","synthetic","L2","ref",tmp_path/"controller",
        tmp_path/"jobs",approved=True,l2_job=tmp_path/"L2_job",poll_seconds=.01)
    assert calls==[("launch","worker",0),("complete","training_gpu0"),
                   ("launch","evaluate",None),("complete","evaluation_cpu")]
    assert result["status"]=="L3_REGISTERED_REAL_PIPELINE_COMPLETE"
    assert json.loads((tmp_path/"controller"/"status.json").read_text())["status"]=="complete"


def test_launch_phase_gpu0_and_cpu_use_explicit_backend_env(tmp_path,monkeypatch):
    calls=[]
    def launch(command,**kwargs):
        calls.append((command,kwargs["env"]));return '{"status":"LAUNCHED","pid":123}'
    monkeypatch.setattr(grid.subprocess,"check_output",launch)
    grid.launch_phase("scripts/phase0/job.py",["worker"],tmp_path/"gpu",gpu=0)
    grid.launch_phase("scripts/phase0/job.py",["evaluate"],tmp_path/"cpu",gpu=None)
    assert calls[0][0][calls[0][0].index("--gpu")+1]=="0"
    assert calls[0][1]["JAX_PLATFORMS"]=="cuda"
    assert "--cpu" in calls[1][0] and calls[1][1]["JAX_PLATFORMS"]=="cpu"
    with pytest.raises(ValueError,match="GPU0 only"):
        grid.launch_phase("anything",[],tmp_path/"bad",gpu=1)


def test_incomplete_grid_refused_before_target_scoring(tmp_path,monkeypatch):
    write_json(tmp_path/"incomplete"/"summary.json",dict(status="L3_REAL_GRID_PREDICTIONS_FROZEN",models=17,prediction_manifests=29))
    write_json(tmp_path/"incomplete"/"study_manifest.json",{})
    monkeypatch.setattr(grid,"evaluate_common",lambda *a,**kw:pytest.fail("Incomplete target scoring"))
    with pytest.raises(ValueError,match="Complete frozen"):
        grid.evaluate_grid(tmp_path/"incomplete",tmp_path/"must_not_score")
