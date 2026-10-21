"""CPU fixtures for the gated real-L3 pipeline; never train biological data."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"/"phase0"))
from w4_artifacts import sha256
from w15a_cellflow_moe_real_train import (VARIANTS, registered_jobs, variant_spec,
    velocity_input, require_complete_synthetic_gate, train_real, load_training_groups,
    fit_group_arrays, read_feature_cohort)
from w15a_cellflow_moe_real_predict import predict_real
from w15a_cellflow_common_evaluation import reference_sources,freeze_common_grid,evaluate_common,KEYS


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+"\n")


@pytest.fixture
def fixture(tmp_path):
    rng=np.random.default_rng(0)
    parent=tmp_path/"parent";parent.mkdir()
    reference=tmp_path/"reference";reference.mkdir()
    old=tmp_path/"old_W13_source";old.mkdir()
    sidecar=tmp_path/"source_sidecar";sidecar.mkdir()
    predictions=[];samples={};target_caches=[];development=[];training=[]
    for start in (5,10,15):
        z=rng.normal(0,.1,(12,3)).astype("float32")
        ids=np.array([f"cell_{start}_{i}" for i in range(12)])
        old_path=old/f"source_{start}.npz";np.savez_compressed(old_path,z=z,cell_ids=ids)
        predictions.append(dict(source_hour=start,target_hour=start+5,path=str(old_path),sha256=sha256(old_path)))
        samples[f"{start}_to_{start+5}"]=list(range(8))
        target=reference/f"future_target_{start+5}.npz";np.savez_compressed(target,target=z[:8]+.1)
        target_caches.append(dict(target_hour=start+5,path=str(target),sha256=sha256(target)))
        source_key=f"r4|source|{start}h"
        order=np.arange(11,-1,-1)
        kinetic=np.tile(np.linspace(-3.,3.,12)[:,None],(1,3)).astype("float32")
        parent_path=parent/f"source_{start}.npz"
        np.savez_compressed(parent_path,z=z[order],cell_ids=ids[order],cell_kin=kinetic[order])
        side_path=sidecar/f"source_{start}.npz"
        np.savez_compressed(side_path,z=z[order],cell_ids=ids[order],cell_kin=kinetic[order],
            cell_directU=(kinetic*.5+z)[order],cell_kin_Umean=np.zeros_like(kinetic),
            cell_kin_Uzero=np.full_like(kinetic,-2.))
        development.append(dict(source_key=source_key,path=str(side_path),sha256=sha256(side_path),
            parent_path=str(parent_path),parent_sha256=sha256(parent_path),replicate="r4",label=f"{start:02d}h",
            time_hours=start))
    z=rng.normal(0,.1,(12,3)).astype("float32");ids=np.array([f"training_{i}" for i in range(12)])
    kinetic=rng.normal(0,.3,(12,3)).astype("float32")
    parent_path=parent/"train.npz";np.savez_compressed(parent_path,z=z,cell_ids=ids,cell_kin=kinetic,target=z+.2)
    side_path=sidecar/"train.npz"
    np.savez_compressed(side_path,z=z,cell_ids=ids,cell_kin=kinetic,cell_directU=kinetic*2,
        cell_kin_Umean=kinetic*.5,cell_kin_Uzero=np.full_like(kinetic,-2.))
    training.append(dict(source_key="r1|source|0h",path=str(side_path),sha256=sha256(side_path),
        parent_path=str(parent_path),parent_sha256=sha256(parent_path),replicate="r1",label="D3",time_hours=0))
    save_json(parent/"manifest.json",dict(kind="CPU_TOY_PARENT_SOURCE_FIXTURE"))
    norm={"mean_cell_kin":np.zeros(3),"scale_cell_kin":np.ones(3),
          "mean_cell_directU":np.ones(3)*.1,"scale_cell_directU":np.ones(3)*2,
          "training_source_keys":np.array([training[0]["source_key"]])}
    np.savez_compressed(sidecar/"normalizer.npz",**norm)
    save_json(sidecar/"manifest.json",dict(status="L3_SOURCE_KINETIC_SIDECAR_READY",training_groups=training,
        development_sources=development,parent_pack=str(parent),parent_pack_manifest_sha256=sha256(parent/"manifest.json"),
        normalizer_sha256=sha256(sidecar/"normalizer.npz")))
    save_json(old/"manifest.json",dict(prediction_groups=predictions))
    save_json(reference/"source_samples.json",samples)
    save_json(reference/"study_manifest.json",dict(pack=str(old),pack_manifest_sha256=sha256(old/"manifest.json"),
        evaluation_caches=target_caches))
    config=dict(width=8,depth=1,condition_dim=4,time_freqs=8,time_width=8,time_depth=1,
        pooling="mean",learning_rate=.001,ot_epsilon=.1,ot_max_iterations=30,ema=1.,batch_size=4,steps=2,log_interval=1)
    return dict(sidecar=sidecar,parent=parent,reference=reference,config=config,normalizer=norm,
                training=training,development=development,target_paths={r["path"] for r in target_caches})


def test_registered_real_grid_is_exactly_six_variants_times_three_seeds():
    jobs=registered_jobs()
    assert len(jobs)==18 and len({(j["variant"],j["seed"]) for j in jobs})==18
    assert {j["seed"] for j in jobs}=={0,1,2}
    assert {j["shuffle_seed"] for j in jobs if j["arm"]=="moe_kin_shuf"}=={0,1,2}
    assert len([j for j in jobs if j["arm"]=="moe_kin_shuf"])==9
    with pytest.raises(ValueError,match="Unregistered"):
        variant_spec("moe_kin_shuf_perm9")


def test_local_direct_u_is_not_broadcast_population_u(fixture):
    pack,_=read_feature_cohort(fixture["sidecar"],fixture["training"][0]["source_key"])
    value=velocity_input(pack,fixture["normalizer"],"moe_directU","source")
    expected=(pack["cell_directU"]-.1)/2
    np.testing.assert_allclose(value,expected,atol=1e-7)
    assert not np.allclose(value,value[0])
    assert np.count_nonzero(velocity_input(pack,fixture["normalizer"],"moe_static","source"))==0


def test_three_permutation_seeds_preserve_local_feature_rows(fixture):
    pack,_=read_feature_cohort(fixture["sidecar"],fixture["training"][0]["source_key"])
    real=velocity_input(pack,fixture["normalizer"],"moe_kin","source")
    outputs=[]
    for seed in (0,1,2):
        shuffled=velocity_input(pack,fixture["normalizer"],f"moe_kin_shuf_perm{seed}","source")
        np.testing.assert_array_equal(np.sort(shuffled,axis=0),np.sort(real,axis=0))
        np.testing.assert_array_equal(shuffled,velocity_input(pack,fixture["normalizer"],f"moe_kin_shuf_perm{seed}","source"))
        outputs.append(shuffled)
    assert not np.array_equal(outputs[0],outputs[1])


def test_running_synthetic_scores_never_opened(tmp_path,monkeypatch):
    partial=tmp_path/"partial_synthetic";partial.mkdir()
    save_json(partial/"status.json",dict(status="running",completed_fits=15))
    (partial/"all_repetitions.csv").write_text("Do not read incomplete scores")
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.name!="all_repetitions.csv","Incomplete scores opened"
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    with pytest.raises(ValueError,match="Complete 120-fit"):
        require_complete_synthetic_gate(partial)


def test_engineering_smoke_cannot_unlock_real_training(tmp_path):
    smoke=tmp_path/"smoke";smoke.mkdir()
    save_json(smoke/"summary.json",dict(status="L3_POPULATION_ENGINEERING_SMOKE_COMPLETE",
        completed_fits=6,full_positive_null_grid_completed=False,population_composition_learning_gate_passed=True))
    with pytest.raises(ValueError,match="has not passed"):
        require_complete_synthetic_gate(smoke)


def test_real_train_gate_precedes_any_data_or_accelerator_loading(tmp_path,monkeypatch):
    import w15a_cellflow_moe_real_train as module
    monkeypatch.setattr(module.jax,"devices",lambda:pytest.fail("Accelerator touched before synthetic gate"))
    with pytest.raises(ValueError,match="Complete 120-fit"):
        train_real("missing_sidecar","missing_pack","missing_cfg","missing_freeze",tmp_path/"incomplete",
            tmp_path/"must_not_exist",variant="moe_static",seed=0)
    assert not (tmp_path/"must_not_exist").exists()


def test_training_target_read_from_parent_not_feature_sidecar(fixture):
    groups=load_training_groups(fixture["sidecar"],"moe_kin")
    assert len(groups)==1 and groups[0]["target"].shape==(12,3)
    np.testing.assert_allclose(groups[0]["target"],groups[0]["z"]+.2,atol=1e-7)


def test_cpu_toy_fit_serialize_common_predict_diagnostics_and_score(fixture,tmp_path,monkeypatch):
    import jax.numpy as jnp
    from flax import serialization
    groups=load_training_groups(fixture["sidecar"],"moe_kin")
    model,params,history,_=fit_group_arrays(groups,fixture["config"],0)
    assert history[-1]["step"]==2 and np.isfinite(history[-1]["loss"])
    training=tmp_path/"cpu_synthetic_checkpoint";training.mkdir()
    (training/"model.msgpack").write_bytes(serialization.to_bytes(params))
    save_json(training/"summary.json",dict(status="CPU_SYNTHETIC_L3_FIXTURE_TRAINED",engineering_fixture=True,
        variant="moe_kin",seed=0,config=fixture["config"],dimensions=3,experts=8,
        sidecar_directory=str(fixture["sidecar"]),sidecar_manifest_sha256=sha256(fixture["sidecar"]/"manifest.json"),
        sidecar_normalizer_sha256=sha256(fixture["sidecar"]/"normalizer.npz"),model_sha256=sha256(training/"model.msgpack")))
    original=np.load
    def no_future(path,*args,**kwargs):
        assert str(path) not in fixture["target_paths"],"Future target cache read during inference"
        return original(path,*args,**kwargs)
    with monkeypatch.context() as context:
        context.setattr(np,"load",no_future)
        result=predict_real(training,tmp_path/"forecast",reference_root=fixture["reference"],engineering_fixture=True)
    assert result["target_read"] is False and len(result["exports"])==5
    paths=[Path(row["path"]) for row in result["exports"]]
    expected=[row["arm"] for row in result["exports"]]
    freeze_common_grid(paths,reference_root=fixture["reference"],expected_seeds=(0,),expected_arms=expected)
    with original(paths[0]/"predictions.npz",allow_pickle=False) as main, original(paths[1]/"predictions.npz",allow_pickle=False) as cohort:
        common=reference_sources(fixture["reference"])["sources"][5]
        pack,_=read_feature_cohort(fixture["sidecar"],fixture["development"][0]["source_key"])
        inputs=velocity_input(pack,fixture["normalizer"],"moe_kin",fixture["development"][0]["source_key"])
        fullq=np.asarray(model.apply(dict(params=params),jnp.asarray(pack["z"]),jnp.asarray(inputs),method=model.route))
        np.testing.assert_array_equal(main["05_10_decay_source_ids"],common["cell_ids"])
        np.testing.assert_array_equal(main["05_10_decay"],cohort["05_10_decay"])
        np.testing.assert_allclose(cohort["05_10_decay_q"],np.repeat(fullq.mean(0)[None],8,axis=0),atol=1e-7)
        assert not np.allclose(fullq.mean(0),main["05_10_decay_q"].mean(0),atol=1e-8,rtol=1e-8)
        assert main["05_10_decay"].shape==(8,8,3)
        np.testing.assert_array_equal(main["05_10_decay_q"],main["05_20_linear_q"])
    metrics=evaluate_common(paths,tmp_path/"scored",reference_root=fixture["reference"],expected_seeds=(0,),expected_arms=expected)
    assert len(metrics["metrics"])==60
    diagnostics=json.loads((tmp_path/"forecast"/"l3_inference_diagnostics.json").read_text())
    assert diagnostics["global_U_independence_established"] is False
    assert diagnostics["inference_without_U_established"] is False
    assert len(diagnostics["U_sensitivity"])==6


def test_engineering_checkpoint_cannot_be_used_by_production_predictor(fixture,tmp_path):
    training=tmp_path/"synthetic";training.mkdir()
    save_json(training/"summary.json",dict(status="CPU_SYNTHETIC_L3_FIXTURE_TRAINED",engineering_fixture=True))
    with pytest.raises(ValueError,match="gated real-data"):
        predict_real(training,tmp_path/"must_not_predict",reference_root=fixture["reference"])


def test_real_train_success_branch_writes_complete_artifacts_with_mocked_gate_boundary(tmp_path,monkeypatch):
    """Mock scientific execution, not Artifacts writing: catch post-fit JSON failures."""
    import w15a_cellflow_moe_real_train as module
    import w15a_cellflow_registered_grid as registered
    configuration=tmp_path/"selected_configuration.json";save_json(configuration,dict(test_fixture=True))
    freeze=tmp_path/"freeze_record.json";save_json(freeze,dict(test_fixture=True))
    sidecar=tmp_path/"sidecar";sidecar.mkdir();save_json(sidecar/"manifest.json",dict(test_fixture=True))
    np.savez_compressed(sidecar/"normalizer.npz",mean_cell_kin=np.zeros(50))
    synthetic=tmp_path/"synthetic_gate";synthetic.mkdir();save_json(synthetic/"summary.json",dict(test_fixture=True))
    config=dict(width=128,depth=2,steps=5000,learning_rate=.0003)
    frozen=dict(configuration_sha256=sha256(configuration),freeze_commit="f"*40)
    monkeypatch.setattr(module,"require_complete_synthetic_gate",lambda path:dict(
        l2_configuration_sha256=sha256(configuration),engineering_gate_passed=True))
    monkeypatch.setattr(registered,"validate_freeze",lambda *a,**kw:(config,frozen))
    monkeypatch.setattr(module,"require_l2_freeze",lambda *a,**kw:None)
    metadata=[dict(replicate=rep,source_key=f"{rep}_D{day}") for rep in ("r1","r2","r3") for day in range(3,8)]
    monkeypatch.setattr(module,"sidecar_metadata",lambda path:dict(training_groups=metadata))
    groups=[dict(z=np.zeros((2,50),dtype="float32")) for _ in metadata]
    monkeypatch.setattr(module,"load_training_groups",lambda *a:groups)
    history=[dict(step=5000,loss=.5)]
    monkeypatch.setattr(module,"fit_group_arrays",lambda *a,**kw:(None,dict(toy=np.ones(1)),history,dict(toy=np.zeros(1))))
    output=tmp_path/"written_training"
    result=module.train_real(sidecar,tmp_path/"parent",configuration,freeze,synthetic,output,variant="moe_static",seed=0)
    assert result["status"]=="L3_REAL_DATA_TRAINED"
    assert (output/"model.msgpack").is_file()
    assert json.loads((output/"training_history.json").read_text())["history"]==history
    assert json.loads((output/"summary.json").read_text())["all_training_steps"]==5000
    assert json.loads((output/"status.json").read_text())["status"]=="complete"
