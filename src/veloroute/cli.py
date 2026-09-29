"""A small explicit CLI: no hidden automatic progression into real model training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import Run, load_config, read_csv, save_csv, save_json
from .protocol import protocol_errors


def main():
    parser = argparse.ArgumentParser(prog="veloroute")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Read protocol readiness and recent runs")
    status.add_argument("--protocol", default="configs/veloroute_g1.draft.yaml")
    status.add_argument("--outputs", default="outputs")
    synthetic = commands.add_parser("synthetic", help="Train and test core networks on paired-truth toy systems")
    synthetic.add_argument("--config", default="configs/veloroute_synthetic.yaml")
    synthetic.add_argument("--output", required=True)
    smoke = commands.add_parser("core-smoke", help="Verify full configured core dimensions and checkpoint reproduction")
    smoke.add_argument("--config", default="configs/veloroute_core.yaml")
    smoke.add_argument("--output", required=True)
    splits = commands.add_parser("reserve-conditions", help="Reserve TFs before viewing response effects")
    splits.add_argument("--guides", required=True)
    splits.add_argument("--seed", type=int, default=20260912)
    splits.add_argument("--output", required=True)
    raw = commands.add_parser("raw-plan", help="Build a manifest-checked raw execution plan, without launching it")
    raw.add_argument("--manifest", required=True)
    raw.add_argument("--day", type=int, required=True)
    raw.add_argument("--sra-bin", required=True)
    raw.add_argument("--raw-root", required=True)
    raw.add_argument("--output", required=True)
    launch = commands.add_parser("run-command", help="Launch one explicit command with persistent status/log/provenance")
    launch.add_argument("--output", required=True)
    launch.add_argument("--timeout", type=int, default=3600)
    launch.add_argument("argv", nargs=argparse.REMAINDER)
    fastq = commands.add_parser("fastq-audit", help="Inspect actual split technical/cDNA reads")
    fastq.add_argument("--r1", required=True)
    fastq.add_argument("--r2", required=True)
    fastq.add_argument("--barcodes", required=True)
    fastq.add_argument("--max-records", type=int, default=100000)
    fastq.add_argument("--output", required=True)
    preprocess = commands.add_parser("preprocess", help="Process configured real SRA libraries through USA counts; no model training")
    preprocess.add_argument("--config", default="configs/renge_preprocess.yaml")
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--reuse-run", help="Reuse hash-verified completed raw runs/reference from a stopped preprocessing run")
    preprocess.add_argument("--days", type=int, nargs="+", help="Process only these configured days; useful after one day is complete")
    import_usa = commands.add_parser("import-usa", help="Import completed USA counts with barcode/guide metadata and QC screening")
    import_usa.add_argument("--config", default="configs/renge_preprocess.yaml")
    import_usa.add_argument("--quant-dir", required=True)
    import_usa.add_argument("--day", type=int, required=True)
    import_usa.add_argument("--output", required=True)
    fold = commands.add_parser("prepare-fold", help="Fit training-only S/U transforms and seal role-specific source/target packs")
    fold.add_argument("--source", required=True)
    fold.add_argument("--target", required=True)
    fold.add_argument("--config", default="configs/veloroute_latent.yaml")
    fold.add_argument("--kind", choices=("engineering", "development"), default="engineering")
    fold.add_argument("--output", required=True)
    pipe_smoke = commands.add_parser("pipeline-smoke", help="Synthetic raw S/U -> frozen transforms -> unpaired training -> predict -> evaluate")
    pipe_smoke.add_argument("--output", required=True)
    pipe_smoke.add_argument("--seed", type=int, default=20260912)
    train = commands.add_parser("train-unpaired", help="Train shared experts/router on training packs; real data requires G1")
    train.add_argument("--source", required=True)
    train.add_argument("--target", required=True)
    train.add_argument("--conditions", required=True)
    train.add_argument("--config", default="configs/veloroute_unpaired.yaml")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--frozen-protocol")
    train.add_argument("--decision")
    train.add_argument("--synthetic-engineering", action="store_true")
    train.add_argument('--exploratory-real', action='store_true')
    train.add_argument("--output", required=True)
    predict = commands.add_parser("predict", help="Source-only generation; this command has no target argument")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--source", required=True)
    predict.add_argument("--conditions", required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--repeats", type=int, default=1)
    predict.add_argument("--transform", help="Frozen transform.npz used for inverse PCA gene-space output")
    predict.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate immutable predictions against an independent future pack")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--target-genes", help="Separate future gene-space pack, opened only during evaluation")
    evaluate.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze-protocol", help="Hash-bind a complete G1 preregistration; rejects missing evidence")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--output", required=True)
    g1 = commands.add_parser("g1", help="Run all fixed-budget A/B probes after protocol freeze, not router training")
    g1.add_argument("--frozen-protocol", required=True)
    g1.add_argument("--train-source", required=True)
    g1.add_argument("--train-target", required=True)
    g1.add_argument("--validation-source", required=True)
    g1.add_argument("--validation-target", required=True)
    g1.add_argument("--conditions", required=True)
    g1.add_argument("--output", required=True)
    workflow = commands.add_parser("workflow", help="Counts -> frozen packs -> public conditions -> explicit G1 review gate -> experiments")
    workflow.add_argument("--config", default="configs/veloroute_pipeline.yaml")
    workflow.add_argument("--output", required=True)
    full_smoke = commands.add_parser('full-smoke', help='Exercise full configured architecture on synthetic S/U')
    full_smoke.add_argument('--config', default='configs/veloroute_full.yaml')
    full_smoke.add_argument('--device', default='cpu')
    full_smoke.add_argument('--output', required=True)
    full_train = commands.add_parser('train-full', help='Full A/B/C/D training; explicit exploratory mode is not G1-GO')
    for name in ('source', 'target', 'conditions', 'output'):
        full_train.add_argument('--'+name, required=True)
    full_train.add_argument('--config', default='configs/veloroute_full.yaml')
    for name in ('transform', 'combinations', 'frozen-protocol', 'decision', 'resume'):
        full_train.add_argument('--'+name)
    full_train.add_argument('--synthetic-engineering', action='store_true')
    full_train.add_argument('--exploratory-real', action='store_true')
    full_predict = commands.add_parser('predict-full', help='Full source-only model prediction without teacher/targets')
    for name in ('checkpoint', 'source', 'conditions', 'output'):
        full_predict.add_argument('--'+name, required=True)
    full_predict.add_argument('--transform')
    full_predict.add_argument('--combinations')
    full_predict.add_argument('--device', default='cpu')
    full_predict.add_argument('--seed', type=int, default=0)
    full_predict.add_argument('--repeats', type=int, default=1)
    full_predict.add_argument('--deterministic', action='store_true')
    explore = commands.add_parser('velocity-increment', help='Exploratory A/B analysis, never an automatic G1-GO')
    explore.add_argument('--config', default='configs/veloroute_velocity_exploratory_20260913.yaml')
    for name in ('fold', 'conditions', 'output'):
        explore.add_argument('--'+name, required=True)
    router_increment = commands.add_parser('router-increment', help='Fixed developer-side router comparisons, not formal G1')
    router_increment.add_argument('--config', default='configs/veloroute_router_increment_20260913.yaml')
    router_increment.add_argument('--output', required=True)
    fixed = commands.add_parser('frozen-candidates', help='Fixed-budget shared-frozen-expert routing follow-up')
    fixed.add_argument('--config', default='configs/veloroute_frozen_candidates_20260914.yaml')
    fixed.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'router-increment':
        from .router_increment import run_router_increment
        print(json.dumps(run_router_increment(args.config, args.output)))
    elif args.command == 'frozen-candidates':
        from .frozen_candidates import run_frozen_candidates
        print(json.dumps(run_frozen_candidates(args.config, args.output)))
    elif args.command == 'full-smoke':
        from .full_smoke import run_full_smoke
        print(json.dumps(run_full_smoke(load_config(args.config), args.output, config_path=args.config, device=args.device)))
    elif args.command == 'train-full':
        from .full_experiments import train_full_model
        print(json.dumps(train_full_model(args.source, args.target, args.conditions, load_config(args.config), args.output,
            transform_path=args.transform, combination_path=args.combinations, frozen_path=args.frozen_protocol,
            decision_path=args.decision, synthetic_engineering=args.synthetic_engineering,
            exploratory_real=args.exploratory_real, resume=args.resume)))
    elif args.command == 'predict-full':
        from .full_experiments import predict_full_model
        print(json.dumps(predict_full_model(args.checkpoint, args.source, args.conditions, args.output,
            transform_path=args.transform, combination_path=args.combinations, seed=args.seed, repeats=args.repeats,
            device=args.device, stochastic=not args.deterministic)))
    elif args.command == 'velocity-increment':
        from .probes import run_g1
        fold = Path(args.fold)
        print(json.dumps(run_g1(None, fold/'train_source.npz', fold/'train_target.npz', fold/'validation_source.npz',
            fold/'validation_target.npz', args.conditions, args.output, exploratory_config_path=args.config)))
    elif args.command == "workflow":
        from .workflow import run_workflow
        print(json.dumps(run_workflow(args.config, args.output)))
    elif args.command == "pipeline-smoke":
        from .pipeline_smoke import run_pipeline_smoke
        print(json.dumps(run_pipeline_smoke(args.output, seed=args.seed)))
    elif args.command == "train-unpaired":
        from .experiments import train_model
        print(json.dumps(train_model(args.source, args.target, args.conditions, load_config(args.config), args.output,
            seed=args.seed, frozen_path=args.frozen_protocol, decision_path=args.decision, engineering=args.synthetic_engineering,
            exploratory_real=args.exploratory_real)))
    elif args.command == "predict":
        from .experiments import predict_model
        print(json.dumps(predict_model(args.checkpoint, args.source, args.conditions, args.output, seed=args.seed,
                                       repeats=args.repeats, transform_path=args.transform)))
    elif args.command == "evaluate":
        from .experiments import evaluate_prediction
        print(json.dumps(evaluate_prediction(args.predictions, args.target, args.output, target_genes_path=args.target_genes)))
    elif args.command == "freeze-protocol":
        from .protocol import freeze_protocol
        config = load_config(args.config)
        frozen = freeze_protocol(config)
        with Run(args.output, stage="protocol_freeze", kind="development", config=config, inputs=[args.config]) as run:
            save_json(run.directory/"frozen_protocol.json", frozen)
    elif args.command == "g1":
        from .probes import run_g1
        print(json.dumps(run_g1(args.frozen_protocol, args.train_source, args.train_target, args.validation_source,
                               args.validation_target, args.conditions, args.output)))
    elif args.command == "prepare-fold":
        from .latent import prepare_fold
        print(json.dumps(prepare_fold(args.source, args.target, load_config(args.config), args.output, kind=args.kind)))
    elif args.command == "preprocess":
        from .preprocess import run_preprocessing
        run_preprocessing(args.config, args.output, reuse_run=args.reuse_run, days=args.days)
    elif args.command == "import-usa":
        from .preprocess import import_sample
        print(json.dumps(import_sample(args.quant_dir, args.day, load_config(args.config), args.output)))
    elif args.command == "status":
        config = load_config(args.protocol)
        runs = []
        for path in sorted(Path(args.outputs).glob("**/provenance.json")):
            record = json.loads(path.read_text())
            entry = {"path": str(path.parent), "status": record.get("status"),
                         "stage": record.get("stage"), "kind": record.get("kind"),
                         "finished_utc": record.get("finished_utc")}
            state_path = path.parent/"status.json"
            if state_path.is_file():
                state = json.loads(state_path.read_text())
                entry["job"] = state
                if state.get("status") == "running" and state.get("pid"):
                    import os
                    try:
                        os.kill(state["pid"], 0)
                        entry["recorded_pid_alive"] = True
                    except ProcessLookupError:
                        entry["recorded_pid_alive"] = False
                        entry["status"] = "stale_running_record_process_absent"
                    except PermissionError:
                        entry["recorded_pid_alive"] = "not_authorized_to_check"
            runs.append(entry)
        print(json.dumps({"core_implementation_authorized": True, "formal_real_training_authorized": False,
                          "G1_readiness_errors": protocol_errors(config), "runs": runs}, ensure_ascii=False, indent=2))
    elif args.command == "synthetic":
        from .synthetic import run_synthetic
        result = run_synthetic(load_config(args.config), args.output, args.config)
        print(json.dumps(result))
        return 0 if result["status"] == "ENGINEERING_PASS" else 1
    elif args.command == "core-smoke":
        from .smoke import run_core_smoke
        print(json.dumps(run_core_smoke(load_config(args.config), args.output, args.config)))
    elif args.command == "reserve-conditions":
        from .splits import make_condition_split
        config = {"seed": args.seed, "validation_TFs": 4, "confirmation_TFs": 5,
                  "selection_uses_response": False, "status": "condition_reservation_not_formal_cell_split"}
        with Run(args.output, stage="condition_reservation", kind="engineering", config=config,
                 inputs=[args.guides], seed=args.seed) as run:
            rows = make_condition_split(read_csv(args.guides), seed=args.seed)
            save_csv(run.directory/"condition_split.csv", rows)
            (run.directory/"RESULTS.md").write_text("# Condition reservation\n\nTF-level roles reserved independently of response values, across all four days.\n\nThis is not a validated cell split or a frozen G1 protocol. CTRL and AAVS1 remain separate.\n")
    elif args.command == "raw-plan":
        from .raw import plan_raw
        config = {"day": args.day, "sra_bin": args.sra_bin, "raw_root": args.raw_root}
        with Run(args.output, stage="raw_plan", kind="engineering", config=config, inputs=[args.manifest]) as run:
            tasks = plan_raw(read_csv(args.manifest), args.day, args.sra_bin, args.raw_root)
            save_json(run.directory/"tasks.json", tasks)
            (run.directory/"RESULTS.md").write_text("# Raw execution plan\n\nNot launched by this command. GEX and gRNA are kept separate; chemistry/reference still require approval.\n")
    elif args.command == "run-command":
        from .raw import run_command
        argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
        print(json.dumps(run_command(argv, args.output, timeout_seconds=args.timeout)))
    elif args.command == "fastq-audit":
        import gzip
        from .raw import audit_fastq_pair
        opener = gzip.open if args.barcodes.endswith(".gz") else open
        with opener(args.barcodes, "rt") as stream:
            barcodes = [line.strip() for line in stream if line.strip()]
        with Run(args.output, stage="raw_layout_audit", kind="engineering",
                 config={"max_records": args.max_records}, inputs=[args.r1, args.r2, args.barcodes]) as run:
            result = audit_fastq_pair(args.r1, args.r2, whitelist=barcodes, max_records=args.max_records)
            save_json(run.directory/"layout.json", result)
            (run.directory/"RESULTS.md").write_text("# Split FASTQ layout audit\n\n```json\n"+json.dumps(result, indent=2)+"\n```\n\nLayout observations do not automatically approve chemistry, UMI boundaries, or S/U quality.\n")
            print(json.dumps(result))
    return 0
