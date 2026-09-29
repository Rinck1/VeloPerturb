"""Record existing environments, assets and tests without importing training stacks."""
import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from audit_renge_assets import now, sha256, write_csv, write_json


def capture(command):
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=40)
        return {"command": command, "returncode": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path("/home/yuchang/veloFM")
    required = [
        "scripts/train/train.py", "src/velofm/ot/sinkhorn_torch.py",
        "scripts/preprocess/build_anndata.py", "requirements.txt", "environment.yml",
        "configs/experiment/norman_cellflow_strict_simpleaf.yaml",
        "src/velofm/evaluation/diagnostics.py", "tests/test_diagnostics.py",
        "scripts/velocity/probe_multimodality.py",
        "scripts/velocity/probe_transport_information.py",
        "scripts/velocity/probe_velocity_increment.py",
        "docs/rna_velocity_cfm_norman_negative_report_20260822.md",
        "docs/rna_velocity_cfm_project_wrapup.md",
        "docs/environment_and_baseline_validation_20260812.md",
        "docs/rna_dynamics_grounded_perturbation_proposal_cn.md",
        "outputs/strict_cellflow_pytorch/simpleaf/runs/norman_cellflow_strict_simpleaf_fold0/final.ckpt",
        "outputs/strict_cellflow_pytorch/simpleaf/eval/official_5fold_summary.csv",
    ]
    assets = []
    for relpath in required:
        path = repo / relpath
        assets.append({"path": str(path), "exists": path.is_file(),
                       "sha256": sha256(path) if path.is_file() else ""})
    write_csv(args.output / "assets.csv", assets)
    code_hashes = []
    for directory in (repo / "src", repo / "scripts", repo / "tests", repo / "configs"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".yaml", ".yml"}:
                code_hashes.append({"path": str(path), "sha256": sha256(path)})
    write_csv(args.output / "repository_snapshot_hashes.csv", code_hashes)
    packages = ["numpy", "torch", "anndata", "scanpy", "scvelo", "scvi-tools", "pytest",
                "scipy", "pandas", "hydra-core", "openpyxl", "pot", "lightning", "jax"]
    interpreters = {
        "cellflow": "/home/yuchang/miniconda3/envs/cellflow/bin/python",
        "scdfm": "/home/yuchang/miniconda3/envs/scdfm/bin/python",
        "cellot": "/home/yuchang/miniconda3/envs/cellot/bin/python",
        "venv-gfg": "/home/yuchang/venv-gfg/bin/python",
    }
    code = (
        "import importlib.metadata as m, json, sys\n"
        "result={'python':sys.version,'executable':sys.executable,'packages':{}}\n"
        f"for package in {packages!r}:\n"
        " try: result['packages'][package]=m.version(package)\n"
        " except m.PackageNotFoundError: result['packages'][package]=None\n"
        "print(json.dumps(result))\n"
    )
    envs, env_rows = {}, []
    for name, executable in interpreters.items():
        result = capture([executable, "-B", "-c", code])
        envs[name] = result
        if result.get("returncode") == 0:
            parsed = json.loads(result["stdout"])
            env_rows.append({"environment": name, "executable": executable,
                             "python": parsed["python"].split()[0], **parsed["packages"]})
    write_json(args.output / "environment_checks.json", envs)
    write_csv(args.output / "environment_versions.csv", env_rows)
    tools = {name: shutil.which(name) for name in (
        "prefetch", "fasterq-dump", "simpleaf", "alevin-fry", "piscem", "samtools", "git", "conda")}
    resources = {"checked_utc": now(), "tools_on_current_path": tools,
                 "disk": {path: dict(zip(("total", "used", "free"), shutil.disk_usage(path)))
                          for path in (str(Path.cwd()), "/data")},
                 "gpu": capture(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"]),
                 "cpu": capture(["getconf", "_NPROCESSORS_ONLN"]),
                 "memory": capture(["free", "-b"])}
    write_json(args.output / "resources.json", resources)
    junit = Path("outputs/veloroute_phase0/tests/junit.xml")
    suites = ET.parse(junit).getroot()
    cases = [{"name": case.attrib.get("name"), "classname": case.attrib.get("classname"),
              "seconds": case.attrib.get("time"),
              "status": "failed" if case.find("failure") is not None or case.find("error") is not None
                        else "skipped" if case.find("skipped") is not None else "passed"}
             for case in suites.iter("testcase")]
    write_json(args.output / "tests.json", {
        "junit": str(junit.resolve()), "sha256": sha256(junit), "cases": cases,
        "python": interpreters["scdfm"], "device": "CPU",
        "scope": "8 new structural checks plus 2 existing import/Sinkhorn checks; not full 63-test suite; not G1 diagnostics",
        "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/yuchang/veloFM/src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/yuchang/miniconda3/envs/scdfm/bin/python -B -m pytest tests/test_phase0_audit.py /home/yuchang/veloFM/tests/test_import.py /home/yuchang/veloFM/tests/test_sinkhorn_backend_parity.py -q -p no:cacheprovider --junitxml=/home/yuchang/wangjiaxuan/outputs/veloroute_phase0/tests/junit.xml",
        "new_test_sha256": sha256("tests/test_phase0_audit.py"),
    })
    write_json(args.output / "provenance.json", {
        "finished_utc": now(), "command": sys.argv, "script_sha256": sha256(__file__),
        "helper_sha256": sha256(Path(__file__).with_name("audit_renge_assets.py")),
        "state_changes": "outputs only; no environment installation; no training; no original data edits",
        "git_revision": None, "repository_revision_method": "source-file SHA-256 snapshot",
    })
    print(json.dumps({"output": str(args.output), "assets_checked": len(assets),
                      "source_files_hashed": len(code_hashes), "test_cases": len(cases)}))


if __name__ == "__main__":
    main()
