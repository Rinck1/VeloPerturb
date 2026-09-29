"""Consolidate measured acceptance evidence, environment and live raw-task snapshot."""
import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from veloroute.artifacts import Run, load_config, read_csv, save_csv, save_json, sha256, utc_now
from veloroute.protocol import protocol_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path("outputs")
    synthetic = root/"veloroute_core_synthetic_20260912_v2"
    smoke = root/"veloroute_core_smoke_20260912"
    raw = root/"veloroute_raw_day4"
    junit = root/"veloroute_core_tests/final_junit.xml"
    inputs = [synthetic/"decision.json", synthetic/"metrics.csv", synthetic/"acceptance_checks.csv",
              synthetic/"provenance.json", smoke/"summary.json", junit,
              raw/"layout_SRR21518305/layout.json", raw/"validate_SRR21518305/status.json",
              root/"veloroute_condition_reservation_20260912/condition_split.csv",
              Path("configs/veloroute_core.yaml"), Path("configs/veloroute_g1.draft.yaml"),
              Path("docs/veloroute_scope_amendment_20260912.md")]
    decision = json.loads((synthetic/"decision.json").read_text())
    summary = json.loads((smoke/"summary.json").read_text())
    layout = json.loads((raw/"layout_SRR21518305/layout.json").read_text())
    tests = list(ET.parse(junit).getroot().iter("testcase"))
    passed = sum(t.find("failure") is None and t.find("error") is None and t.find("skipped") is None for t in tests)
    if decision["status"] != "ENGINEERING_PASS" or summary["status"] != "ENGINEERING_PASS" or passed != len(tests):
        raise RuntimeError("Cannot report a completed core: an acceptance check failed")
    config = {"kind": "engineering_delivery", "user_approved_core_before_G1": True,
              "real_data_G1": "not_run", "synthetic_result_version": "v2_same_router_initialization"}
    with Run(args.output, stage="core_delivery", kind="engineering", config=config, inputs=inputs, seed=[0, 1, 2]) as run:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (run.directory/"environment.freeze.txt").write_text(freeze.stdout)
        pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        save_json(run.directory/"dependency_check.json", {"returncode": pip_check.returncode, "stdout": pip_check.stdout, "stderr": pip_check.stderr})
        if pip_check.returncode:
            raise RuntimeError("Dependency check failed")
        toolkit = Path("tools/downloads/sratoolkit.current-ubuntu64.tar.gz.part")
        sra = Path("data/renge/raw/sra/SRR21518305/SRR21518305.sra")
        save_json(run.directory/"raw_and_tools_provenance.json", {
            "checked_utc": utc_now(), "sra_toolkit_version": "3.4.1",
            "download_url": "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz",
            "official_md5_url": "https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/md5sum.txt",
            "official_and_local_md5_verified": "ec6e9056a2bfebcf23c6cd6e02951ef2",
            "toolkit_archive_sha256": sha256(toolkit),
            "SRR21518305": {"path": str(sra.resolve()), "sha256": sha256(sra), "bytes": sra.stat().st_size,
                              "library_type": "gRNA", "archive_validation": "vdb-validate passed",
                              "fastq_index_channel": 1, "fastq_barcode_channel": 2, "fastq_library_sequence_channel": 3},
            "sources": ["https://github.com/ncbi/sra-tools/wiki/01.-Downloading-SRA-Toolkit",
                        "https://www.ncbi.nlm.nih.gov/sra/SRR21518305"],
        })
        download_state = json.loads((raw/"prefetch_SRR21518308/status.json").read_text())
        if download_state.get("status") == "running":
            try:
                os.kill(download_state["pid"], 0)
                download_state["recorded_pid_alive_at_check"] = True
            except ProcessLookupError:
                download_state["recorded_pid_alive_at_check"] = False
        download_state["checked_utc"] = utc_now()
        download_state["observed_files"] = [{"path": str(p), "bytes": p.stat().st_size}
                                              for p in Path("data/renge/raw/sra/SRR21518308").glob("*") if p.is_file()]
        save_json(run.directory/"download_snapshot.json", download_state)
        metrics = [
            {"metric": "unit_tests_passed", "value": passed, "scope": "engineering"},
            {"metric": "synthetic_acceptance_passed", "value": decision["passed_checks"], "scope": "four_systems_three_seeds"},
            {"metric": "configured_parameters", "value": summary["parameters"], "scope": "default_core"},
            {"metric": "checkpoint_roundtrip_max_error", "value": summary["checkpoint_roundtrip_max_error"], "scope": "default_core"},
            {"metric": "gRNA_paired_reads_scanned", "value": layout["records_examined"], "scope": "SRR21518305"},
            {"metric": "barcode_match_offset0", "value": layout["published_barcode_match_fraction_by_zero_based_offset"]["0"], "scope": "SRR21518305"},
            {"metric": "barcode_match_offset1", "value": layout["published_barcode_match_fraction_by_zero_based_offset"]["1"], "scope": "SRR21518305"},
        ]
        save_csv(run.directory/"metrics.csv", metrics)
        save_json(run.directory/"formal_readiness.json", {"G1": "not_run", "missing": protocol_errors(load_config("configs/veloroute_g1.draft.yaml"))})
        (run.directory/"RESULTS.md").write_text(f"""# VeloRoute 核心框架交付（2026-09-12）

状态：核心网络已实现并完成工程验收；真实数据 G1 尚未运行。用户授权修改已记录。

## 已完成

- 源侧 velocity router → 共享骨干/两个低秩专家 → 模式锁定 RK4 → checkpoint 保存与恢复。
- 默认 50 维状态、50 维 velocity、256 维条件；{summary['parameters']:,} 个参数，含 {summary['router_parameters']:,} 个 router 参数。
- 已验证逐专家损失、梯度隔离、同条件责任度与 Sinkhorn、源侧输入限制、训练折 PCA、分层置换和条件级 bootstrap。
- 单元测试 {passed}/{len(tests)} 通过；四类合成系统 × 3 seeds，{decision['passed_checks']}/{decision['checks']} 项预设工程检查通过。
- 默认尺寸完成 3 个优化步骤、32×50 预测和 checkpoint 恢复，恢复前后最大误差为 {summary['checkpoint_roundtrip_max_error']}。
- 项目独立环境：NumPy 1.26.4 / PyTorch 2.6.0+cpu；依赖检查通过。共享环境没有修改。

## 合成结果的含义

相同表达、不同 velocity 的已知真值系统中，真实 velocity router 的路由准确率为 100%，静态/推理期置换对照均约 51%。

这只证明网络能使用所提供的动态信号完成工程任务。专家使用合成真实模式与路径监督，真实未配对数据没有该真值；静态混合也可能匹配同样的终态边际。不能据此宣称 G1-GO、真实命运恢复或论文效果。

当前没有实现/验证：真实 S/U→冻结 velocity 适配、RENGE 的未配对模式发现与完整训练、GFG/ESM2 来源审计、可靠度门控、自适应 K、CUDA 正式训练栈。

## 真实数据进度

- day4 gRNA `SRR21518305` 已下载、NCBI 校验通过、technical reads 完整提取；已扫描全部 {layout['records_examined']:,} 对 barcode/文库序列。
- `_1` 是 8 bp 样本索引、`_2` 是 26 bp barcode/UMI、`_3` 是 91 bp 文库序列。此处只认证该 run，不能直接替代全部文库的布局检查。
- barcode 起始位置匹配率约 88.26%，右移一位约 0.036%；不能把偶见首位 N 当作固定前缀剪掉。
- day4 表达 run `SRR21518308` 的本报告快照状态：`{download_state['status']}`。实时状态应查看 raw 任务的 status.json 或运行 `veloroute status`，不要把下载启动当成完成。
- 已按 TF 预留 14 个训练、4 个开发验证、5 个最终确认条件，两个 gRNA 和四个时间点保持同一角色。没有根据响应效果选条件；正式细胞标签与切分仍未完成。

## 入口与后续

- 使用说明：项目根目录 `README.md`。
- 核心代码：`src/veloroute/model.py`、`coupling.py`。
- 合成完整证据：`outputs/veloroute_core_synthetic_20260912_v2/`；第一版保留。v2 只增加 static/real router 相同初始化，不改验收阈值或预算。
- 最新测试：`outputs/veloroute_core_tests/final_junit.xml`。
- 默认尺寸烟测：`outputs/veloroute_core_smoke_20260912/`。
- 原始数据任务：`outputs/veloroute_raw_day4/`，含命令、PID、日志和返回码。

下一步优先完成表达文库与 S/U；然后实际数据接入、预注册和 G1。新增网络代码不能替代这些证据。
""")
    print(json.dumps({"output": args.output, "status": "CORE_ENGINEERING_DELIVERED", "tests": passed,
                      "synthetic_checks": decision["passed_checks"], "raw_download_status_at_snapshot": download_state["status"]}))


if __name__ == "__main__":
    main()
