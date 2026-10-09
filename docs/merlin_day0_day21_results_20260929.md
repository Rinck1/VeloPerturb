# MeRLin Day0→Day21：GFG velocity 纵向命运探针

## 数据合同

- Day0：SRR33960310；SRR33960311 被审计为同一 10x 文库的重复测序，不作为独立生物复本合并。
- Day21：SRR33960308；SRR33960309 尚未量化，因此没有被拼入结果。
- 克隆条码：只保留 `n_barcodes==1` 的无歧义 assignment；保留 `clone_reads>=1`，并将更高 reads 阈值作为后续敏感性分析，不在本次改变主样本。
- 共享克隆：376（两侧 QC 后再取交集）；按克隆切分 train/validation/confirmation = 226/75/75。
- 质量控制：克隆 assignment `n_barcodes==1`，并要求每个细胞 `S+U>=100`。
- pack：`/data/yuchang/veloroute_merlin_day0_day21_fold_20260929_v8/`。
- source 只含 Day0 S/U、PCA-50、稳态基线 velocity；target 只含 Day21 S 的 PCA。Day21 U/velocity 未导出。
- confirmation target 没有打开；本结果是 validation-only exploratory probe。

## GFG probe

输出：`/data/yuchang/veloroute_merlin_day0_gfg_fate_probe_20260929_gpu_v6/`。

GFG 使用冻结 checkpoint `a9d9fa39063312230e4ba64309e9264ba3c2cb7ac549f04544b366df57c1340b`；U/S 先按全基因 spliced library size 归一化到 10,000，再截取训练折冻结的 2,000 基因。U shuffled 只在同一 Day0 测序 run 内打乱。Day21 程序分数由 S-only 表达构造，仅作为未来 outcome，不进入 source。

| validation arm | normalized cell MSE | clone-equal MSE |
|---|---:|---:|
| z-only | 0.9950 | 0.01482 |
| z + simple velocity | 1.0491 | 0.01545 |
| z + GFG velocity | 1.0035 | 0.01487 |
| z + GFG shuffled-U | 1.0121 | 0.01489 |

## 判决

当前严格 QC 数据不支持“GFG velocity 为 Day21 命运 router 提供增量”的主张：

1. GFG 没有优于 z-only；
2. GFG 略优于 shuffled-U，但这个差异没有克服 z-only，也没有形成稳定的任务增量证据；
3. simple velocity 明显恶化，说明把未校准速度直接拼入 fate predictor 不稳定；
4. confirmation 仍封存，因此没有通过确认性门槛。

这不是 velocity 的信息论不可能证明，而是当前 MeRLin Day0→Day21、GFG checkpoint、S-only outcome 和线性增量探针下的严格阴性结果。当前 velocity 的安全身份应保持为 source-side reliability/quality prior，而不是 fate label 或 router 命运依据。

## 下一步

不进入 full router/专家网络主线训练，不打开 confirmation 进行追逐式调参。309 的 SRA/FASTQ 已完成，当前正在提取其 clone barcode；完成后做 308↔309 技术复本一致性/敏感性审计。若复本审计仍不提供动态增量，再预注册一个不依赖表达派生程序的 endpoint。当前项目的 router 破局证据仍需来自真正有动态任务增量的数据，而不是继续扩大 K 或 EM。
