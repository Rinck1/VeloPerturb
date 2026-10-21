# Traxler KO150：可训练数据与冻结 fold（2026-10-21）

把 Traxler 从"已定量 + 作者标签"推进到**模型可直接读取的 fold**。不读 KO 效应/DE/签名；split 在任何效应分析之前冻结。

## 1. 可训练 manifest（`outputs/w15_traxler_trainable_v1/`）
脚本 `scripts/phase0/w15_traxler_trainable.py`。

- `cells.csv.gz`：28,303 个作者 QC + 单 guide 细胞，字段含 `library/time/KO/source/S/U/QC`。
- 深度（作者 QC 细胞内）：
  | 时间 | 细胞 | median S | median U | median U/(S+U) | pooled U/(S+U) |
  |---|---:|---:|---:|---:|---:|
  | 0h | 12,176 | 9,639.5 | 675 | 6.32% | 6.91% |
  | 6h | 11,578 | 11,687.5 | 709 | 5.52% | 5.84% |
  | 24h | 4,549 | 23,338 | 1,770 | 7.09% | 7.55% |
- 来源（我们的 dominant-HTO ≥90% 规则，**非作者 GMM**）：a 9,002 / b 8,743 / c 7,095 / UNRESOLVED 3,463。

## 2. 冻结 split（源侧覆盖，不用效应）
| 任务 | KO 数 | train / val / test |
|---|---:|---|
| 0h→6h | 132（各时点 ≥30 细胞） | 106 / 13 / 13 |
| 6h→24h | 70 | 56 / 7 / 7 |

分层按 `min(0h,6h)` 覆盖四分位，seed 0；只在覆盖内随机。记录在 `split_0to6.json` / `split_6to24.json`。

## 3. 模型可读 fold
脚本 `scripts/phase0/w15_traxler_build_fold.py`。每个任务一个目录：

| 任务 | 目录 | train src/tgt | val src/tgt | test src/tgt |
|---|---|---|---:|---:|---:|
| 0h→6h | `outputs/w15_traxler_fold_0to6_v1` | 8,741 / 8,209 | 1,098 / 1,064 | 994 / 1,041 |
| 6h→24h | `outputs/w15_traxler_fold_6to24_v1` | 4,944 / 2,042 | 592 / 257 | 605 / 253 |

每个 fold 含：
- `{role}_source.npz`：`z, velocity, u_features, u_predicted, depth, cell_ids, conditions`
- `{role}_target.npz`：`z, cell_ids, conditions`
- `{role}_gene_source.npz`：`gene_us`（U 在前、S 在后，2000 选中基因）
- `transform.npz`、`conditions.npz`
- 2000 HVG，PCA-50，1867/1844 个 supported 速度基因。

**条件嵌入**：`conditions.npz` 用 **one-hot over KO**（0→6 为 132 维，6→24 为 70 维），作为 ESM 就绪前的占位；模型从 `embeddings.shape[1]` 自动读 `esm_dim`。ESM 生成待补。

## 4. 训练方式
现有 L2/L3 runner 直接指向 fold 目录 + `conditions.npz` 即可；三臂/对照仍用 `kin / S摘要 / direct-U / shuffled`。

## 5. 必须声明的限制
1. **来源 = dominant-HTO ≥90% 规则**，不是作者 GMM；`biological_replicate` 仍是**未解决**（a/b/c 是三个独立冻存管，不是三个供体）。
2. 作者逐细胞 HTO 判决未恢复（3,463 个 UNRESOLVED）。
3. **未读任何 KO 效应**；split 先冻结。
4. 条件嵌入是 one-hot 占位，不是 ESM。
5. 0h→6h 的 0h 源群体是"未处理 + KO"，velocity 在 0h 可能近似 0（需在训练时注意）。

## 6. 下一步
1. 生成 KO 的 ESM 嵌入替换 one-hot（需 ESM2 权重）。
2. 恢复/冻结 HTO 来源（若要做"来源留出"）。
3. 用 L2/L3 三臂在 0h→6h 上跑首轮，判据仍是 `kin > S摘要 且 > direct-U 且 > shuffled`。
