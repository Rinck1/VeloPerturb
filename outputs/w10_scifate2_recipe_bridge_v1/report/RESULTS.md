# W10 recipe bridge

All fixed recipes, arms, seeds and checkpoints are reported.

| Recipe | Steps | Real energy | vs static | vs shuffled |
|---|---:|---:|---:|---:|
| b0_w8_exact | 1000 | 0.619230 | -0.247% | -0.079% |
| b0_w8_exact | 4000 | 0.615184 | +1.306% | +1.481% |
| b0_w8_exact | 12000 | 0.603696 | +3.043% | +3.180% |
| b1_direct_distance | 1000 | 0.619107 | -0.229% | -0.057% |
| b1_direct_distance | 4000 | 0.616474 | +1.071% | +1.399% |
| b1_direct_distance | 12000 | 0.604896 | +2.657% | +3.123% |
| b2_full_center | 1000 | 0.614709 | -0.378% | -0.128% |
| b2_full_center | 4000 | 0.614725 | +0.399% | +1.150% |
| b2_full_center | 12000 | 0.616762 | +1.010% | +1.366% |
| b3_source_U | 1000 | 0.618869 | +0.633% | +1.020% |
| b3_source_U | 4000 | 0.615612 | +0.619% | +2.043% |
| b3_source_U | 12000 | 0.619138 | +1.747% | +2.846% |
| b4_full_regularization | 1000 | 0.619852 | +0.505% | +0.899% |
| b4_full_regularization | 4000 | 0.617094 | +0.157% | +1.877% |
| b4_full_regularization | 12000 | 0.618900 | +1.582% | +3.103% |
| b5_w9_exact | 1000 | 0.626985 | +0.331% | +0.343% |
| b5_w9_exact | 4000 | 0.621022 | -0.136% | +0.724% |
| b5_w9_exact | 12000 | 0.621074 | -0.089% | +1.509% |

The mean branch is frozen in all arms; this isolates local routing, not whole-model S-only.
