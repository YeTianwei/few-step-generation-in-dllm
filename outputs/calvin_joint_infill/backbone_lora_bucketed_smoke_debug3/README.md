# CALVIN Joint Infill Backbone 适配实验报告

- 实验日期：`2026-03-31`
- 训练开始时间：`2026-03-31T17:40:49`
- 训练结束时间：`2026-03-31T17:40:50`

## 1. 实验目的
本次实验用于比较 coord_only、backbone_lora 和 backbone_lora_plus_coord 三种适配方式，判断瓶颈更偏向 backbone 还是 coordination。

## 2. 数据与配置
- 数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- train_mode：`backbone_lora`
- 训练样本：`2`
- 测试样本：`2`
- action 表示：`bucketed_int`
- action_bucket_count：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_smoke_debug3`

## 3. 训练摘要
- 最终平均训练损失：`6.843750`
- 可训练参数量：`20185088`
- 保存产物：`{"backbone_adapter": "/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_smoke_debug3/backbone_adapter"}`

## 4. 指标对比
| metric | baseline | trained | coordinated |
|---|---:|---:|---:|
| text_region_token_acc | 0.0454 | 0.0123 | - |
| action_region_token_acc | 0.1993 | 0.0030 | - |
| joint_region_token_acc | 0.1455 | 0.0058 | - |
| joint_region_exact | 0.0000 | 0.0000 | - |
| effective_steps | 8.0000 | 8.0000 | - |
| latency_sec | 0.7279 | 1.4889 | - |

## 5. 典型案例
| sample_id | baseline_joint | trained_joint | coordinated_joint | baseline_preview |
|---|---:|---:|---:|---|
| rotate_blue_block_left_1504812_1504876 | 0.0127 | 0.0056 | - | Instruction: grasp the blue block, then turn it left Assistant response: the of the of the the the the of the the the of the of the the the the the the the t... |
| turn_off_lightbulb_1430331_1430368 | 0.2782 | 0.0060 | - | Instruction: turn off the yellow light Assistant response:, the the the the the the the, the the the the the, the the, the the the the the the the,,, the the... |

## 6. 结论
- 如果 trained 明显优于 baseline，说明 backbone 适配比 coordination-only 更关键。
- 如果 coordinated 继续优于 trained，说明在 backbone 已适配后 coordination 仍有增益空间。
- 如果 trained 仍然接近 baseline，则更应优先降低动作表示难度而不是继续堆协调模块。
