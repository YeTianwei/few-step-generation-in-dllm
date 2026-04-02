# CALVIN Joint Infill Backbone 适配实验报告

- 实验日期：`2026-03-31`
- 训练开始时间：`2026-03-31T20:03:46`
- 训练结束时间：`2026-03-31T20:07:17`

## 1. 实验目的
本次实验用于比较 coord_only、backbone_lora 和 backbone_lora_plus_coord 三种适配方式，判断瓶颈更偏向 backbone 还是 coordination。

## 2. 数据与配置
- 数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- train_mode：`backbone_lora`
- 训练样本：`800`
- 测试样本：`200`
- action 表示：`bucketed_int`
- action_bucket_count：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_800_200`

## 3. 训练摘要
- 最终平均训练损失：`1.902852`
- 可训练参数量：`20185088`
- 保存产物：`{"backbone_adapter": "/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_800_200/backbone_adapter"}`

## 4. 指标对比
| metric | baseline | trained | coordinated |
|---|---:|---:|---:|
| text_region_token_acc | 0.0690 | 0.0618 | - |
| action_region_token_acc | 0.0680 | 0.7113 | - |
| joint_region_token_acc | 0.0656 | 0.5421 | - |
| joint_region_exact | 0.0000 | 0.0000 | - |
| effective_steps | 8.0000 | 8.0000 | - |
| latency_sec | 0.8325 | 1.7966 | - |

## 5. 典型案例
| sample_id | baseline_joint | trained_joint | coordinated_joint | baseline_preview |
|---|---:|---:|---:|---|
| push_blue_block_left_1690476_1690540 | 0.0155 | 0.6259 | - | Instruction: go push the blue block to the left Assistant response: the the, the the the the the the the the the of the the the the the the of the the the th... |
| turn_off_led_1747210_1747274 | 0.0141 | 0.6126 | - | Instruction: toggle the button to turn off the green light Assistant response: the the the of the the the the the the of the the of the the of the the the th... |
| push_blue_block_left_0375305_0375369 | 0.0180 | 0.6162 | - | Instruction: push left the blue block Assistant response: the, the, the the the the the the of the of the the the the the the the the the the the the the the... |
| turn_off_led_1684309_1684373 | 0.0155 | 0.6039 | - | Instruction: toggle the button to turn off the green light Assistant response: the the the the of the the the the the the the the the the the the the of the ... |
| place_in_slider_0801509_0801573 | 0.0197 | 0.6046 | - | Instruction: put the block in the slider Assistant response: the the the the of the of the the the the the the of the the the the of the the of the the of th... |

## 6. 结论
- 如果 trained 明显优于 baseline，说明 backbone 适配比 coordination-only 更关键。
- 如果 coordinated 继续优于 trained，说明在 backbone 已适配后 coordination 仍有增益空间。
- 如果 trained 仍然接近 baseline，则更应优先降低动作表示难度而不是继续堆协调模块。
