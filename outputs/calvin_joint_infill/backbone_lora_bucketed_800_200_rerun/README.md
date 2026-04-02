# CALVIN Joint Infill Backbone 适配实验报告

- 实验日期：`2026-04-02`
- 训练开始时间：`2026-04-02T19:35:09`
- 训练结束时间：`2026-04-02T19:38:38`

## 1. 实验目的
本次实验用于比较 coord_only、backbone_lora 和 backbone_lora_plus_coord 三种适配方式，判断瓶颈更偏向 backbone 还是 coordination。

## 2. 数据与配置
- 数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- train_mode：`backbone_lora`
- 训练样本：`800`
- 测试样本：`200`
- action 表示：`bucketed_int`
- action_bucket_count：`8`
- 输出目录：`/data/ytw/VLA_baseline/few-step-generation-in-dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_800_200_rerun`

## 3. 训练摘要
- 最终平均训练损失：`1.906152`
- 可训练参数量：`20185088`
- 保存产物：`{"backbone_adapter": "/data/ytw/VLA_baseline/few-step-generation-in-dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_800_200_rerun/backbone_adapter"}`

## 4. 指标对比
| metric | baseline | trained | coordinated |
|---|---:|---:|---:|
| text_region_token_acc | 0.0690 | 0.0634 | - |
| action_region_token_acc | 0.0680 | 0.7047 | - |
| joint_region_token_acc | 0.0656 | 0.5377 | - |
| joint_region_exact | 0.0000 | 0.0000 | - |
| effective_steps | 8.0000 | 8.0000 | - |
| latency_sec | 0.8381 | 1.8140 | - |

## 5. 典型案例
| sample_id | baseline_joint | trained_joint | coordinated_joint | baseline_preview |
|---|---:|---:|---:|---|
| turn_on_led_0586763_0586827 | 0.0161 | 0.6247 | - | Instruction: turn on the led Assistant response: the the the the the the the the the the the the the the the the the the the the the the the the the the the ... |
| push_blue_block_left_1690476_1690540 | 0.0155 | 0.6112 | - | Instruction: go push the blue block to the left Assistant response: the the, the the the the the the the the the of the the the the the the of the the the th... |
| move_slider_left_1779820_1779884 | 0.0210 | 0.6165 | - | Instruction: move the door all the way to the left Assistant response: the the the the the the of the the the the of the the the the the the the of the the t... |
| push_blue_block_left_0375305_0375369 | 0.0180 | 0.6097 | - | Instruction: push left the blue block Assistant response: the, the, the the the the the the of the of the the the the the the the the the the the the the the... |
| open_drawer_0180794_0180858 | 0.0102 | 0.5990 | - | Instruction: go open the drawer Assistant response: the the the the the the the the of the the the the the the the of the the the the the the the of the the ... |

## 6. 结论
- 如果 trained 明显优于 baseline，说明 backbone 适配比 coordination-only 更关键。
- 如果 coordinated 继续优于 trained，说明在 backbone 已适配后 coordination 仍有增益空间。
- 如果 trained 仍然接近 baseline，则更应优先降低动作表示难度而不是继续堆协调模块。
