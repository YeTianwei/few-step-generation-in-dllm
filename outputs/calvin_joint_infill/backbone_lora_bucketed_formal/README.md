# CALVIN Joint Infill Backbone 适配实验报告

- 实验日期：`2026-03-31`
- 训练开始时间：`2026-03-31T19:33:57`
- 训练结束时间：`2026-03-31T19:34:14`

## 1. 实验目的
本次实验用于比较 coord_only、backbone_lora 和 backbone_lora_plus_coord 三种适配方式，判断瓶颈更偏向 backbone 还是 coordination。

## 2. 数据与配置
- 数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- train_mode：`backbone_lora`
- 训练样本：`64`
- 测试样本：`32`
- action 表示：`bucketed_int`
- action_bucket_count：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_formal`

## 3. 训练摘要
- 最终平均训练损失：`3.521240`
- 可训练参数量：`20185088`
- 保存产物：`{"backbone_adapter": "/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/backbone_lora_bucketed_formal/backbone_adapter"}`

## 4. 指标对比
| metric | baseline | trained | coordinated |
|---|---:|---:|---:|
| text_region_token_acc | 0.0712 | 0.0501 | - |
| action_region_token_acc | 0.0500 | 0.3973 | - |
| joint_region_token_acc | 0.0539 | 0.3065 | - |
| joint_region_exact | 0.0000 | 0.0000 | - |
| effective_steps | 8.0000 | 8.0000 | - |
| latency_sec | 0.7768 | 1.7024 | - |

## 5. 典型案例
| sample_id | baseline_joint | trained_joint | coordinated_joint | baseline_preview |
|---|---:|---:|---:|---|
| turn_off_led_1747210_1747274 | 0.0141 | 0.3333 | - | Instruction: toggle the button to turn off the green light Assistant response: the the the of the the the the the the of the the of the the of the the the th... |
| turn_on_led_1176935_1176999 | 0.0160 | 0.3232 | - | Instruction: turn on the led lamp Assistant response: the the the the the the the the the the the the the the the of the of the the the the the the of the th... |
| turn_on_led_0190346_0190410 | 0.0155 | 0.3219 | - | Instruction: turn on the led Assistant response: the the the the the the of the the the the the the the the the the the the the the the the the the the the t... |
| open_drawer_0638876_0638940 | 0.0122 | 0.3148 | - | Instruction: grasp the handle of the drawer and open it Assistant response: of the the of the the of the of the the the the the the of the the the the the th... |
| rotate_pink_block_left_0295233_0295297 | 0.0149 | 0.3169 | - | Instruction: take the pink block and rotate it left Assistant response: the the the the the the the the the the the the the the the the the the the the the t... |

## 6. 结论
- 如果 trained 明显优于 baseline，说明 backbone 适配比 coordination-only 更关键。
- 如果 coordinated 继续优于 trained，说明在 backbone 已适配后 coordination 仍有增益空间。
- 如果 trained 仍然接近 baseline，则更应优先降低动作表示难度而不是继续堆协调模块。
