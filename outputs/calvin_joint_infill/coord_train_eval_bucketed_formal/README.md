# CALVIN Joint Infill 协调模块训练报告

- 实验日期：`2026-03-31`
- 训练开始时间：`2026-03-31T16:38:22`
- 训练结束时间：`2026-03-31T16:56:18`
- 评估时间：`2026-03-31T16:56:18`

## 1. 实验目的
本次实验在真实 CALVIN joint infill 样本上只训练 coordination module，并在固定 test split 上比较 baseline 与 coordinated 的区域恢复质量。

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 训练样本数：`800`
- test 样本数：`200`
- text 区域：`think`
- action 表示：`bucketed_int（8 桶）`
- token 长度过滤：`仅保留 target token <= 4096 的样本；保留 1000 条，过滤 0 条`
- split 说明：`固定 8:2 切分，train=800，test=200，seed=42`
- split 清单：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal/split_manifest.json`

## 3. 模型与配置
- 模型：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- 学习率：`0.001`
- epoch 数：`10`
- coord_tokens：`64`
- few_step_budget：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name "coord_train_eval_bucketed_formal"
```

## 5. 指标定义
- `epoch_loss`：每个 epoch 的平均 masked CE loss
- `*_region_token_acc`：对应区域 token 恢复率
- `joint_region_exact`：联合区域完全恢复比例

## 6. 训练结果
- 训练完成 epoch 数：`10`
- 最终平均训练损失：`4.606582`
- coord module 保存路径：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal/coordination_module`
- 训练设备：`cuda`

## 7. test 结果总表
| metric | baseline | coordinated | delta |
|---|---:|---:|---:|
| text_region_token_acc | 0.0690 | 0.0734 | 0.0044 |
| action_region_token_acc | 0.0680 | 0.1031 | 0.0351 |
| joint_region_token_acc | 0.0656 | 0.0931 | 0.0275 |
| joint_region_exact | 0.0000 | 0.0000 | 0.0000 |
| effective_steps | 8.0000 | 1.0000 | -7.0000 |
| latency_sec | 0.8243 | 0.1119 | -0.7124 |

## 8. 可视化结果
- loss 曲线：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal/figures/loss_curve.png`
- baseline/coordinated 指标对比图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal/figures/metric_compare.png`
- task-level delta 图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_bucketed_formal/figures/task_joint_delta.png`

## 9. 典型案例
| sample_id | joint_acc_baseline | joint_acc_coord | delta | coordinated_preview |
|---|---:|---:|---:|---|
| place_in_slider_0923613_0923661 | 0.0387 | 0.2454 | 0.2067 | Instruction: put the object in the cabinet Assistant response:,,, the the the the the the the the the the the the the the the the the the the the t... |
| move_slider_right_1180855_1180919 | 0.0288 | 0.2310 | 0.2022 | Instruction: slide the door to the right Assistant response:,,,, the, the the the the the the the the the the the the,, the the the the the the the... |
| open_drawer_1318630_1318694 | 0.0269 | 0.2117 | 0.1848 | Instruction: go open the drawer Assistant response:,,,, the,, the, the the the the the the the the the the the the the the the,,, the the the the t... |
| unstack_block_0433390_0433454 | 0.0288 | 0.2085 | 0.1797 | Instruction: take off the stacked block Assistant response:,,,,,,, the the the the the the the the the the the,,, the the the the the the the the t... |
| unstack_block_1557311_1557375 | 0.0318 | 0.1873 | 0.1554 | Instruction: collapse the stacked blocks Assistant response:,,, the the the the the the the the the the the the the the, the the the the the the, t... |

## 10. Task-Level 摘要
| task | count | delta_joint |
|---|---:|---:|
| push_blue_block_left | 3 | 0.1373 |
| unstack_block | 7 | 0.0596 |
| push_into_drawer | 6 | 0.0586 |
| place_in_drawer | 8 | 0.0583 |
| move_slider_right | 12 | 0.0493 |
| turn_on_led | 9 | 0.0465 |
| stack_block | 9 | 0.0432 |
| place_in_slider | 13 | 0.0375 |
| open_drawer | 18 | 0.0346 |
| push_red_block_right | 1 | 0.0341 |

## 11. 结论与下一步
- 如果训练后的 coordinated 在 test split 上优于 baseline，说明 coordination module 在真实 CALVIN 标注上具备学习价值。
- 如果提升有限，可以先缩短 action 序列、调整四舍五入位数，或增加 coord_tokens/few_step_budget 再继续验证。
- 这仍然是代理表示训练，不是最终离散 action token/block 方案。
