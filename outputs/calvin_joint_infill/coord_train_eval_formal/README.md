# CALVIN Joint Infill 协调模块训练报告

- 实验日期：`2026-03-27`
- 训练开始时间：`2026-03-27T17:15:54`
- 训练结束时间：`2026-03-27T18:12:50`
- 评估时间：`2026-03-27T18:12:50`

## 1. 实验目的
本次实验在真实 CALVIN joint infill 样本上只训练 coordination module，并在固定 test split 上比较 baseline 与 coordinated 的区域恢复质量。

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 训练样本数：`800`
- test 样本数：`200`
- text 区域：`think`
- action 区域：四舍五入到 `4` 位后的原始动作串
- token 长度过滤：`仅保留 target token <= 4096 的样本；保留 1000 条，过滤 0 条`
- split 说明：`固定 8:2 切分，train=800，test=200，seed=42`
- split 清单：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal/split_manifest.json`

## 3. 模型与配置
- 模型：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- 学习率：`0.001`
- epoch 数：`10`
- coord_tokens：`64`
- few_step_budget：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name "coord_train_eval_formal"
```

## 5. 指标定义
- `epoch_loss`：每个 epoch 的平均 masked CE loss
- `*_region_token_acc`：对应区域 token 恢复率
- `joint_region_exact`：联合区域完全恢复比例

## 6. 训练结果
- 训练完成 epoch 数：`10`
- 最终平均训练损失：`3.726738`
- coord module 保存路径：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal/coordination_module`
- 训练设备：`cuda`

## 7. test 结果总表
| metric | baseline | coordinated | delta |
|---|---:|---:|---:|
| text_region_token_acc | 0.0466 | 0.0505 | 0.0039 |
| action_region_token_acc | 0.0440 | 0.0436 | -0.0004 |
| joint_region_token_acc | 0.0449 | 0.0450 | 0.0001 |
| joint_region_exact | 0.0000 | 0.0000 | 0.0000 |
| effective_steps | 8.0000 | 8.0000 | 0.0000 |
| latency_sec | 2.7629 | 2.8542 | 0.0913 |

## 8. 可视化结果
- loss 曲线：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal/figures/loss_curve.png`
- baseline/coordinated 指标对比图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal/figures/metric_compare.png`
- task-level delta 图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/coord_train_eval_formal/figures/task_joint_delta.png`

## 9. 典型案例
| sample_id | joint_acc_baseline | joint_acc_coord | delta | coordinated_preview |
|---|---:|---:|---:|---|
| place_in_slider_0840783_0840828 | 0.0028 | 0.0107 | 0.0079 | Instruction: put the object in the cabinet Assistant response: the the the the the the the the the the the the the the the the the the the the the ... |
| turn_on_lightbulb_1646415_1646462 | 0.0023 | 0.0102 | 0.0079 | Instruction: move up the switch Assistant response:, the the the, the the the the the the the the the the the the the the the the the the the the t... |
| place_in_slider_1103985_1104033 | 0.0062 | 0.0138 | 0.0076 | Instruction: put the block in the cabinet Assistant response: the the the the the the the the the the the the the the the the the the the the the t... |
| unstack_block_0501228_0501267 | 0.0057 | 0.0128 | 0.0071 | Instruction: unstack the blocks Assistant response:,, the the the, the the the the the the the,,, the the the the the the the the the the the the t... |
| lift_red_block_drawer_1684982_1685029 | 0.0053 | 0.0113 | 0.0060 | Instruction: lift the red block in the drawer Assistant response: the the the the the the the the the the the the the the the the the the the the t... |

## 10. Task-Level 摘要
| task | count | delta_joint |
|---|---:|---:|
| lift_pink_block_drawer | 3 | 0.0018 |
| place_in_slider | 13 | 0.0017 |
| lift_red_block_drawer | 4 | 0.0015 |
| turn_off_lightbulb | 10 | 0.0014 |
| unstack_block | 7 | 0.0012 |
| place_in_drawer | 8 | 0.0009 |
| turn_on_led | 9 | 0.0003 |
| lift_pink_block_table | 4 | 0.0003 |
| lift_pink_block_slider | 6 | 0.0003 |
| push_pink_block_right | 2 | 0.0002 |

## 11. 结论与下一步
- 如果训练后的 coordinated 在 test split 上优于 baseline，说明 coordination module 在真实 CALVIN 标注上具备学习价值。
- 如果提升有限，可以先缩短 action 序列、调整四舍五入位数，或增加 coord_tokens/few_step_budget 再继续验证。
- 这仍然是代理表示训练，不是最终离散 action token/block 方案。
