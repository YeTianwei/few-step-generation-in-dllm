# CALVIN Joint Infill 协调模块训练报告

## 1. 实验目的
本次实验在真实 CALVIN joint infill 样本上只训练 coordination module，并在固定 holdout 上比较 baseline 与 coordinated 的区域恢复质量。

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 训练样本数：`4`
- holdout 样本数：`1`
- text 区域：`think`
- action 区域：四舍五入到 `4` 位后的原始动作串
- token 长度过滤：`仅保留 target token <= 4096 的样本；保留 1000 条，过滤 0 条`

## 3. 模型与配置
- 模型：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- 学习率：`0.001`
- epoch 数：`1`
- coord_tokens：`64`
- few_step_budget：`8`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/fit_smoke`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_fit.py --experiment_name "fit_smoke"
```

## 5. 指标定义
- `epoch_loss`：每个 epoch 的平均 masked CE loss
- `*_region_token_acc`：对应区域 token 恢复率
- `joint_region_exact`：联合区域完全恢复比例

## 6. 训练结果
- 训练完成 epoch 数：`1`
- 最终平均训练损失：`6.578125`
- coord module 保存路径：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/fit_smoke/coordination_module`

## 7. holdout 结果总表
| metric | baseline | coordinated | delta |
|---|---:|---:|---:|
| text_region_token_acc | 0.0334 | 0.0334 | 0.0000 |
| action_region_token_acc | 0.0553 | 0.0553 | 0.0000 |
| joint_region_token_acc | 0.0534 | 0.0534 | 0.0000 |
| joint_region_exact | 0.0000 | 0.0000 | 0.0000 |
| effective_steps | 8.0000 | 8.0000 | 0.0000 |
| latency_sec | 3.0882 | 3.1575 | 0.0693 |

## 8. 可视化结果
- loss 曲线：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/fit_smoke/figures/loss_curve.png`
- baseline/coordinated 指标对比图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/fit_smoke/figures/metric_compare.png`

## 9. 典型案例
| sample_id | joint_acc_baseline | joint_acc_coord | delta | coordinated_preview |
|---|---:|---:|---:|---|
| rotate_red_block_right_0761558_0761622 | 0.0534 | 0.0534 | 0.0000 | Instruction: rotate the red block 90 degrees to the right Assistant response:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |

## 10. 结论与下一步
- 如果训练后的 coordinated 在 holdout 上优于 baseline，说明 coordination module 在真实 CALVIN 标注上具备学习价值。
- 如果提升有限，可以先缩短 action 序列、调整四舍五入位数，或增加 coord_tokens/few_step_budget 再继续验证。
- 这仍然是代理表示训练，不是最终离散 action token/block 方案。