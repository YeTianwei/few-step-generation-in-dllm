# CALVIN Joint Infill Baseline 测试报告

- 实验日期：`2026-03-27`
- 运行时间：`2026-03-27T17:14:49`

## 1. 实验目的
本次实验在固定 test split 上评估 baseline 的 joint infill 恢复质量。

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 评估样本数：`200`
- text 区域：`think`
- action 区域：四舍五入到 `4` 位后的原始动作串
- token 长度过滤：`仅保留 target token <= 4096 的样本；保留 1000 条，过滤 0 条`
- split 说明：`固定 8:2 切分，train=800，test=200，seed=42`
- split 清单：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/baseline_eval_formal/split_manifest.json`

## 3. 模型与配置
- 模型：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- coord module：`None`
- sampler steps：`24`
- few_step_budget：`8`
- coord_tokens：`64`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/baseline_eval_formal`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_eval.py --experiment_name "baseline_eval_formal"
```

## 5. 指标定义
- `text_region_token_acc`：text 区域 token 命中率
- `action_region_token_acc`：action 区域 token 命中率
- `joint_region_token_acc`：text+action 联合区域 token 命中率
- `joint_region_exact`：联合区域完全一致比例
- `effective_steps`：有效去噪步数
- `latency_sec`：单条样本推理时间

## 6. 结果总表
| metric | baseline | coordinated | delta |
|---|---:|---:|---:|
| text_region_token_acc | 0.0466 | - | - |
| action_region_token_acc | 0.0440 | - | - |
| joint_region_token_acc | 0.0449 | - | - |
| joint_region_exact | 0.0000 | - | - |
| effective_steps | 8.0000 | - | - |
| latency_sec | 2.9988 | - | - |

## 7. 可视化结果
- 指标对比图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/baseline_eval_formal/figures/metric_compare.png`
- task-level 图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/baseline_eval_formal/figures/task_joint_bar.png`

## 8. 典型案例
| sample_id | joint_acc_baseline | baseline_preview |
|---|---:|---|
| rotate_blue_block_left_1504812_1504876 | 0.0601 | Instruction: grasp the blue block, then turn it left Assistant response:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |
| turn_off_lightbulb_1430331_1430368 | 0.0085 | Instruction: turn off the yellow light Assistant response: the the the the the the the the the the the the the the the the the the the the the the ... |
| rotate_red_block_left_1776846_1776910 | 0.0601 | Instruction: take the red block and turn it left Assistant response:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |
| open_drawer_0170122_0170186 | 0.0509 | Instruction: open the drawer Assistant response:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |
| open_drawer_0638876_0638940 | 0.0499 | Instruction: grasp the handle of the drawer and open it Assistant response:,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,... |

## 9. Task-Level 摘要
| task | count | baseline_joint |
|---|---:|---:|
| push_pink_block_left | 1 | 0.0627 |
| lift_blue_block_table | 7 | 0.0619 |
| rotate_pink_block_left | 4 | 0.0616 |
| rotate_red_block_left | 3 | 0.0604 |
| push_blue_block_left | 3 | 0.0602 |
| move_slider_left | 13 | 0.0586 |
| push_into_drawer | 6 | 0.0559 |
| lift_red_block_slider | 7 | 0.0548 |
| close_drawer | 8 | 0.0544 |
| lift_blue_block_drawer | 6 | 0.0541 |

## 10. 结论与下一步
- 这份报告作为正式 baseline 参考线，后续所有 coordinated 结果都必须与它在同一 test split 上比较。
- 如果 baseline 指标已经极高，coordination 的提升空间会被压缩；如果 baseline 很低，则更应关注 task-level 的稳定性而不是单点案例。
- 正式结论以与 coordinated 的同 split 对比为准。
