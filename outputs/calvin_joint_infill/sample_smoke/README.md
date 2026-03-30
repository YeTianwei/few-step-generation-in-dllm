# CALVIN Joint Infill 采样对比报告

## 1. 实验目的
本次实验用真实 CALVIN 标注数据验证 joint infill 链路是否打通，并比较：

- `enable_coordination=False`
- `enable_coordination=True`

关注点是 text/action 两个区域的联合恢复质量，而不是最终任务成功率。

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 采样样本数：`1`
- text 区域：`think`
- action 区域：四舍五入到 `4` 位后的原始动作串
- token 长度过滤：`仅保留 target token <= 4096 的样本；保留 1000 条，过滤 0 条`

## 3. 模型与配置
- 模型：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- coord module：`/data/ytw/VLA_baseline/dllm/.models/a2d/proxy-coordination-nl`
- sampler steps：`24`
- few_step_budget：`8`
- coord_tokens：`64`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/sample_smoke`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_sample.py --experiment_name "sample_smoke"
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
| text_region_token_acc | 0.0289 | 0.0318 | 0.0029 |
| action_region_token_acc | 0.0774 | 0.0774 | 0.0000 |
| joint_region_token_acc | 0.0727 | 0.0730 | 0.0003 |
| joint_region_exact | 0.0000 | 0.0000 | 0.0000 |
| effective_steps | 8.0000 | 1.0000 | -7.0000 |
| latency_sec | 3.5100 | 0.5366 | -2.9734 |

## 7. 可视化结果
- 指标对比图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/sample_smoke/figures/metric_compare.png`

## 8. 典型案例
| sample_id | joint_acc_baseline | joint_acc_coord | delta | baseline_preview | coordinated_preview |
|---|---:|---:|---:|---|---|
| lift_pink_block_slider_0981473_0981537 | 0.0727 | 0.0730 | 0.0003 | Instruction: pick up the pink block in the sliding cabinet Assistant response:,, the the,, the,, the the,,,, the,, the the,, the,,,,,,,,,,,,,,,,,,,... | Instruction: pick up the pink block in the sliding cabinet Assistant response:,, the the the the the,, the the,, the, the the the the the, the the,... |

## 9. 结论与下一步
- 如果 `joint_region_token_acc` 或 `action_region_token_acc` 在 coordinated 下高于 baseline，说明真实 CALVIN 数据上已经出现了 text/action 协调恢复的信号。
- 如果增益不稳定，优先检查样本长度、动作序列化粒度和 coord module 初始化来源，而不是直接否定 joint infill 思路。
- 本轮仍然是代理表示验证；action 还没有进入最终离散 token/block 设计。