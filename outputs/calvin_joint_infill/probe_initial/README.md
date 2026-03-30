# CALVIN Joint Infill 数据探查报告

## 1. 实验目的
本次实验用于验证 `/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl` 是否适合直接构造成 joint infill 样本，并检查：

- 原始 JSONL 是否可以稳定读取
- `think + serialized actions` 的样本构造是否稳定
- text/action 两个区域是否能被 marker 正确定位
- 数据长度和任务分布是否适合后续的小规模验证

## 2. 数据来源与样本构造
- 原始数据路径：`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`
- 原始数据只读：是
- 本次扫描样本数：`1000`
- text 区域：`think`
- action 区域：`actions` 四舍五入到 `4` 位后序列化
- 动作序列格式：`[a1,...,a7]; [a1,...,a7]; ...`
- 本次是否做截断：`无过滤，仅做全量探查`

## 3. 模型与配置
- tokenizer：`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`
- 随机种子：`42`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/probe_initial`

## 4. 运行命令
```bash
cd /data/ytw/VLA_baseline/dllm
/home/timer/miniconda3/envs/dllm/bin/python /data/ytw/VLA_baseline/dllm/examples/a2d/bd3lm/calvin_joint_infill_probe.py --experiment_name "probe_initial"
```

## 5. 指标定义
- `think_char_stats`：`think` 字符长度分布
- `action_step_stats`：动作步数分布
- `target_token_stats`：联合样本字符串的 token 长度分布
- `mask_validation_pass_rate`：marker 能成功切出 text/action 区域的比例

## 6. 结果总表
- 样本总数：`1000`
- 任务数：`34`
- mask 验证通过率：`1.0000`
- think 长度统计：`{"min": 884.0, "p50": 1289.0, "p90": 1488.0, "p95": 1554.0, "max": 1918.0, "mean": 1295.092}`
- action 步数统计：`{"min": 33.0, "p50": 64.0, "p90": 64.0, "p95": 64.0, "max": 64.0, "mean": 58.186}`
- target token 长度统计：`{"min": 1902.0, "p50": 3508.0, "p90": 3569.0, "p95": 3583.0, "max": 3656.0, "mean": 3231.155}`

Top-10 任务分布：

| task | count |
|---|---:|
| open_drawer | 74 |
| place_in_drawer | 59 |
| move_slider_left | 58 |
| move_slider_right | 58 |
| place_in_slider | 57 |
| turn_on_led | 56 |
| close_drawer | 52 |
| turn_off_lightbulb | 48 |
| turn_on_lightbulb | 45 |
| turn_off_led | 44 |

## 7. 可视化结果
- 任务分布柱状图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/probe_initial/figures/task_distribution.png`
- think 长度分布图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/probe_initial/figures/think_length_hist.png`
- action 步数分布图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/probe_initial/figures/action_steps_hist.png`
- target token 长度分布图：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/probe_initial/figures/target_tokens_hist.png`

## 8. 典型样本预览
| sample_id | task_name | action_steps | think_preview |
|---|---|---:|---|
| lift_pink_block_slider_0981473_0981537 | lift_pink_block_slider | 64 | Subgoal A is to open the sliding cabinet and locate the pink block inside, and Subgoal B is to grasp and extract the pink block from the cabinet. Step 1: the arm moves up and left while advancing forward (0.13s–1.33s)... |
| turn_off_led_1054784_1054848 | turn_off_led | 64 | Subgoal A is to position the gripper over the LED light’s control interface, and Subgoal B is to execute a toggle action that deactivates the light. Step 1: from 0.13s–0.27s, the gripper opens and the arm moves backwa... |
| unstack_block_0205584_0205648 | unstack_block | 64 | Subgoal A is to approach and position the gripper above the top block of the tower, and Subgoal B is to grasp and remove that top block without disturbing lower blocks. Step 1 (0.00s–0.53s): the arm opens the gripper ... |
| lift_pink_block_slider_0273722_0273786 | lift_pink_block_slider | 64 | Subgoal A is to locate and approach the pink block inside the open cabinet, and Subgoal B is to grasp and extract it. Step 1: from 0.13s–1.33s, the arm moves right to reposition laterally toward the cabinet’s interior... |
| rotate_red_block_right_0761558_0761622 | rotate_red_block_right | 64 | Subgoal A is to grasp the red block securely, Subgoal B is to rotate it 90 degrees clockwise while maintaining grip, and Subgoal C is to release it in the rotated orientation. Step 1: from 0.47s–0.87s, the gripper clo... |
| lift_pink_block_table_1006701_1006747 | lift_pink_block_table | 46 | Subgoal A is to position the gripper above the pink block for controlled grasping, and Subgoal B is to lift the block vertically while maintaining secure contact. Step 1: from 0.53s–1.00s, the arm moves upward and ope... |
| push_into_drawer_0471231_0471278 | push_into_drawer | 47 | Subgoal A is to secure the block for controlled transfer, and Subgoal B is to insert it into the drawer by sweeping motion followed by release inside the target region. Step 1: the gripper closes between 0.00s–0.47s (... |
| open_drawer_0766733_0766797 | open_drawer | 64 | Subgoal A is to position the gripper at the drawer handle for secure contact, and Subgoal B is to execute a pulling motion that translates the drawer outward while maintaining grasp. Step 1: from 0.07s–1.93s, the arm ... |
| place_in_drawer_0837034_0837067 | place_in_drawer | 33 | Subgoal A is to reposition the arm so the block can be lowered into the open drawer, and Subgoal B is to release the block inside the drawer cavity. Step 1: from 0.07s to 1.40s, the arm moves backward to clear the tab... |
| lift_red_block_drawer_1235928_1235992 | lift_red_block_drawer | 64 | Subgoal A is to locate and approach the red block inside the drawer, and Subgoal B is to grasp and extract it. Step 1 (0.00s–2.27s): the arm remains stationary while the gripper camera confirms the red block’s positio... |

## 9. 结论与下一步
- 全量 1000 条样本都可以被读取，并且可以稳定构造成 `Instruction + think + serialized actions` 的联合样本。
- marker 验证通过率如果接近 1，说明现有 coord_proxy 的 text/action 区域切分逻辑可以直接复用。
- 这份数据已经足以支持后续的小规模 joint infill 采样对比和 coordination module 训练。