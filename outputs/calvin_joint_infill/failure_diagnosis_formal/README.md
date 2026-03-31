# CALVIN Joint Infill 失败模式诊断报告

- 生成日期：`2026-03-30`
- 生成时间：`2026-03-30T16:52:14`
- 输出目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill/failure_diagnosis_formal`
- 扫描根目录：`/data/ytw/VLA_baseline/dllm/outputs/calvin_joint_infill`

## 1. 结论摘要
- 主结论：协调模块对 text 有轻微帮助，但 action 侧没有同步改善，joint 指标几乎没变。
- 配对实验数量：`3`
- case 总数：`402`
- 预测未变化比例：`58.42%`
- `no_change` 标签占比：`69.80%`
- `repetition_collapse` 标签占比：`99.50%`
- `punctuation_collapse` 标签占比：`72.77%`
- `text_only_gain` 标签占比：`23.27%`

## 2. 扫描到的实验
- `baseline_eval_formal`: cases=200, paired=False, role=baseline_only
- `coord_train_eval_formal`: cases=200, paired=True, role=paired
- `fit_smoke`: cases=1, paired=True, role=smoke
- `probe_initial`: cases=10, paired=False, role=probe
- `sample_smoke`: cases=1, paired=True, role=smoke

## 3. 长度分桶
- 分桶边界：`[1932.0, 2526.0, 3473.0, 3500.0, 3528.0, 3611.0]`

| bucket | count | baseline joint | coord joint | delta | coord repeat | coord punct | coord unique |
|---|---:|---:|---:|---:|---:|---:|---:|
| [1932, 2526) | 80 | 0.0138 | 0.0125 | -0.0013 | 0.5710 | 0.1865 | 0.3350 |
| [2526, 3473) | 80 | 0.0375 | 0.0388 | 0.0014 | 0.7241 | 0.5811 | 0.2184 |
| [3473, 3500) | 78 | 0.0595 | 0.0597 | 0.0002 | 0.7940 | 0.8066 | 0.1636 |
| [3500, 3528) | 83 | 0.0556 | 0.0557 | 0.0001 | 0.7402 | 0.7474 | 0.1882 |
| [3528, 3611] | 81 | 0.0582 | 0.0585 | 0.0003 | 0.7748 | 0.7894 | 0.1667 |

## 4. 任务 Delta 排名

### 提升最多
| task | count | delta joint | delta text | delta action |
|---|---:|---:|---:|---:|
| lift_pink_block_drawer | 3 | 0.0018 | 0.0165 | 0.0000 |
| place_in_slider | 13 | 0.0017 | 0.0155 | 0.0000 |
| lift_red_block_drawer | 4 | 0.0015 | 0.0130 | 0.0000 |
| turn_off_lightbulb | 10 | 0.0014 | 0.0122 | 0.0003 |
| unstack_block | 7 | 0.0012 | 0.0097 | -0.0002 |
| place_in_drawer | 8 | 0.0009 | 0.0089 | 0.0000 |
| turn_on_led | 9 | 0.0003 | 0.0016 | 0.0001 |
| lift_pink_block_table | 4 | 0.0003 | 0.0029 | 0.0000 |
| lift_pink_block_slider | 7 | 0.0003 | 0.0028 | 0.0000 |
| push_pink_block_right | 2 | 0.0002 | 0.0017 | 0.0000 |

### 回退最多
| task | count | delta joint | delta text | delta action |
|---|---:|---:|---:|---:|
| turn_on_lightbulb | 6 | -0.0055 | 0.0155 | -0.0083 |
| stack_block | 9 | -0.0021 | 0.0012 | -0.0026 |
| move_slider_left | 13 | -0.0002 | -0.0000 | -0.0002 |
| move_slider_right | 12 | -0.0001 | -0.0003 | -0.0001 |
| rotate_blue_block_left | 4 | 0.0000 | 0.0000 | 0.0000 |
| rotate_red_block_left | 3 | 0.0000 | 0.0000 | 0.0000 |
| rotate_pink_block_left | 4 | 0.0000 | 0.0000 | 0.0000 |
| lift_red_block_slider | 7 | 0.0000 | 0.0000 | 0.0000 |
| push_pink_block_left | 1 | 0.0000 | 0.0000 | 0.0000 |
| push_into_drawer | 6 | 0.0000 | 0.0000 | 0.0000 |

## 5. 退化模式统计
| signal | baseline mean | coord mean | delta |
|---|---:|---:|---:|
| adjacent_repeat_ratio | 0.7176 | 0.7209 | 0.0033 |
| punctuation_ratio | 0.6497 | 0.6234 | -0.0263 |
| digit_ratio | 0.0000 | 0.0001 | 0.0001 |
| unique_ratio | 0.2079 | 0.2141 | 0.0063 |
| top_token_share | 0.7122 | 0.7098 | -0.0024 |

## 6. 自动标签
| label | count | rate |
|---|---:|---:|
| action_failure | 53 | 13.18% |
| no_change | 141 | 35.07% |
| punctuation_collapse | 147 | 36.57% |
| repetition_collapse | 201 | 50.00% |
| text_only_gain | 47 | 11.69% |

## 7. 典型案例
| sample_id | task_name | delta_joint | labels | preview |
|---|---|---:|---|---|
| place_in_slider_0840783_0840828 | place_in_slider | 0.0079 | repetition_collapse, text_only_gain, action_failure | Instruction: put the object in the cabinet Assistant response: the the the the the the the the the the the the the the the the the the the the the ... |
| turn_on_lightbulb_1646415_1646462 | turn_on_lightbulb | 0.0079 | repetition_collapse, text_only_gain, action_failure | Instruction: move up the switch Assistant response:, the the the, the the the the the the the the the the the the the the the the the the the the t... |
| place_in_slider_1103985_1104033 | place_in_slider | 0.0076 | repetition_collapse, text_only_gain, action_failure | Instruction: put the block in the cabinet Assistant response: the the the the the the the the the the the the the the the the the the the the the t... |
| unstack_block_0501228_0501267 | unstack_block | 0.0071 | repetition_collapse, punctuation_collapse, text_only_gain, action_failure | Instruction: unstack the blocks Assistant response:,, the the the, the the the the the the the,,, the the the the the the the the the the the the t... |
| lift_red_block_drawer_1684982_1685029 | lift_red_block_drawer | 0.0060 | repetition_collapse, text_only_gain, action_failure | Instruction: lift the red block in the drawer Assistant response: the the the the the the the the the the the the the the the the the the the the t... |
| place_in_slider_0923613_0923661 | place_in_slider | 0.0053 | repetition_collapse, text_only_gain, action_failure | Instruction: put the object in the cabinet Assistant response: the the the the the the the the the the the the the the the the the the the the the ... |
| lift_pink_block_drawer_0150333_0150380 | lift_pink_block_drawer | 0.0045 | repetition_collapse, text_only_gain, action_failure | Instruction: lift the pink block in the drawer Assistant response: the the the the the the the the the the the the the the the the the the the the ... |
| turn_off_lightbulb_0035623_0035687 | turn_off_lightbulb | 0.0040 | repetition_collapse, text_only_gain, action_failure | Instruction: toggle the light switch to turn off the light bulb Assistant response: the the the the the the the the the the the the the the the the... |
| place_in_drawer_1705080_1705126 | place_in_drawer | 0.0035 | repetition_collapse, text_only_gain, action_failure | Instruction: place the block in the drawer Assistant response: the the the the the the the the the the the the the the the the the the the the the ... |
| turn_off_led_0064120_0064167 | turn_off_led | 0.0031 | repetition_collapse, text_only_gain, action_failure | Instruction: turn off the led light Assistant response: the the the the the the the the the the the the the the the the the the the the the the the... |

## 8. 下一步建议
- 优先处理 action 表示压缩，因为长度分桶里长序列通常伴随更高的重复和标点塌缩。
- 如果希望 coordination 真正起作用，需要先确认 backbone 能在同一 joint target 上给出更稳定的序列结构。
- 当前最强信号不是 joint 提升，而是 text 端轻微改善但 action 端几乎没动。
