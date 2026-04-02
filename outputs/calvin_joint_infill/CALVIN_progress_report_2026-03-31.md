# CALVIN 实验进展汇报

日期：2026-03-31

## 1. 这阶段在做什么

这阶段我主要在 CALVIN 数据上搭一个 joint infill 实验，用来观察模型在同时恢复文本和动作时的表现，重点看两件事：

- action 表示怎么设计更合适
- backbone 适配和 coordination module，哪一个更关键

从当前结果看，这条线对 few-step generation 是有参考价值的，但目前更像是在做代理实验。也就是说，我现在先用一个更容易控制、方便比较的 infill 任务，去判断哪些因素会真正影响后面的 few-step 生成。

这次实验使用的数据路径是：
`/data/ytw/VLA_baseline/calvin/outputs/task_ABC_D_batches/training.jsonl`

总样本数是 `1000`，固定切成 `800/200` 的 train/test。

---

## 2. 目前采用的任务设定

我现在保留的设定是：

- 输入里包含 instruction
- 目标里同时包含一段文本回复和 action sequence
- 训练和评测时把目标区域 mask 掉，再让模型恢复

动作表示目前统一用 `bucketed_int`，8 个桶。

之所以直接用这个版本，是因为之前试过原始浮点动作表示，序列太长，结果很差，而且对当前判断帮助已经不大了。现在继续保留那部分细节意义不大，所以后面主要汇报压缩表示之后的结果。

在 `bucketed_int` 下，平均 target 长度大约是 `1194` token；相比之前的原始浮点表示，长度已经明显下降，任务也更稳定。

---

## 3. 已完成的主要实验

### 3.1 `bucketed_int` baseline

这是当前的基础参考线，不加 coordination，也不做 backbone 适配。

结果是：

- `text_region_token_acc = 0.0690`
- `action_region_token_acc = 0.0680`
- `joint_region_token_acc = 0.0656`
- `latency_sec = 0.8263`

这组结果说明两点：

- 动作表示压缩之后，任务已经比之前容易很多
- 但单靠原始 backbone，整体恢复质量还是比较有限，尤其 action 侧还有很大提升空间

### 3.2 `bucketed_int + coordination-only`

在这个实验里，我冻结 backbone，只训练 coordination module。

结果是：

- baseline `joint_region_token_acc = 0.0656`
- coordinated `joint_region_token_acc = 0.0931`
- `text_region_token_acc = 0.0690 -> 0.0734`
- `action_region_token_acc = 0.0680 -> 0.1031`
- `latency_sec = 0.8243 -> 0.1119`

我的理解是：

- coordination 在当前设定下是有效的
- 提升不只是 text 侧，action 侧也有明显改善
- 但这个提升量级还不算决定性，更像是在 baseline 之上做增强

### 3.3 `bucketed_int + backbone_lora`

这个实验是让我目前判断变化最大的一组。

我先做过一个小规模版本，效果就已经很明显；后来又补了和前面完全对齐的正式版：

- train：`800`
- test：`200`
- epoch：`1`
- action 表示：`bucketed_int`

正式版结果如下：

- baseline `text_region_token_acc = 0.0690`
- trained `text_region_token_acc = 0.0618`
- baseline `action_region_token_acc = 0.0680`
- trained `action_region_token_acc = 0.7113`
- baseline `joint_region_token_acc = 0.0656`
- trained `joint_region_token_acc = 0.5421`
- baseline `latency_sec = 0.8325`
- trained `latency_sec = 1.7966`

这组结果说明：

- backbone 适配带来的提升远大于 coordination-only
- 最大的变化出现在 action 恢复上
- text 指标有一点回落，说明现在这个 LoRA 版本更偏向把 action 学好，text/action 平衡还可以继续调
- latency 变慢了，但从收益幅度看，目前这个代价是可以接受的

---

## 4. 目前能下的判断

到现在为止，我觉得比较稳的判断有下面几个。

### 4.1 当前最关键的变量是 action 表示

如果动作表示太长、太密，模型很容易直接崩掉，后面的 coordination 或训练技巧都很难发挥作用。

把 action 改成 `bucketed_int` 之后，baseline 就明显改善了。这说明在 CALVIN 这类任务上，先把 action 表示做轻，是第一步。

### 4.2 coordination 有用，但不是当前最大的收益来源

在 `bucketed_int` 设定下，coordination 确实能把指标往上推，尤其对 action 侧有帮助。

但和 backbone_lora 相比，它的提升还是小很多。所以如果要排优先级，我现在不会把 coordination 放在第一位。

### 4.3 目前更大的瓶颈在 backbone 对任务分布的适配

这点是当前最明确的结论。

在同样的 `bucketed_int`、同样的 `800/200` 设定下：

- `coordination-only` 的 `joint_region_token_acc` 是 `0.0931`
- `backbone_lora` 的 `joint_region_token_acc` 是 `0.5421`

所以现在更像是：

- 先把动作表示处理好
- 再让 backbone 学会这个任务
- coordination 更适合作为后续增强项

### 4.4 这条线和 few-step generation 的关系

严格说，现在做的不是最终的 few-step generation，而是一个代理实验。

它的价值在于帮助回答：

- 哪种 action 表示更适合少步数生成
- 少量适配参数到底该优先加在 coordination，还是加在 backbone
- 文本和动作一起建模时，真正决定性能的部分在哪

所以这条线没有完全偏，但后面需要更主动地往 few-step generation 目标上收。也就是说，后续不能只看 joint infill 指标本身，还要开始看不同 few-step budget 下的效果变化。

---

## 5. 下一步准备做什么

下一步我准备优先做 `backbone_lora_plus_coord`。

原因是：

- `bucketed_int` 已经证明是对的
- `coordination-only` 已经证明不是没用
- `backbone_lora` 已经证明是当前主收益来源

所以现在最值得问的问题是：

在 backbone 已经适配好的前提下，coordination 还能不能继续提供额外收益。

如果 `backbone_lora_plus_coord` 比 `backbone_lora` 还好，那说明 coordination 适合作为第二层增强。

如果没有继续提升，甚至回退，那说明现阶段应该把重点放在 backbone 训练策略和 few-step 设定本身，而不是继续堆 coordination。

除了这个实验之外，后面我还想补两类分析：

- 不同 `few_step_budget` 下的性能和速度对比
- text/action 权重和平衡问题，看看能不能在不牺牲 text 的情况下继续保住 action 提升

---

## 6. 这次汇报想传达的核心信息

如果只讲最核心的内容，我会这样概括：

1. 我已经把 CALVIN 上的 joint infill 实验搭起来了，并且当前设定已经基本稳定。
2. 当前最关键的前提是把 action 表示压缩到可学的形式，`bucketed_int` 是目前有效的版本。
3. 在这个基础上，coordination 是有效的，但更大的收益来自 backbone 适配。
4. 下一步最值得做的是 `backbone_lora_plus_coord`，并逐步把分析重新对齐回 few-step generation 这个主目标。
