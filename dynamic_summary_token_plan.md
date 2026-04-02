dy# Dynamic Summary Token 方案总结与待确认问题

日期：2026-04-02

---

## 1. 背景与约束

### 1.1 项目背景

- 三阶段 VLA 训练 Pipeline，Stage 3 使用离散块扩散（Block Diffusion）并行解码 text 和 action
- 我负责推理加速（few-step generation）
- 当前在 dLLM 框架 + CALVIN 1k 标注数据上做 proxy 实验验证

### 1.2 Leader 反馈

- 原始 $z_c$ 潜变量方案在 text 端 scaling 时会塌缩，走不通
- global token 建模宏观协调信号的方向可以尝试
- **方案最好是 training-free 的**，方便后续 merge 到主线

### 1.3 已有实验结论（保留不变）

- action 表示压缩（bucketed_int）是第一优先级，已验证有效
- backbone LoRA 适配提升幅度远大于 coordination-only
- coordination 在合理表示下是有效的，不是无效 idea

---

## 2. 新方案：Dynamic Summary Token

### 2.1 核心思想

在离散扩散去噪的每一步，将已确定（unmasked）位置的隐状态聚合成少量 global token，插入到 attention 序列中，让所有待恢复位置都能直接 attend 到全局摘要。

### 2.2 为什么这能帮助 few-step generation

逻辑链条：

1. 标准离散扩散靠**多步迭代**来隐式传播 token 间的依赖（尤其是 text-action 跨模态依赖）
2. Dynamic Summary Token 让模型在**每一步**就能显式看到全局状态 → 单步去噪质量更高
3. 单步更准 → 可以使用更 aggressive 的 unmasking schedule（每步 unmask 更多 token）
4. 每步 unmask 更多 → 总步数更少 → 推理加速

关键点：global token 解决的是"让 few-step 不崩"的前提条件，真正减步数还需要配合 aggressive unmasking schedule。

### 2.3 具体机制

**Step 1：分区域聚合隐状态**

在第 k 步去噪完成后，分别对 text 区域和 action 区域的已确定 token 做 pooling：

$$s_{\text{text}}^k = \text{Pool}(\{h_i^k \mid i \in U^k \cap \text{TextRegion}\})$$

$$s_{\text{action}}^k = \text{Pool}(\{h_i^k \mid i \in U^k \cap \text{ActionRegion}\})$$

**Step 2：构造 global token**

最简单版本：

$$g^k = \frac{s_{\text{text}}^k + s_{\text{action}}^k}{2}$$

或保留两个独立 token $[g_{\text{text}}^k, g_{\text{action}}^k]$。

**Step 3：注入 attention**

在第 k+1 步 forward pass 时，将 $g^k$ 作为额外 token 拼接到序列中参与 self-attention。

**Step 4：每步更新**

随着去噪推进，$U^k$ 不断扩大，$g^k$ 的内容动态更新，越来越精确。

### 2.4 为什么是 training-free

- $g^k$ 从 backbone 已有隐状态直接计算，不引入新参数
- Transformer self-attention 天然支持变长序列
- 不会发生 posterior collapse（因为没有需要优化的 posterior）

### 2.5 与原始 $z_c$ 方案的区别

| | $z_c$ 潜变量 | Dynamic Summary Token |
|---|---|---|
| 参数 | 需要训练 coordination module | 无新参数 |
| 构造方式 | 编码器压缩 | 直接 pooling 隐状态 |
| 塌缩风险 | scaling 时会塌缩 | 无（没有 posterior） |
| 表达能力 | 更强（学习过的压缩） | 较弱（简单聚合） |
| merge 成本 | 需要训练 + 部署额外模块 | 仅修改采样逻辑 |

---

## 3. 待确认的技术问题（需要看代码后确认）

### 3.1 Forward Pass 结构

**问题**：dLLM 在去噪的每一步是怎么调用 backbone forward 的？

- 每步做完整 forward pass？
- 还是有 KV cache 之类的优化？

**为什么重要**：如果每步都是 full forward，插入 global token 很自然；如果有 cache，需要处理 cache 失效。

### 3.2 Attention Mask 管理

**问题**：去噪过程中 attention mask 是怎么设置的？

- masked 位置能 attend to unmasked 位置吗？
- unmasked 位置能 attend to masked 位置吗？
- 你之前改的 bidirectional mask 具体是怎么实现的？

**为什么重要**：直接决定 global token 插入后的信息流方向。

### 3.3 隐状态获取与计算开销

**问题**：如何获取隐状态来构造 global token，同时避免额外 forward pass？

**可能的方案**：

- **方案 A**：用上一步去噪的隐状态构造当前步的 global token（零额外开销，但信息滞后一步）
- **方案 B**：当前步先做一次 forward 拿隐状态，再插入 global token 重新 forward（准确但计算量翻倍）
- **方案 A 优先**，因为信息滞后一步在多步去噪中通常可以接受

**需要确认**：backbone forward 过程中，中间层隐状态是否容易获取？还是只能拿到最后一层输出？

### 3.4 区域边界识别

**问题**：在去噪过程中，如何确定哪些位置属于 text region、哪些属于 action region？

- marker token（如 "Action sequence:"）在去噪时是 masked 还是始终可见？
- 如果 marker 也被 mask 了，需要额外机制来标记区域边界

### 3.5 Pooling 策略

**问题**：Mean pooling 是最简单的，但可能不是最优的。

**可选方案**：

- Mean pooling（最简单）
- Confidence-weighted pooling：用 token 预测置信度作为权重（training-free，信息量更大）
- Max pooling（保留最显著特征）
- Attention pooling：用一个固定 query 对隐状态做 attention（接近 training-free，但 query 向量需要选定）

**建议**：先用 mean pooling 做 baseline，确认机制有效后再尝试 confidence-weighted。

### 3.6 Position Encoding 处理

**问题**：插入的 global token 不对应原始序列的真实位置，给它什么 position id？

**可选方案**：

- 固定特殊 position（如 0）
- 不给位置编码，纯靠内容信息
- 复用 instruction 区域的某个 position

**需要确认**：dLLM 用的是什么类型的位置编码？RoPE？Absolute？

---

## 4. 实验计划

### 4.1 第一阶段：验证 Dynamic Summary Token 有效性

在 bucketed_int baseline（不加 LoRA）上：

1. 实现 Dynamic Summary Token 机制
2. 在默认步数下对比有无 global token 的 joint/text/action token accuracy
3. 如果有效，进入第二阶段

### 4.2 第二阶段：验证 few-step 加速

1. 设计步数梯度实验：full steps → 1/2 steps → 1/4 steps
2. 对比有无 global token 在不同步数下的精度衰减曲线
3. 核心指标：在多少步数下，有 global token 的精度 ≥ 无 global token 的 full-step 精度

### 4.3 第三阶段：叠加 backbone LoRA

1. 在 backbone_lora checkpoint 上测试 dynamic summary token
2. 验证两者是否有叠加收益
3. 在 LoRA 基础上重复 few-step 实验

### 4.4 评测 Protocol（待确认）

- 当前默认去噪步数是多少？
- 精度掉多少以内算可接受？（建议定一个 threshold，比如 joint token acc 下降不超过 5%）
- 是否需要 task-level 的分析？

---

## 5. 给 Claude Code 的上下文提示

在 Claude Code 中继续工作时，建议先让它了解：

1. 项目的核心目标是在 dLLM 的离散扩散采样过程中，插入 training-free 的 global token 来协调 text 和 action 的并行解码
2. 需要先阅读现有的 sampler 代码（特别是去噪循环的实现），回答上面 3.1-3.6 的问题
3. 然后基于回答设计具体的实现方案
4. 代码修改应尽量集中在 sampler 侧，不改 backbone 结构
