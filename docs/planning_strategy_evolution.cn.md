# 规划策略演进

灵感来源：arXiv:2603.04735v1（基于树搜索 + 自动验证的神经符号 AI 发现系统）。
适配本项目的通用深度研究场景与科学技能体系。

---

## 1. 当前架构概览

### 1.1 规划管线

```
User Query
  → plan_maker (generates natural-language plan)
  → plan_reviewer (critiques plan)
  → plan_review_confirmation (decides exit/retry)
  → [loop up to 10 iterations]
  → plan_candidate_selector (ranks candidates, keeps baseline unless margin > 0.12)
  → plan_parser (structures plan → stages[] + success_criteria[])
  → stage_orchestrator (executes stages one by one)
      per stage:
        → implementation_loop (coding → review → review_confirmation, loop escalates on exit=true)
        → criteria_checker (LLM inspects files, updates met/evidence)
        → stage_reflector (may modify/add stages)
  → summary_agent
```

### 1.2 关键数值限制

| 参数 | 值 | 位置 |
|------|-----|------|
| 规划循环最大迭代次数 | 10 | `agent.py:896` |
| 计划候选方案上限 | 8 | `plan_selector.py` |
| 最小切换阈值（保留基线方案） | 0.12 | `ADS_PLAN_RANK_MIN_SWITCH_MARGIN` |
| 编排器最大迭代次数 | 50 | `stage_orchestrator.py:294` |
| 阶段重试（未通过审核） | 在 50 次迭代内无限制 | 编排器继续循环 |
| 典型阶段数量 | 3–7 | `plan_parser.md:19` 指导性建议，未强制执行 |
| 事件压缩阈值 | 40 个事件 | 多处 |
| 编码事件硬上限 | 100 个事件 | `implementation_loop.py:38` |

### 1.3 阶段记录结构

```python
{
    "index": int,
    "stage_id": "s1",
    "title": str,
    "description": str,
    "completed": bool,
    "status": "pending|in_progress|approved|retrying|failed",
    "implementation_result": str,
    "depends_on": ["s0"],          # DAG dependencies
    "inputs_required": [...],       # expected input artifacts
    "outputs_produced": [...],      # output artifacts
    "evidence_refs": [...],         # evidence references
    "subtasks": [...],              # sub-decomposition (field exists, unused)
}
```

### 1.4 成功标准记录结构

```python
{
    "index": int,
    "criteria": str,       # human-readable criterion
    "met": bool,           # updated by criteria_checker
    "evidence": str | None # file paths, metrics cited by LLM
}
```

### 1.5 当前验证方式

criteria_checker 是一个**纯 LLM agent**，配备文件检查工具（read_file、list_directory 等）。
它读取工作目录中的文件，主观判断每个标准是否达成。
不存在程序化/数值化验证机制。

### 1.6 当前自适应机制

- **stage_reflector**：可以修改未完成阶段的描述或在每个阶段完成后追加新阶段。Prompt 指示"保持保守"、"仅在确实必要时添加"。
- **plan_candidate_selector**：在规划循环中收集计划，通过 Jaccard 覆盖度 + 历史记录 + 阶段数量进行排名。保守策略：除非挑战者超过阈值，否则保留基线方案。
- **无子阶段分解**：`subtasks` 字段存在于 schema 中，但从未被填充或使用。
- **无回溯机制**：失败/重试的阶段按线性方式重新尝试，没有回退到替代方案的能力。

---

## 2. 已识别的改进策略

来源：arXiv:2603.04735v1 方法论，适配通用深度研究场景。

### 策略 A：程序化验证（P0）

**核心洞察**："所有能通过程序化方式验证的，都应该用程序化方式验证。"

论文的自动数值反馈捕获了 >80% 的错误。当前 criteria_checker 完全依赖 LLM 判断——主观、易产生幻觉、且无法执行代码。

**当前差距**：
- criteria_checker 使用文件工具（read_file、list_directory）但无法执行代码
- 验证过程是主观的：LLM 读取文件内容并"决定"标准是否达成
- 没有数值比较、没有 schema 校验、没有超出 LLM 自主选择范围的存在性检查

**建议方案**：
构建一个在 LLM criteria_checker **之前**运行的验证层，提供事实性信号：

```
Tier 1（程序化验证，无需 LLM）：
  - 文件存在性检查：预期输出文件是否存在？
  - Schema 校验：CSV 是否有预期的列？JSON 是否能正确解析？
  - 数值阈值检查：AUC > 0.85？p-value < 0.05？（从结果文件中提取）
  - 图片验证：图表文件大小非零、格式正确？
  - 代码执行状态：脚本是否无错误运行？

Tier 2（LLM 辅助验证，携带 Tier 1 信号）：
  - criteria_checker 接收 Tier 1 结果作为附加上下文
  - LLM 专注于真正需要推理的定性判断
  - "程序化检查已通过：[file_exists=true, auc=0.91]。现在评估分析质量。"
```

**集成点**：在 stage_orchestrator 中 implementation_loop 和 criteria_checker 之间新增回调或 agent（第 535–571 行）。或将 Tier 1 结果注入 criteria_checker 的 prompt 上下文。

**风险**：低。纯增量式改动，不替换现有 criteria_checker。

---

### 策略 B：Negative Prompting 实现计划多样性（P0）

**核心洞察**："生成真正不同的方案，而非同一想法的迭代优化。"

论文使用 negative prompting 强制发现了 6 种不同的解法。当前规划循环产出的是同一方案的迭代优化——reviewer 提反馈，plan_maker 调整。

**当前差距**：
- 规划循环（最多 10 次迭代）在 reviewer 通过后立即退出
- 所有迭代都在优化同一个计划，而非生成替代方案
- plan_candidate_selector 收集的变体只是渐进式修订，不是多样化策略

**建议方案**：
在 reviewer 通过方案 A 后，不立即退出，而是：

```
Round 1: plan_maker → reviewer 通过 → 候选方案 A ✓（存储）
Round 2: plan_maker + negative prompt：
         "一个有效计划已获批准（摘要：{Plan_A_summary}）。
          生成一个根本不同的分析策略。
          不要使用 {key_methods_from_A}。
          探索替代方法论、不同的分析框架、
          或对问题的不同分解方式。"
         → reviewer 评估 → 候选方案 B ✓（存储）
Round 3:（可选，如预算允许）→ 候选方案 C ✓（存储）
最终：   plan_candidate_selector 对 A、B、C 排名 → 选最优
```

**为什么 negative prompting 优于"在一个 prompt 中列出 3 种策略"**：
- 分轮生成 + 明确排除能产生更大分歧的想法
- 单 prompt 多策略倾向于产生表面变体（核心相同，标签不同）
- 每个候选方案都独立接受完整的 reviewer 审查

**集成点**：修改 agent.py 中规划循环的退出逻辑（第 876–897 行）。在首次 `exit=true` 后，存储候选方案并注入 negative prompt 进入下一迭代，而非退出。

**风险**：中等。规划成本增加 2-3 倍（但规划阶段仅占总成本的 ~10%）。需要精心设计 prompt 以确保负面约束不会降低质量。

---

### 策略 C：自适应深度——动态阶段分解（P1）

**核心洞察**："研究深度应该是不均匀的。某些步骤值得更深入的探究；深度和广度应该从研究发现中涌现，而非预先固定。"

论文的树搜索自然地将更多探索分配给有前景的分支。当前系统具有固定深度的阶段：每个阶段获得一次 implementation_loop 执行，没有子分解。

**当前差距**：
- `subtasks` 字段存在于阶段记录中但从未使用
- 所有阶段被平等对待：每个阶段一次 implementation_loop 执行
- 没有机制表示"这个阶段出乎意料地复杂，需要进一步分解"
- stage_reflector 可以修改描述和添加阶段，但不能将一个阶段分解为子阶段
- 没有回溯：如果阶段 3 揭示阶段 2 的方法有误，没有机制重新审视

**建议方案（分阶段实施）**：

Phase 1 ——**基于复杂度信号的阶段分解**：
```
implementation_loop 尝试某个阶段后：
  如果 review 表示"过于复杂"或"部分完成"或 attempt > 1：
    → 调用 "stage_decomposer" agent
    → 将当前阶段拆分为 2-3 个子阶段
    → 将子阶段插入阶段列表（替换原始阶段）
    → 以更细粒度继续编排
```

Phase 2 ——**基于发现的回溯**：
```
stage_reflector 识别出根本性问题后：
  如果 reflector 表示"阶段 N 的方法被研究发现否定"：
    → 标记阶段 N 需要用新方法重新实现
    → 注入 negative prompt："之前的方法是 {old_approach}。失败原因是 {reason}。使用不同的方法。"
    → 这将策略 B（negative prompting）应用于阶段级别
```

Phase 3 ——**类树形探索**（未来，如果 Phase 1+2 不够）：
```
对于高风险阶段：
  → 并行生成 2 种实现方案
  → 对两者运行轻量级验证
  → 继续使用中间结果更好的那个
```

**集成点**：
- Phase 1：在 stage_orchestrator 的重试检测之后添加分解逻辑（第 477–514 行）
- Phase 2：扩展 stage_reflector 的 prompt 和回调以支持"重新尝试"指令
- Phase 3：需要并行执行能力（当前不支持）

**风险**：
- Phase 1：低到中等。范围明确。
- Phase 2：中等。需要谨慎的状态管理以避免无限循环。
- Phase 3：高。架构级变更。

---

### 策略 D：分层验证与模型升级（P1）

**核心洞察**："探索使用快速/低成本模型，验证和精炼使用最强模型。"

论文使用标准 Gemini 进行树搜索，然后使用更高级的模型进行深度验证和简化。

**当前差距**：
- 模型路由是基于角色的（plan_maker、reviewer、coding 等），而非基于阶段
- 相同类型的所有阶段使用相同模型，不考虑复杂度
- 没有"升级"机制：如果一个阶段失败，它用同一个模型重试

**建议方案**：

```
阶段感知的模型选择：
  plan_parser 为阶段标注 complexity_hint: "standard" | "complex" | "critical"

  执行时：
    "standard" 阶段 → 默认编码模型
    "complex" 阶段 → 更强模型（如 gpt-5.2）
    "critical" 阶段（最终整合、报告）→ 可用的最强模型

  重试时（attempt > 1）：
    自动升级到下一级模型
    "如果默认模型一次没做成，换更强的模型试试"
```

```
深度验证环节：
  所有阶段完成后、summary_agent 之前：
    → deep_verifier agent（使用最强模型）
    → 检查：结果是否真正满足所有成功标准？
    → 检查：各阶段之间是否存在逻辑不一致？
    → 如果发现问题：注入修正阶段并重新进入编排
```

**集成点**：
- 阶段复杂度：扩展 plan_parser 输出 schema，增加 `complexity_hint`
- 模型升级：在 stage_orchestrator 中添加重试感知的模型选择
- 深度验证器：在 agent.py workflow 中 stage_orchestrator 和 summary_agent 之间新增 agent

**风险**：低。增量式改动。Token 成本增加有限。

---

### 策略 E：增强计划评分（P0）

**核心洞察**："多维度的自动化评分能实现更好的剪枝。"

论文的 PUCT 评分在 600 个节点间平衡了探索和利用。当前 plan_selector 的评分公式维度有限。

**当前评分公式**（plan_learning.py）：
```
score = 0.45 * jaccard_coverage
      + stage_count_bonus (0.05 for 3-7 stages)
      + historical_pattern_score
      - 0.03 * cold_start_penalty
      - 0.12 * retry_rate
      + 0.08 * baseline_bonus
```

**建议新增的评分维度**：

| 维度 | 信号 | 成本 |
|------|------|------|
| **依赖图有效性** | depends_on 是否构成有效 DAG？无环？所有引用存在？ | 零（纯计算） |
| **数据流覆盖度** | outputs_produced 是否覆盖下游 inputs_required？ | 零（纯计算） |
| **技能覆盖率** | 所需能力在 skill_registry 中的命中率 | 零（查询） |
| **粒度均匀性** | 阶段描述长度的方差。非常不均匀 → 扣分 | 零（字符串长度） |
| **方法多样性** | 如果策略 B 激活：此计划与其他计划的差异度 | 低（方法术语的 Jaccard 距离） |

**集成点**：扩展 plan_learning.py 中的 `rank_plan_candidates()`。所有新维度无需 LLM 调用即可计算。

**风险**：极小。纯评分扩展。

---

## 3. 实施优先级

| 优先级 | 策略 | 工作量 | 风险 | 影响 |
|--------|------|--------|------|------|
| **P0** | A：程序化验证 | 小到中 | 低 | 高——消除验证中的幻觉 |
| **P0** | B：Negative Prompting | 中 | 中 | 高——产生根本不同的分析方案 |
| **P0** | E：增强计划评分 | 小 | 极小 | 中——无 LLM 成本的更好计划选择 |
| **P1** | C Phase 1：阶段分解 | 中 | 低到中 | 高——复杂阶段的自适应深度 |
| **P1** | D：模型升级 | 小到中 | 低 | 中——更好的资源分配 |
| **P2** | C Phase 2：回溯 | 中到大 | 中 | 高——但实现起来复杂 |
| **P3** | C Phase 3：树形探索 | 大 | 高 | 不确定——对数据科学任务可能不值得 |

### 建议实施顺序

```
Sprint 1: E（评分增强）+ A（验证 Tier 1）
  → 速赢，无 LLM 成本增加，立即提升质量

Sprint 2: B（negative prompting）+ A（验证 Tier 2）
  → 计划多样性 + 携带程序化上下文的 LLM 辅助验证

Sprint 3: D（模型升级）+ C Phase 1（阶段分解）
  → 自适应深度和模型路由

Sprint 4+: C Phase 2（回溯）、C Phase 3（树形探索）
  → 仅在 Sprint 1-3 效果不足时实施
```

---

## 4. 核心设计原则

源自论文方法论，适配研究任务：

1. **尽早验证、频繁验证**：不要等到最后才检查。在每个阶段边界进行程序化检查。

2. **多样性优于精炼**：多个真正不同的方案胜过打磨一个方案。使用 negative prompting 强制产生分歧。

3. **自适应深度**：不是所有阶段都值得同等深度。分解复杂阶段；不要过度规划简单阶段。让执行过程中的发现驱动深度决策。

4. **模型匹配任务**：探索可以使用快速/低成本模型。验证和综合需要最强可用模型。

5. **程序化检查优于 LLM 判断**：如果可以通过运行代码或检查文件来验证，就不要问 LLM。将 LLM 判断力保留给定性评估。

6. **回溯不是失败**：发现某种方法行不通是有价值的信息。系统应该能够尝试不同的方法（阶段级别的 negative prompting），而不是反复重试同一个失败策略。

---

## 5. 待讨论的开放问题

- [ ] 策略 B：生成多少个替代计划？2 个（最小多样性）还是 3 个（更多覆盖，更高成本）？
- [ ] 策略 A：程序化验证应该是强制性的（失败时阻断）还是建议性的（将信号传递给 LLM）？
- [ ] 策略 C：什么触发阶段分解？LLM 审核说"过于复杂"？重试次数 > N？阶段描述长度？
- [ ] 策略 D：如何标注阶段复杂度？解析时由 LLM 判断？启发式方法（描述长度、关键词检测）？
- [ ] 成本预算：这些改进的最大可接受成本倍率是多少？（当前：1x → 全部策略：规划阶段 ~3-4x，执行阶段 ~1.2x）
- [ ] 策略 C Phase 2：如何防止无限回溯循环？最大回溯次数？Negative prompt 必须排除所有之前尝试过的方法？

---

## 附录：源论文摘要

**论文**："Solving an Open Problem in Theoretical Physics using AI-Assisted Discovery"（arXiv:2603.04735v1）

**核心系统**：Gemini Deep Think + 树搜索（PUCT）+ 自动数值反馈

**关键成果**：
- 在搜索树中探索了约 600 个候选节点
- 自动验证器剪枝了 >80% 的分支（代数错误、数值发散）
- Negative prompting 强制发现了 6 种根本不同的解法
- 最优雅的解法（Gegenbauer 方法）是第 6 个被发现的——单路径搜索将永远错过它
- 分层验证：自动搜索 → 人工引导的深度验证（使用更强模型）

**可迁移原则**（不限于数学领域）：
1. 每步验证 + 自动反馈
2. 多路径探索 + 多样性强制
3. 评分与剪枝，将资源集中在有前景的路径
4. 分层精炼：快速探索 → 深度验证
5. Negative prompting 跳出解空间中的局部最优
