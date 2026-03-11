# Phase 2 实施计划：Innovation OS V1.1 智能层

## 目标
为 agentic-data-scientist 添加智能层：自动模式路由 + 运行期创新触发 + 方法评估/回退 + 深度验证

## 架构分析与分阶段策略

### 原 spec 的 9 个模块 → 重新分组为 3 个子阶段

**分析结论**：
- TRIZ operator 和 abduction operator 需要非常特定的触发条件（矛盾检测、异常解释），与 Phase 2 核心目标（自动路由 + 方法评估）正交
- 建议：TRIZ/abduction 推迟到 Phase 3（与 morphology_operator 一起，都是"高级发现算子"）
- deep_verifier 是独立的后处理步骤，风险低，可以最后做
- method_critic + backtracker 修改 stage_orchestrator 内部循环，风险最高，需要最谨慎

### 子阶段分解

**P2-A: 自动路由层**（problem_framer + task_router）
- 修改点：agent.py 根工作流、CLI
- 风险：低（新增前置 agent，不修改现有逻辑）
- CLI --innovation-mode 保留为 override

**P2-B: 运行期智能**（method_critic + method_backtracker + innovation_trigger_detector）
- 修改点：stage_orchestrator.py 内部循环
- 风险：最高（修改核心执行路径）
- 关键约束：routine 模式零回归

**P2-C: 深度验证 + summary 增强**（deep_verifier + innovation_summary_contract）
- 修改点：agent.py 根工作流（插入 summary 前）、summary prompt
- 风险：低（新增后处理步骤）

## 各模块详细设计

### P2-A1: problem_framer

**位置**: `agents/adk/problem_framer.py`
**Prompt**: `prompts/base/problem_framer.md`
**State 输出**: `StateKeys.FRAMED_PROBLEM`

输出 schema:
```json
{
  "research_goal": "一句话目标",
  "task_type": "data_analysis|modeling|exploration|discovery|routine_processing",
  "knowns": ["已知信息"],
  "unknowns": ["未知/待发现"],
  "contradictions": ["如有矛盾/约束冲突"],
  "complexity_signals": ["信号: 多组学/新方法/对比/优化/..."],
  "recommended_mode": "routine|hybrid|innovation"
}
```

**设计要点**：
- 使用较便宜的模型（plan_maker_model）
- 必须快速（< 15s），否则 routine 任务体验退化
- config temperature=0.2（确定性输出）

### P2-A2: task_router

**位置**: `agents/adk/task_router.py`
**不需要 prompt**（纯逻辑 agent）
**State 输出**: `StateKeys.INNOVATION_MODE`（覆写 build_initial_state_delta 设置的默认值）

**路由规则（优先级从高到低）**：
1. CLI override: `--innovation-mode X` → 强制使用 X（检查 env var `ADS_INNOVATION_MODE_OVERRIDE`）
2. 如果 framed_problem 不存在 → routine
3. 规则引擎：
   - task_type == "routine_processing" → routine
   - task_type == "discovery" || contradictions 非空 → innovation
   - complexity_signals 数量 >= 2 → hybrid
   - recommended_mode 作为 tiebreaker
4. LLM 兜底（仅当规则引擎无法决定时）

**CLI 变更**：
- `--innovation-mode` 默认值从 "routine" 改为 "auto"
- "auto" → 使用 task_router 自动判断
- "routine|hybrid|innovation" → 作为 override

### P2-B1: method_critic

**位置**: `agents/adk/method_critic.py`
**Prompt**: `prompts/base/method_critic.md`
**State 输出**: `StateKeys.METHOD_CRITIC_OUTPUT`

**触发条件**：
- innovation_mode != "routine"
- 且 stage review 不通过（approved == False）
- 且 attempts >= 2（第一次失败先正常重试）

输出 schema:
```json
{
  "issue_type": "execution_failure|method_failure",
  "confidence": 0.85,
  "evidence": ["具体证据"],
  "recommendation": "retry|backtrack|continue",
  "explanation": "判断理由"
}
```

**集成点**：stage_orchestrator.py line ~478（review 不通过分支内）

### P2-B2: method_backtracker

**位置**: `agents/adk/method_backtracker.py`
**不需要 prompt**（纯逻辑 agent，读 state 决策）
**State 输出**: 修改 `StateKeys.SELECTED_METHOD`, `StateKeys.STANDBY_METHODS`

**触发条件**：
- method_critic 输出 issue_type == "method_failure" && recommendation == "backtrack"
- 且 budget.remaining("backtrack") > 0
- 且 standby_methods 非空

**执行逻辑**：
1. 消耗 budget.backtrack
2. 当前方法标记 status="failed", rejection_reason=critic 的 explanation
3. 从 standby 中取 top-1 作为新 selected_method
4. 记录到 StateKeys.BACKTRACK_HISTORY
5. 触发重新规划：清空当前计划，重新走 plan_instantiator → plan_parser → stage_orchestrator

**限制**：每 stage 最多 1 次，每任务最多 budget.backtrack 次

### P2-B3: innovation_trigger_detector

**位置**: `agents/adk/innovation_trigger.py`
**不需要 prompt**（纯程序化检测）
**State 输出**: `StateKeys.INNOVATION_TRIGGER`

**检测维度**（全部程序化，无 LLM 调用）：
- `mediocre_review`: review 通过但带保留意见（关键词检测）
- `excessive_retries`: attempts > 2
- `criteria_stagnation`: criteria met 数量连续 2 个 stage 没有增加
- `verifier_warnings`: programmatic_verifier verdict == "warn"

**触发输出**：
```json
{
  "triggered": true|false,
  "signals": ["mediocre_review", "excessive_retries"],
  "strength": 0.7,
  "recommended_action": "none|escalate_review|consider_method_switch"
}
```

**设计要点**：
- 纯程序化，零 LLM 成本
- 只在 non-routine 模式下运行
- 集成点：stage_orchestrator 在 criteria_checker 之后、stage_reflector 之前

### P2-C1: deep_verifier

**位置**: `agents/adk/deep_verifier.py`
**Prompt**: `prompts/base/deep_verifier.md`
**State 输出**: `StateKeys.DEEP_VERIFICATION`

**触发条件**：
- innovation_mode != "routine"
- 且 budget.remaining("verification") > 0

**执行逻辑**：
- 使用 review_model（已配置为较强模型）
- 输入：所有 stage 结果 + criteria 状态 + method card + plan
- 输出：全局一致性评估 + 未覆盖风险

**集成点**：agent.py 根工作流，在 stage_orchestrator 之后、summary_agent 之前

### P2-C2: innovation_summary_contract

**修改文件**: `prompts/base/summary.md`
**新增条件块**：当 innovation_mode != "routine" 时，要求 summary 包含：
- 方法选择过程说明
- 被淘汰方法及原因
- 如有回退：回退过程说明
- 深度验证结果

## State Keys 扩展

```python
# Phase 2 新增
StateKeys.FRAMED_PROBLEM = "framed_problem"
StateKeys.INNOVATION_TRIGGER = "innovation_trigger"
StateKeys.METHOD_CRITIC_OUTPUT = "method_critic_output"
StateKeys.BACKTRACK_HISTORY = "backtrack_history"
StateKeys.DEEP_VERIFICATION = "deep_verification"
```

## 测试策略

### 单元测试（每个模块）

| 模块 | 测试文件 | 预估 tests |
|------|----------|-----------|
| problem_framer | test_problem_framer.py | ~15 |
| task_router | test_task_router.py | ~20 |
| method_critic | test_method_critic.py | ~12 |
| method_backtracker | test_method_backtracker.py | ~15 |
| innovation_trigger | test_innovation_trigger.py | ~12 |
| deep_verifier | test_deep_verifier.py | ~10 |

### 端到端规划测试

| 测试 | 模式 | 验证点 |
|------|------|--------|
| P2-T1 | auto (简单查询) | task_router → routine |
| P2-T2 | auto (发现性查询) | task_router → innovation |
| P2-T3 | override routine | CLI override 生效，忽略 router |
| P2-T4 | hybrid + 复杂任务 | 完整 innovation 流程 |
| P2-T5 | innovation + deep_verifier | 验证后处理 |

## 实施顺序

1. [x] P2-A1: problem_framer (core + prompt + unit tests) — 19 tests passed
2. [x] P2-A2: task_router (core + CLI 变更 + unit tests) — 40 tests passed (含 data_analysis cap 规则 2 tests)
3. [x] P2-A: agent.py 集成 problem_framer + task_router — _ModeGatedPlanningAgent + 根工作流重构, 4 tests
4. [x] P2-A: 运行 P2-T1~T3 验证自动路由 — 全部通过（修复 agent parent 冲突 bug）
5. [x] P2-B3: innovation_trigger_detector (纯程序化，风险最低) — 44 tests passed
6. [x] P2-B1: method_critic (core + prompt + unit tests) — 34 tests passed
7. [x] P2-B2: method_backtracker (core + unit tests) — 17 tests passed
8. [x] P2-B: stage_orchestrator 集成 trigger + critic + backtracker — 完成
9. [x] P2-C1: deep_verifier (core + prompt + unit tests) — 20 tests passed
10. [x] P2-C2: innovation_summary_contract (prompt 修改 + state 扩展) — 完成
11. [x] P2-C: agent.py 集成 deep_verifier — 完成
12. [x] 全量测试 + 文档更新 — 472 tests passed, 文档已更新, E2E P2-T1~T3 通过
13. [x] Prompt 精炼热修复 — problem_framer.md 细化 + task_router data_analysis cap 规则 → 474 tests passed, 回归 4/4 通过

## 推迟到 Phase 3 的模块

- TRIZ operator → 需要矛盾检测基础设施（problem_framer.contradictions 先就位）
- abduction operator → 需要异常检测基础设施（innovation_trigger 先就位）
- 这两个推迟不影响 Phase 2 核心目标

## Status
**Phase 2 全部完成。474 unit tests passed + 3 E2E plan-only tests passed + 4 回归测试通过。文档已更新。**

### Prompt 精炼热修复（2026-03-10）

**问题**：multi-omics 标准分析被错误分类为 `discovery`，导致过度路由到 innovation 模式。

**修复**：
1. `problem_framer.md`（68→78行）：细化 task_type 定义，添加明确的示例和反例区分"在多种数据上使用标准方法"（= data_analysis）vs"需要发明新方法"（= discovery）。收紧 complexity_signals 和 contradictions 定义。
2. `task_router.py`：新增规则 #3 — `task_type == "data_analysis" AND signals >= 2 → hybrid`（封顶，永不升级到 innovation）。

**回归测试矩阵**：

| 测试 | 查询 | task_type | contradictions | signals | 路由结果 | 状态 |
|------|------|-----------|---------------|---------|----------|------|
| A: 矛盾查询 | 多组学差异分析+批次校正 | data_analysis | 0 (was 2) | 2 (was 4) | hybrid (was innovation) | ✅ |
| B: 标准DE | 标准差异表达分析 | data_analysis | 0 | 0 (was 3) | routine (was innovation) | ✅ |
| C: 机制发现 | 发现新基因调控机制 | discovery | 0 | 3 (was 4) | innovation | ✅ 无回归 |
| D: 相关性分析 | 基因表达-表型相关性 | data_analysis | 0 | 0 (was 1) | routine (was hybrid) | ✅ |

### E2E 测试结果
| 测试 | 模式 | 结果 | 耗时 | 关键验证 |
|------|------|------|------|----------|
| P2-T1 | auto (简单查询) | ✅ 通过 | 113s | task_type=routine_processing → routine, 4 stages + 6 criteria |
| P2-T2 | auto (发现性查询) | ✅ 通过 | 440s | task_type=discovery → innovation, 3 method cards, m1 score=0.83, 5 stages + 6 criteria |
| P2-T3 | override routine | ✅ 通过 | 101s | CLI override 生效, 无 auto-routing, 走原有 plan_maker 流程, 6 stages + 7 criteria |

### Bug fix
- 修复 agent parent 冲突：非 auto 模式下 `mode_gated_planning_loop` 包裹了 `high_level_planning_loop`，导致 routine 分支重复添加 → 改为只在 auto 分支创建 `_ModeGatedPlanningAgent`
