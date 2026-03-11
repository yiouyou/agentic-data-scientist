# Innovation OS 开发计划

> 状态：**Phase 3 完成** | 当前阶段：**Phase 3 P3-A ✅ P3-B ✅ P3-C ✅ P3-D ✅ (598 tests + 4 E2E)** | 最后更新：2026-03-11

---

## 目录

1. [项目概述](#1-项目概述)
2. [架构总览](#2-架构总览)
3. [Phase 0：基础增强](#3-phase-0基础增强)
4. [Phase 1：Innovation OS V1.0 核心](#4-phase-1innovation-os-v10-核心)
5. [Phase 2：Innovation OS V1.1 智能层](#5-phase-2innovation-os-v11-智能层)
6. [Phase 3：Innovation OS V1.2 高级特性](#6-phase-3innovation-os-v12-高级特性)
7. [集成点地图](#7-集成点地图)
8. [测试策略](#8-测试策略)
9. [进度跟踪](#9-进度跟踪)

---

## 1. 项目概述

### 1.1 目标

将现有"以计划为中心（plan-centric）"的执行系统，演进为"计划治理 + 发现驱动（plan-governed discovery）"的创新操作系统。

### 1.2 核心原则

- **增量式改造**：保留现有 plan-centric 主干，插入 discovery-centric 子系统
- **先方法，后计划**：method → plan，而非 plan 中暗含 method
- **方法对象化**：所有创新方向表示为 Method Card 结构化对象
- **预算受控**：所有发散行为受 budget_controller 约束

### 1.3 参考文档

- `docs/planning_strategy_evolution.md` — 策略分析（英文）
- `docs/planning_strategy_evolution.cn.md` — 策略分析（中文）
- `docs/开发思路1.txt` — Innovation OS 完整设计文档

---

## 2. 架构总览

### 2.1 当前架构

```
User Query
  → high_level_planning_loop (max 10 iterations)
      → plan_maker → plan_reviewer → plan_review_confirmation
  → plan_candidate_selector
  → high_level_plan_parser
  → stage_orchestrator (max 50 iterations)
      per stage:
        → implementation_loop (coding → review → confirmation)
        → criteria_checker
        → stage_reflector
  → summary_agent
```

关键位置（agent.py）：
- 规划循环：line 876–897
- 候选选择器：line 899–905
- 计划解析器：line 914–932
- 阶段编排器：line 1020–1026
- 根工作流组装：line 1031–1053

### 2.2 目标架构（Phase 3 完成后）

```
User Query
  → task_router (routine/hybrid/innovation)
  → [innovation/hybrid only]:
      → problem_framer
      → innovation_trigger_detector
      → method_discovery_layer
          → baseline_method_generator
          → negative_prompting_generator
          → [V1.1+] TRIZ_operator / abduction_operator
      → method_candidate_selector
  → plan_instantiator (method → plan)
  → plan_reviewer → plan_review_confirmation
  → plan_parser
  → stage_orchestrator
      per stage:
        → implementation_loop
        → programmatic_verifier → criteria_checker
        → [V1.1+] method_critic
        → stage_reflector / [V1.2+] stage_decomposer
        → [V1.1+] method_backtracker (if method failure)
  → [V1.1+] deep_verifier
  → summary_agent (with innovation_summary_contract)
```

---

## 3. Phase 0：基础增强

> 目标：独立于 Innovation OS 的即时质量提升

### 3.1 模块清单

| 模块 | 类型 | 新增/修改 | 位置 |
|------|------|-----------|------|
| programmatic_verifier (Tier 1) | 新增 | 新文件 | `core/programmatic_verifier.py` |
| enhanced plan scoring | 修改 | 现有文件 | `core/plan_learning.py` |
| criteria record 扩展 | 修改 | 现有文件 | `core/state_contracts.py` |

### 3.2 programmatic_verifier (Tier 1)

**功能**：在 criteria_checker 之前运行，提供事实性信号。

**Hard checks（阻断级）**：
- 文件存在性：预期输出文件是否存在
- Schema 校验：CSV/JSON 可解析、列数正确
- 脚本执行状态：无 traceback

**Soft signals（信号级）**：
- 数值阈值提取：从结果文件中解析 AUC/p-value 等
- 图片验证：文件非空、格式正确
- 输出覆盖度：预期 artifact 生成比例

**输出 schema**：
```python
{
    "hard_checks": {"file_exists": True, "schema_valid": True, ...},
    "soft_signals": {"auc": 0.88, "p_value": 0.03, ...},
    "verdict": "pass|warn|fail"
}
```

**集成点**：
- `stage_orchestrator.py` line 535 之前（criteria_checker 调用前）
- 验证结果注入 state，criteria_checker prompt 通过 `{programmatic_verification?}` 读取

**实现步骤**：
1. 创建 `core/programmatic_verifier.py`
2. 实现 `run_programmatic_checks(working_dir, stage, criteria) -> VerificationResult`
3. 在 `stage_orchestrator.py` 中 criteria_checker 调用前插入验证调用
4. 修改 `prompts/base/criteria_checker.md`，添加 `{programmatic_verification?}` 占位符
5. 扩展 criteria record：添加 `verified_by` 字段

### 3.3 enhanced plan scoring

**新增评分维度**（`plan_learning.py`）：

| 维度 | 实现 | 成本 |
|------|------|------|
| 依赖图有效性 | 检查 depends_on 构成有效 DAG | 零 |
| 数据流覆盖度 | outputs_produced 覆盖 inputs_required | 零 |
| 粒度均匀性 | 阶段描述长度方差 | 零 |

**实现步骤**：
1. 在 `plan_learning.py` 添加 `_dag_validity_score()`
2. 添加 `_dataflow_coverage_score()`
3. 添加 `_granularity_uniformity_score()`
4. 将新维度集成到 `score_plan_candidate()`

### 3.4 测试项目

**单元测试**（`tests/unit/`）：
- `test_programmatic_verifier.py`：测试各类文件检查逻辑（30 tests）
- `test_plan_learning.py`：扩展现有测试覆盖新评分维度（15 tests）+ P0-T1/P0-T2 正式测试（15 tests）

**规划测试**（plan-only 模式，不执行）：

| 测试名 | 查询 | 验证点 | 状态 |
|--------|------|--------|------|
| P0-T1: 常规分析 | "分析 sales_2023.csv 的销售趋势" | 好计划 > 坏计划，DAG 有效，dataflow 覆盖，粒度均匀性，循环检测 | ✅ 8 tests passed |
| P0-T2: 多阶段依赖 | 复杂 multi-omics 查询 | 结构化计划 > 扁平计划，并行分支有效，评分差距 > 0.10 | ✅ 7 tests passed |

**额外修复**：
- GBK 编码 bug：`cli/main.py` 所有 `click.echo` 替换为 `_safe_echo`，解决 Windows 中文终端 Unicode 字符崩溃
- plan_parser prompt 改善：`prompts/base/plan_parser.md` 示例和指令中增加 `stage_id`/`depends_on`/`inputs_required`/`outputs_produced` 字段

### 3.5 验收标准

- [x] programmatic_verifier 能检测文件存在性、schema 有效性
- [x] 新评分维度在 `score_plan_candidate()` 中生效
- [x] criteria record 包含 `verified_by` 字段
- [x] 所有现有测试（181个）继续通过
- [x] 新增测试覆盖 verifier 和 scoring 逻辑
- [x] P0-T1/P0-T2 正式规划测试全部通过
- [x] plan_parser prompt 引导生成 depends_on/outputs_produced 字段
- [x] GBK 编码修复覆盖所有 CLI 输出路径
- [x] **最终测试结果：249 tests passed（181 原有 + 68 新增）**

---

## 4. Phase 1：Innovation OS V1.0 核心

> 目标：方法对象化 + 多方法候选 + 方法→计划转换

### 4.1 模块清单

| 模块 | 类型 | 位置 |
|------|------|------|
| Method Card schema | 新增 | `core/method_card.py` |
| State Keys 扩展 | 修改 | `core/state_contracts.py` |
| method_discovery_layer | 新增 | `agents/adk/method_discovery.py` |
| method_candidate_selector | 新增 | `agents/adk/method_selector.py` |
| plan_instantiator | 新增 | `agents/adk/plan_instantiator.py` |
| budget_controller | 新增 | `core/budget_controller.py` |
| CLI --innovation-mode | 修改 | `cli/main.py` |
| 方法发现 prompt | 新增 | `prompts/base/method_discovery.md` |
| 方法评分 prompt | 新增 | `prompts/base/method_selector.md` |
| 计划实例化 prompt | 新增 | `prompts/base/plan_instantiator.md` |

### 4.2 Method Card Schema

```python
# core/method_card.py

@dataclass
class MethodCard:
    method_id: str                    # "m1", "m2", ...
    method_family: str                # "baseline|negative_variant"
    title: str
    core_hypothesis: str
    assumptions: list[str]
    invalid_if: list[str]             # 使方法失效的条件
    cheap_test: str                   # 低成本验证方案
    failure_modes: list[str]
    required_capabilities: list[str]  # 所需技能/工具类型
    expected_artifacts: list[str]     # 预期产出文件
    orthogonality_tags: list[str]     # 用于多样性评分
    status: str                       # "proposed|selected|standby|failed|succeeded"
    selection_score: float | None
    rejection_reason: str | None

def make_method_card(**kwargs) -> dict:
    """Create a normalized method card record."""
    ...

def validate_method_card(card: dict) -> list[str]:
    """Validate method card structure, return list of errors."""
    ...
```

### 4.3 method_discovery_layer

**策略**：baseline + negative prompting（V1.0 暂不做 TRIZ/abduction 专用算子）

```
Round 1: 生成 baseline method card
  → LLM 根据用户查询生成常规分析方法

Round 2: 生成 alternative method card (negative prompting)
  → Prompt: "已有一个常规方法（摘要: {baseline_summary}）。
     生成一个根本不同的分析策略。
     不要使用 {baseline_core_methods}。"

[innovation_mode only]
Round 3: 生成第 3 个 method card (更强 negative prompting)
  → 排除前两个方法的核心假设
```

**实现为 ADK BaseAgent 子类**（类似 StageOrchestratorAgent）：
- 内部循环调用 LLM 生成 method cards
- 每轮注入前一轮的排除约束
- 结果写入 `state[StateKeys.METHOD_CANDIDATES]`

### 4.4 method_candidate_selector

**评分公式**：
```
score(method) =
    0.30 * feasibility_score        # 当前工具/数据能否执行
  + 0.20 * orthogonality_score      # 与其他方法的差异度
  + 0.15 * cheap_testability_score  # 能否低成本验证
  + 0.15 * capability_coverage      # 所需技能的可用比例
  + 0.10 * novelty_score            # 是否引入不同核心假设
  + 0.10 * baseline_bonus           # baseline 获得稳定性加分
  - similarity_penalty              # 与已有方法过于相似则扣分
```

**输出**：
- `state[StateKeys.SELECTED_METHOD]` = top-1 method card
- `state[StateKeys.STANDBY_METHODS]` = [top-2, ...] 备用方法
- `state[StateKeys.METHOD_SELECTION_TRACE]` = 评分详情

### 4.5 plan_instantiator

**职责**：将 selected method card 展开为可执行研究计划

**与现有模块的关系**：
- 替代 `plan_maker` 在 innovation/hybrid 模式下的角色
- 输出格式与 plan_maker 相同（自然语言计划文本）
- 后续仍然经过 `plan_reviewer` → `plan_review_confirmation` → `plan_parser`

**prompt 设计要点**：
- 接收 method card 的结构化字段作为输入
- 必须在计划中体现 method card 的 core_hypothesis
- 必须包含 cheap_test 对应的验证阶段
- 阶段标注 `source_method_id`

### 4.6 budget_controller

```python
# core/budget_controller.py

@dataclass
class InnovationBudget:
    method_generation: int = 2       # 允许生成多少个 method cards
    backtrack: int = 0               # 允许多少次方法级回退（V1.0 = 0）
    decomposition: int = 0           # 允许多少次阶段分解（V1.0 = 0）
    verification: int = 1            # 允许多少轮深度验证

BUDGET_PRESETS = {
    "routine":    InnovationBudget(method_generation=0, verification=1),
    "hybrid":     InnovationBudget(method_generation=2, backtrack=1, verification=2),
    "innovation": InnovationBudget(method_generation=3, backtrack=1, decomposition=1, verification=3),
}
```

### 4.7 CLI 扩展

```python
# cli/main.py 新增参数
@click.option(
    '--innovation-mode',
    type=click.Choice(['routine', 'hybrid', 'innovation']),
    default='routine',
    help='Innovation mode: routine (standard), hybrid (limited discovery), innovation (full discovery)',
)
```

### 4.8 根工作流重组

```python
# agent.py 中根工作流组装逻辑

if innovation_mode == "routine":
    # 原有流程不变
    workflow = SequentialAgent(sub_agents=[
        high_level_planning_loop,
        plan_candidate_selector,
        high_level_plan_parser,
        stage_orchestrator,
        summary_agent,
    ])
elif innovation_mode in ("hybrid", "innovation"):
    workflow = SequentialAgent(sub_agents=[
        method_discovery_layer,          # 新增：生成 method cards
        method_candidate_selector,       # 新增：选择最佳方法
        plan_instantiator_loop,          # 新增：方法 → 计划（含 reviewer）
        plan_candidate_selector,         # 复用：排名计划候选
        high_level_plan_parser,          # 复用：解析计划
        stage_orchestrator,              # 复用：阶段执行（含 programmatic_verifier）
        summary_agent,                   # 复用：总结
    ])
```

### 4.9 Stage Record 扩展

```python
# state_contracts.py 扩展
# make_stage_record() 新增字段：
{
    ...existing fields...,
    "source_method_id": str | None,    # 来源方法 ID
    "method_family": str | None,       # 方法族
    "mode": str | None,                # "execution|exploration|verification"
}
```

### 4.10 测试项目

**单元测试**（`tests/unit/`）：
- `test_method_card.py`：Method Card 创建、校验、序列化
- `test_budget_controller.py`：预算预设、消耗、耗尽检测
- `test_method_selector.py`：评分公式、排名逻辑
- 扩展 `test_state_contracts.py`：新字段兼容性

**规划测试**（plan-only，不执行）：

| 测试名 | 模式 | 查询 | 验证点 |
|--------|------|------|--------|
| P1-T1: routine 不触发发现 | routine | "整理实验数据生成报告" | 无 method cards 生成，走原有流程 |
| P1-T2: hybrid 生成 2 个方法 | hybrid | "分析基因表达数据找差异基因" | 生成 2 个 method cards（baseline + alternative），选择 top-1 展开 |
| P1-T3: innovation 生成 3 个方法 | innovation | multi-omics 整合查询 | 生成 3 个 method cards，方法族不同，top-1 展开为计划 |
| P1-T4: 方法多样性 | innovation | "提高预测性能同时保持可解释性" | alternative methods 的 core_hypothesis 与 baseline 显著不同 |
| P1-T5: 计划追溯 | hybrid | 任意研究查询 | 生成的计划 stages 标注 source_method_id |

### 4.11 验收标准

- [x] Method Card schema 定义完整，校验逻辑通过
- [x] method_discovery_layer 在 hybrid 模式生成 2 个、innovation 模式生成 3 个 method cards
- [x] method_candidate_selector 评分并选择 top-1，保留 standby
- [x] plan_instantiator 将 method card 展开为符合现有格式的计划
- [x] routine 模式完全走原有流程，无性能退化
- [x] budget_controller 正确限制 method 生成数量
- [x] CLI --innovation-mode 参数生效
- [x] 所有现有测试继续通过
- [x] 规划测试 P1-T1~T5 全部通过
- [x] **最终测试结果：300 unit tests passed + 5 端到端规划测试通过**

---

## 5. Phase 2：Innovation OS V1.1 智能层

> 目标：自动模式路由 + 创新触发 + 方法评估 + 方法回退

### 5.1 模块清单

| 模块 | 类型 | 位置 |
|------|------|------|
| problem_framer | 新增 | `agents/adk/problem_framer.py` |
| task_router | 新增 | `agents/adk/task_router.py` |
| innovation_trigger_detector | 新增 | `agents/adk/innovation_trigger.py` |
| method_critic | 新增 | `agents/adk/method_critic.py` |
| method_backtracker | 新增 | `agents/adk/method_backtracker.py` |
| deep_verifier | 新增 | `agents/adk/deep_verifier.py` |
| TRIZ operator | 新增 | `agents/adk/operators/triz.py` |
| abduction operator | 新增 | `agents/adk/operators/abduction.py` |
| innovation_summary_contract | 修改 | `prompts/base/summary.md` |

### 5.2 核心逻辑

**problem_framer**：
- 输入：user query + available skills
- 输出：结构化问题表述（research_goal, task_type, knowns, unknowns, contradictions）
- 实现：LLM agent + schema 校验

**task_router**：
- 输入：problem_framer 输出
- 输出：mode 选择（routine/hybrid/innovation）
- 实现：规则引擎 + LLM 兜底
- 替代 Phase 1 的手动 CLI 选择（CLI 参数保留为 override）

**innovation_trigger_detector**：
- 运行期触发：在 stage_orchestrator 循环内，每个 stage 完成后检查
- 触发条件：review 提示平庸、attempt > 1、指标冲突、异常结果
- 输出：是否进入 discovery mode + 推荐算子

**method_critic**：
- 输入：当前 method card + 实现结果 + verifier 信号 + review 反馈
- 输出：issue_type (execution_failure|method_failure) + recommendation
- 集成点：stage_orchestrator 中 review 不通过时调用

**method_backtracker**：
- 输入：失败方法 + 失败原因 + standby methods
- 输出：切换动作 + negative constraints
- 规则：每 stage 最多 1 次，每任务最多 2 次
- 受 budget_controller.backtrack 预算约束

**deep_verifier**：
- 在所有 stage 完成后、summary_agent 之前运行
- 使用最强模型做全局一致性检查
- 可触发修正 stage 重新进入编排

### 5.3 测试项目

| 测试名 | 模式 | 查询 | 验证点 |
|--------|------|------|--------|
| P2-T1: 自动路由-常规 | auto | "整理CSV数据" | task_router → routine |
| P2-T2: 自动路由-创新 | auto | "发现新的基因调控机制" | task_router → innovation |
| P2-T3: 运行期触发 | hybrid | 模拟 review 返回"方案平庸" | innovation_trigger 触发 |
| P2-T4: 方法回退 | innovation | 模拟 method_critic 判定方法失败 | backtracker 切换到 standby |
| P2-T5: TRIZ 触发 | innovation | "提高分辨率但不增加噪声" | TRIZ operator 被触发 |
| P2-T6: 深度验证 | hybrid | 完整流程 | deep_verifier 检查一致性 |

### 5.4 验收标准

- [x] task_router 对常规/研究/创新任务路由正确（40 tests passed，含 2 个 data_analysis 路由 cap 测试）
- [x] problem_framer 输出结构化问题表述（19 tests passed）
- [x] innovation_trigger 在异常/停滞/平庸时触发（44 tests passed）
- [x] method_critic 正确区分执行失败和方法失败（34 tests passed）
- [x] method_backtracker 成功切换到 standby 方法（17 tests passed）
- [x] backtrack 预算正确消耗和限制（budget_controller 集成测试通过）
- [x] deep_verifier 能识别证据不足的结论（20 tests passed）
- [x] summary 包含方法选择过程的完整说明（innovation_summary_contract 完成）
- [x] TRIZ/abduction operator 推迟到 Phase 3（设计决策：需要 Phase 2 基础设施先就位）
- [x] **最终测试结果：474 unit tests passed（300 Phase 1 + 174 Phase 2 新增），0 failed**
- [x] E2E 规划测试 P2-T1~T3 全部通过（修复了 routine 模式 agent parent 冲突 bug）
- [x] Prompt 精炼热修复：problem_framer.md 分类定义细化 + task_router data_analysis cap 规则（回归测试 4/4 通过）

---

## 6. Phase 3：Innovation OS V1.2 高级特性

> 目标：阶段分解 + 形态学算子 + 创新记忆 + 评测基准

### 6.1 模块清单

| 模块 | 类型 | 位置 |
|------|------|------|
| triz_operator | 新增 (P3-A) | `agents/adk/operators/triz.py` |
| abduction_operator | 新增 (P3-A) | `agents/adk/operators/abduction.py` |
| stage_decomposer | 新增 | `agents/adk/stage_decomposer.py` |
| morphology_operator | 延期至 Phase 4+ | `agents/adk/operators/morphology.py` |
| episodic_innovation_memory | 新增 | `core/innovation_memory.py` |
| benchmark 评测集 | 新增 | `tests/benchmarks/` |

### 6.1.1 P3-A 实现记录

**TRIZ operator** (`operators/triz.py`):
- 纯函数 `generate_triz_candidates()`，在 `MethodDiscoveryAgent` 主循环后条件调用
- 输入：`FRAMED_PROBLEM.contradictions`
- 输出：`method_family="triz_resolution"` 的方法卡
- 不消耗 `budget.method_generation`，作为独立的 operator 轮次
- 6 个 TRIZ 原则映射到数据科学：Separation, Inversion, Dynamization, Prior Counteraction, Nesting, Universality

**Abduction operator** (`operators/abduction.py`):
- 纯函数 `generate_abduction_candidates()`，在 TRIZ 之后条件调用
- 输入：`FRAMED_PROBLEM.unknowns` + `FRAMED_PROBLEM.complexity_signals`（方案1：规划阶段）
- 输出：`method_family="abductive_hypothesis"` 的方法卡
- 要求 LLM 生成 competing hypotheses（至少 2 个竞争假说）

**方案2 备忘（未来增强选项）**:
> Abduction operator 当前使用方案1（规划阶段，FRAMED_PROBLEM 输入）。
> 方案2 可在执行阶段运行，使用 `INNOVATION_TRIGGER` 信号作为输入，
> 在 `stage_orchestrator` 检测到 trigger 时动态生成假说方法卡。
> 优势：基于真实执行反馈，更符合经典 abduction 定义。
> 劣势：需要在 stage_orchestrator 中加入运行时方法发现逻辑，
> 并处理方法卡生命周期中断（plan 已实例化后如何使用新方法卡）。
> 与 P3-B stage_decomposer 和 Phase 2 method_backtracker 有职责重叠。
> 建议：在 P3-B/P3-C 完成后评估是否仍有必要实现方案2。

### 6.2 核心逻辑

**stage_decomposer**：
- 触发条件：review 说"过于复杂"或 attempt > 1
- 将当前 stage 拆分为 2-3 个 sub-stages
- 利用现有 `subtasks` 字段或直接替换 stage 列表
- 受 budget_controller.decomposition 预算约束

**morphology_operator**：
- 多维设计空间枚举
- 输入：problem dimensions
- 输出：组合式 method cards

**episodic_innovation_memory**：
- SQLite 存储（扩展 history_store 或独立表）
- 记录：method_id, family, hypothesis, outcome, failure_reason, negative_constraints
- 用途：selector 参考历史失败、backtracker 避免重复、negative prompting 排除

### 6.3 Benchmark 评测集

| Benchmark | 类型 | 验证目标 |
|-----------|------|----------|
| A: 常规执行 | routine | 不过度创新，成本无显著增加 |
| B: 明显矛盾 | innovation | TRIZ 被正确触发，产出不同方法 |
| C: 异常解释 | innovation | abduction 生成竞争假设 |
| D: 多维设计 | innovation | morphology 构造组合候选 |
| E: 方法回退 | hybrid | 主方法失败后成功切换 |

### 6.4 验收标准

- [ ] 复杂 stage 被正确分解为 sub-stages
- [ ] morphology operator 产出组合式候选
- [ ] 创新记忆正确记录和检索
- [ ] Benchmark A-E 全部通过
- [ ] routine 模式性能无退化

---

## 7. 集成点地图

### 7.1 State Keys 扩展总表

```python
# Phase 0 新增
StateKeys.PROGRAMMATIC_VERIFICATION = "programmatic_verification"

# Phase 1 新增
StateKeys.METHOD_CANDIDATES = "method_candidates"
StateKeys.SELECTED_METHOD = "selected_method"
StateKeys.STANDBY_METHODS = "standby_methods"
StateKeys.METHOD_SELECTION_TRACE = "method_selection_trace"
StateKeys.INNOVATION_MODE = "innovation_mode"
StateKeys.INNOVATION_BUDGET = "innovation_budget"

# Phase 2 新增
StateKeys.FRAMED_PROBLEM = "framed_problem"
StateKeys.INNOVATION_TRIGGER = "innovation_trigger"
StateKeys.METHOD_CRITIC_OUTPUT = "method_critic_output"
StateKeys.BACKTRACK_HISTORY = "backtrack_history"
StateKeys.DEEP_VERIFICATION = "deep_verification"
```

### 7.2 文件修改矩阵

| 文件 | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|------|---------|---------|---------|---------|
| `core/state_contracts.py` | ✏️ criteria 扩展 | ✏️ method/budget keys | ✏️ trigger/critic keys | ✏️ memory keys |
| `core/plan_learning.py` | ✏️ 新评分维度 | — | — | — |
| `core/programmatic_verifier.py` | 🆕 | — | — | — |
| `core/method_card.py` | — | 🆕 | — | — |
| `core/budget_controller.py` | — | 🆕 | ✏️ backtrack | ✏️ decomposition |
| `core/innovation_memory.py` | — | — | — | 🆕 |
| `agents/adk/agent.py` | ✏️ verifier 接入 | ✏️ 工作流重组 | ✏️ 新 agent 接入 | ✏️ decomposer 接入 |
| `agents/adk/stage_orchestrator.py` | ✏️ verifier 调用 | ✏️ method 追溯 | ✏️ critic/trigger | ✏️ decomposer |
| `agents/adk/method_discovery.py` | — | 🆕 | ✏️ TRIZ/abduction | ✏️ morphology |
| `agents/adk/method_selector.py` | — | 🆕 | ✏️ history 信号 | — |
| `agents/adk/plan_instantiator.py` | — | 🆕 | — | — |
| `agents/adk/problem_framer.py` | — | — | 🆕 | — |
| `agents/adk/task_router.py` | — | — | 🆕 | — |
| `agents/adk/method_critic.py` | — | — | 🆕 | — |
| `agents/adk/method_backtracker.py` | — | — | 🆕 | — |
| `agents/adk/deep_verifier.py` | — | — | 🆕 | — |
| `agents/adk/stage_decomposer.py` | — | — | — | 🆕 |
| `cli/main.py` | — | ✏️ --innovation-mode | ✏️ auto 模式 | — |
| `prompts/base/criteria_checker.md` | ✏️ 添加验证上下文 | — | — | — |

（🆕 = 新文件，✏️ = 修改现有文件）

---

## 8. 测试策略

### 8.1 测试层级

```
Level 1: 单元测试 (pytest)
  → 每个新模块的独立逻辑测试
  → 无 LLM 调用，全 mock

Level 2: 集成测试 (pytest)
  → 模块间交互测试
  → Mock LLM 返回固定 JSON

Level 3: 规划测试 (plan-only, 真实 LLM)
  → ADS_PLAN_ONLY=true 模式
  → 验证规划阶段的端到端行为
  → 不执行实际代码

Level 4: Benchmark 评测 (Phase 3)
  → 标准化查询集
  → 对比 routine/hybrid/innovation 模式
```

### 8.2 规划测试执行方式

```powershell
# Phase 0 规划测试
$env:ADS_PLAN_ONLY="true"; uv run agentic-data-scientist --mode orchestrated --verbose "查询..."

# Phase 1+ 规划测试
$env:ADS_PLAN_ONLY="true"; $env:ADS_INNOVATION_MODE="hybrid"; uv run agentic-data-scientist --mode orchestrated --innovation-mode hybrid --verbose "查询..."
```

### 8.3 规划测试验证检查项

每次规划测试后检查：
1. 日志中是否出现预期的 agent 调用序列
2. Method cards 是否生成（如适用）
3. 计划是否包含 source_method_id（如适用）
4. 阶段数量和依赖是否合理
5. 成功标准是否可验证
6. 无 timeout / response_format 错误

---

## 9. 进度跟踪

### Phase 0：基础增强

| 任务 | 状态 | 备注 |
|------|------|------|
| programmatic_verifier 核心逻辑 | ✅ 完成 | `core/programmatic_verifier.py` 新建 |
| enhanced plan scoring | ✅ 完成 | DAG + dataflow + granularity 三维度 |
| criteria record 扩展 | ✅ 完成 | `verified_by` 字段 + `WORKING_DIR` state key |
| stage_orchestrator 集成 | ✅ 完成 | verifier → criteria_checker 链 |
| criteria_checker prompt 更新 | ✅ 完成 | `{programmatic_verification?}` 占位符 |
| plan_parser prompt 改善 | ✅ 完成 | 示例增加 stage_id/depends_on/inputs_required/outputs_produced |
| GBK 编码修复 | ✅ 完成 | `_safe_echo` 覆盖所有 CLI 输出路径 |
| 单元测试 | ✅ 完成 | 249 passed (181 原有 + 68 新增) |
| 规划测试 P0-T1 | ✅ 完成 | 8 tests: 好/坏计划区分、DAG/dataflow/granularity/cycle 验证 |
| 规划测试 P0-T2 | ✅ 完成 | 7 tests: 多阶段依赖、并行分支、评分差距 > 0.10 |

### Phase 1：Innovation OS V1.0

| 任务 | 状态 | 备注 |
|------|------|------|
| Method Card schema | ✅ 完成 | `core/method_card.py`: make/validate/summary |
| budget_controller | ✅ 完成 | `core/budget_controller.py`: presets + consume/remaining |
| State Keys 扩展 | ✅ 完成 | 6 个新 StateKeys + stage record 新增 source_method_id/method_family |
| method_discovery_layer | ✅ 完成 | `agents/adk/method_discovery.py`: MethodDiscoveryAgent(BaseAgent) |
| method_candidate_selector | ✅ 完成 | `agents/adk/method_selector.py`: LLM + programmatic fallback scoring |
| plan_instantiator | ✅ 完成 | `agents/adk/plan_instantiator.py`: method → plan via LLM |
| CLI --innovation-mode | ✅ 完成 | `cli/main.py`: routine/hybrid/innovation, 设 ADS_INNOVATION_MODE env |
| agent.py 工作流重组 | ✅ 完成 | routine 走原有流程; hybrid/innovation 走 discovery→selector→instantiator |
| Prompt 模板（3个） | ✅ 完成 | method_discovery.md, method_selector.md, plan_instantiator.md |
| 单元测试 | ✅ 完成 | 300 passed (249 原有 + 59 新增, 减少 8 个重复测试) |
| 规划测试 P1-T1 | ✅ 通过 | routine 模式: 无 method discovery，走原有流程，5 stages + 6 criteria，128.18s |
| 规划测试 P1-T2 | ✅ 通过 | hybrid 模式: 生成 2 个 method cards (m1=DESeq2, m2=ML预测), 选择 m1 score=0.82, standby=1, 5 stages + 6 criteria，257.76s |
| 规划测试 P1-T3 | ✅ 通过 | innovation 模式: 生成 3 个 method cards (m1=相关性网络, m2=通路知识驱动, m3=潜在空间建模), 选择 m1 score=0.78, standby=2, 5 stages + 8 criteria，424.61s |
| 规划测试 P1-T4 | ✅ 通过 | 方法多样性: 从 T2+T3 验证，alternative methods 的 core_hypothesis 与 baseline 明显不同 |
| 规划测试 P1-T5 | ✅ 通过 | 计划追溯: 从 T2+T3 验证，plan 文本中包含多处 source_method_id 引用 |
| LLM API 修复 | ✅ 完成 | 3 个新 agent 的 LLM 调用修正为 LlmRequest + async generator 模式 |

### Phase 2：Innovation OS V1.1

| 任务 | 状态 | 备注 |
|------|------|------|
| problem_framer | ✅ 完成 | `agents/adk/problem_framer.py` (163行) + prompt, 19 tests |
| task_router | ✅ 完成 | `agents/adk/task_router.py` (156行): 规则引擎 + LLM 兜底, 40 tests (含 data_analysis cap) |
| _ModeGatedPlanningAgent + 根工作流重构 | ✅ 完成 | `agent.py`: auto 分支 + CLI default→auto, 4 tests (ModeGated) |
| innovation_trigger_detector | ✅ 完成 | `agents/adk/innovation_trigger.py` (128行): 纯程序化, 零 LLM 成本, 44 tests |
| method_critic | ✅ 完成 | `agents/adk/method_critic.py` (175行) + prompt: LLM 诊断, 34 tests |
| method_backtracker | ✅ 完成 | `agents/adk/method_backtracker.py` (109行): 纯逻辑 agent, 17 tests |
| stage_orchestrator 集成 (B) | ✅ 完成 | 不通过分支: critic+backtracker; criteria 后: innovation_trigger |
| deep_verifier | ✅ 完成 | `agents/adk/deep_verifier.py` (245行) + prompt: LLM 全局一致性检查, 20 tests |
| innovation_summary_contract | ✅ 完成 | `summary.md` 添加 `{innovation_summary_section?}`, state_contracts 初始化 |
| agent.py 集成 (C) | ✅ 完成 | deep_verifier 插入所有 3 个执行工作流分支 |
| TRIZ operator | ➡️ 推迟 Phase 3 | 需要矛盾检测基础设施（problem_framer.contradictions 先就位） |
| abduction operator | ➡️ 推迟 Phase 3 | 需要异常检测基础设施（innovation_trigger 先就位） |
| 单元测试 | ✅ 完成 | 474 passed (300 原有 + 174 Phase 2 新增) |
| 规划测试 P2-T1~T3 | ✅ 通过 | P2-T1: auto→routine (113s), P2-T2: auto→innovation 3 methods (440s), P2-T3: CLI override routine (101s). 修复了 agent parent 冲突 bug |
| Prompt 精炼热修复 | ✅ 完成 | problem_framer.md 分类定义细化（68→78行）+ task_router 新增 data_analysis cap 规则 + 回归测试 4/4 通过 |

### Phase 3：Innovation OS V1.2

| 任务 | 状态 | 备注 |
|------|------|------|
| triz_operator (P3-A) | ✅ 完成 | `operators/triz.py` + `triz_operator.md` prompt + method_discovery.py 集成 |
| abduction_operator (P3-A) | ✅ 完成 | `operators/abduction.py` + `abduction_operator.md` prompt + 方案1(规划阶段) |
| method_card 扩展 | ✅ 完成 | _VALID_FAMILIES += triz_resolution, abductive_hypothesis |
| P3-A 单元测试 | ✅ 完成 | 45 新增 (test_triz_operator: 25, test_abduction_operator: 20), 全量 519 passed, 0 failed |
| P3-A E2E P3-T1 | ✅ 通过 | innovation mode + contradictions + unknowns: ProblemFramer→discovery(signals=3,contradictions=1), TRIZ+abduction均触发, 5 method cards(3 base + 1 triz_1 + 1 abd_1), selector选中triz_1(score=0.84), ~583s |
| P3-A E2E P3-T2 | ✅ 通过 | auto mode + simple task: ProblemFramer→routine_processing(signals=0,contradictions=0), TaskRouter→routine, MethodDiscovery跳过, MethodSelector跳过, PlanInstantiator跳过, 99s. 验证operators在routine模式完全不触发 |
| P3-A E2E P3-T3 | ✅ 通过 | innovation mode + unknowns无contradictions: ProblemFramer→data_analysis(signals=3,contradictions=0), TRIZ未触发(无contradictions), Abduction触发(3 unknowns+3 signals), 4 method cards(3 base + 1 abd_1), selector选中m1(score=0.84), 5 stages+6 criteria, 507s |
| stage_decomposer (P3-B) | ✅ 完成 | `agents/adk/stage_decomposer.py` (~300行) + prompt + stage_orchestrator集成 + agent.py接入, 20 tests, 全量539 passed |
| episodic_innovation_memory (P3-C) | ✅ 完成 | `core/innovation_memory.py` (212行) SQLite存储 + method_discovery.py集成(episodic_memory_constraints注入) + prompt更新, 43 tests, 全量582 passed |
| Benchmark 评测集 A-E (P3-D) | ✅ 完成 | `tests/unit/test_benchmark_evaluation.py` 16 tests: A(routine跳过), B(TRIZ触发), C(abduction触发), D(TRIZ+abduction同时), E(backtrack切换). D替代morphology改为验证双算子组合. 全量598 passed |
| P3 Final E2E | ✅ 通过 | innovation mode + 多组学复杂任务: ProblemFramer→modeling(signals=2,contradictions=0), 4 candidates(3 base + 1 abd_1), abduction触发(4 unknowns+2 signals), selector选中abd_1(score=0.76), 5 stages+7 criteria, ~659s. 生成高质量abductive假说(PTM-Decoupling Regime Detection) |
| morphology_operator | ➡️ 延期 Phase 4+ | 用户确认延期 |
| 全量回归测试 | ✅ 完成 | **598 passed, 0 failed** (539 Phase3-B + 43 P3-C + 16 P3-D) |
