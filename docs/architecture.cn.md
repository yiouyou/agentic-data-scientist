# 架构与技术设计

本文档说明 Agentic Data Scientist 的技术内部实现、设计决策与实现细节。

## 目录

- [智能体层级](#智能体层级)
- [工作流设计动机](#工作流设计动机)
- [上下文窗口管理](#上下文窗口管理)
- [事件压缩系统](#事件压缩系统)
- [循环检测](#循环检测)
- [阶段编排](#阶段编排)
- [评审确认逻辑](#评审确认逻辑)
- [性能考量](#性能考量)

## 智能体层级

ADK 工作流由多个专用智能体构成，采用层级结构：

```
Workflow Root (SequentialAgent)
├── Planning Loop (NonEscalatingLoopAgent)
│   ├── Plan Maker (LoopDetectionAgent)
│   ├── Plan Reviewer (LoopDetectionAgent)
│   └── Plan Review Confirmation (LoopDetectionAgent)
├── Plan Parser (LoopDetectionAgent)
├── Stage Orchestrator (Custom Agent)
│   └── For each stage:
│       ├── Implementation Loop (NonEscalatingLoopAgent)
│       │   ├── Coding Agent (ClaudeCodeAgent)
│       │   ├── Review Agent (LoopDetectionAgent)
│       │   └── Implementation Review Confirmation (LoopDetectionAgent)
│       ├── Criteria Checker (LoopDetectionAgent)
│       └── Stage Reflector (LoopDetectionAgent)
└── Summary Agent (LoopDetectionAgent)
```

### 智能体类型

**LoopDetectionAgent**
- 在 ADK 的 LlmAgent 基础上增加自动循环检测
- 监控输出中的重复模式
- 在智能体卡住时阻止无限生成
- 用于所有基于 LLM 的规划与评审智能体

**ClaudeCodeAgent**
- 封装 Claude Code SDK，负责实现工作
- 可访问 380+ 科学技能
- 具备文件系统与代码执行能力
- 出于安全原因受 working directory 沙箱限制

**NonEscalatingLoopAgent**
- 管理迭代优化流程且不升级失败信号
- 允许多轮反馈而不让工作流直接失败
- 用于规划循环和实现循环
- 防止拒绝信号向上层传播

**StageOrchestratorAgent**
- 自定义编排器，管理逐阶段执行
- 协调实现循环、标准检查和反思
- 基于进展执行自适应重规划
- 维护阶段状态与成功标准跟踪

## 工作流设计动机

### 为什么将规划与执行分离？

**问题：** 直接实现常常遗漏需求，导致返工和结果不完整。

**方案：** 引入独立规划阶段并进行验证
- Plan Maker 只关注完整规划
- Plan Reviewer 提供独立校验
- 先定义成功标准，再进入实现
- 在成本最低的早期发现问题

**权衡：** 会增加 API 调用与耗时，但结果质量显著更高。

### 为什么采用迭代优化？

**问题：** 单次规划或单次实现通常存在质量缺陷。

**方案：** 在规划和实现两端都引入评审循环
- 规划循环：直到计划完整并通过验证
- 实现循环：直到代码满足要求
- 每一轮迭代都增量提升质量

**机制：** NonEscalatingLoopAgent 支持多轮迭代而不触发失败升级。

### 为什么要自适应重规划？

**问题：** 刚性计划无法适应实现过程中出现的新发现。

**方案：** 每个阶段后由 Stage Reflector 调整剩余阶段
- 分析已完成内容
- 识别仍需完成的工作
- 修改或扩展剩余阶段
- 保证最终交付符合实际需求

**收益：** 计划基于现实动态演进，而非停留在初始假设。

### 为什么持续验证？

**问题：** 多阶段流程中很难客观跟踪进度。

**方案：** 每个阶段后由 Criteria Checker 更新成功标准状态
- 检查生成文件和结果
- 更新已满足的标准
- 提供客观进展证据
- 清晰展示剩余工作

**收益：** 可量化进度，避免模糊推进与需求遗漏。

## 上下文窗口管理

框架采用激进的上下文管理策略，以在模型 token 上限内支持长时间分析。

### 挑战

多阶段分析会产生数百条事件：
- 每次智能体轮次都会增加多条事件（消息、工具调用、工具响应）
- 事件会在整个工作流中持续累积
- 若不管理，上下文会超过 1M token 限制
- token 溢出会导致工作流失败

### 策略总览

采用多层保护：
1. **回调压缩**：每次智能体轮次后自动触发
2. **手动压缩**：在关键编排点触发
3. **硬上限裁剪**：紧急兜底
4. **大文本截断**：避免单条事件占用过多 token

### 事件压缩系统

#### 工作方式

压缩系统通过基于 LLM 的摘要，在删除旧事件的同时保留关键上下文：

1. **阈值检测**：每轮智能体执行后检测事件数
2. **生成摘要**：超过阈值后，用 LLM 总结旧事件
3. **替换事件**：旧事件替换为单条摘要事件
4. **截断文本**：对剩余事件中的大文本进行截断（>5KB）
5. **直接赋值**：使用 `session.events = new_events`，确保 ADK 识别变更

#### 关键实现细节

```python
# Compression triggered when events exceed threshold
if len(events) > EVENT_THRESHOLD:
    # Summarize old events using LLM
    summary = await generate_summary(old_events)

    # Replace old events with summary
    new_events = [summary_event] + recent_events

    # Truncate large text in remaining events
    truncated_events = truncate_large_text(new_events)

    # Direct assignment (not append/pop) to ensure ADK sees change
    session.events = truncated_events
```

#### 为什么必须直接赋值？

初版使用 `pop()` 修改列表，但 ADK 的 session service 无法识别这些变化。改为直接列表赋值（`session.events = new_events`）后，ADK 才能正确更新上下文。

#### 压缩参数

- **EVENT_THRESHOLD**：30 条事件（触发压缩阈值）
- **EVENT_OVERLAP**：10 条事件（保留的近期上下文）
- **MAX_EVENTS**：50 条事件（紧急裁剪硬上限）
- **TRUNCATE_SIZE**：5000 字符（单条事件文本最大长度）

这些偏激进的默认值可保证复杂分析期间上下文始终可控。

### 防止 token 溢出

#### 回调压缩

```python
def compression_callback(session):
    """Called after each agent turn."""
    if len(session.events) > EVENT_THRESHOLD:
        compress_events(session)
```

通过 ADK 回调系统自动触发。

#### 手动压缩

```python
# After implementation loop completes
await compress_session_events(session, force=True)
```

在关键编排节点触发（例如实现循环后、反思前）。

#### 硬上限裁剪

```python
if len(session.events) > MAX_EVENTS:
    # Emergency: discard oldest events
    session.events = session.events[-MAX_EVENTS:]
```

当压缩不足时的安全兜底。

#### 大文本截断

```python
def truncate_event_text(event, max_size=5000):
    """Truncate large text content in events."""
    if len(event.text) > max_size:
        event.text = event.text[:max_size] + "... [truncated]"
    return event
```

避免单条事件消耗过多 token。

### 为什么这很重要

若没有激进压缩：
- 复杂分析会突破 1M token 上限
- 工作流中途失败
- 用户可能损失数小时进展

有压缩后：
- 分析可持续数小时并产生数百事件
- 总上下文保持在 1M token 以下
- 工作流能够顺利完成

## 循环检测

### 问题

LLM 智能体有时会进入无限循环：
- 重复同样输出
- 陷入循环论证
- 生成无穷无尽的同质变体

### 方案

LoopDetectionAgent 监控输出中的重复模式：

```python
class LoopDetectionAgent(LlmAgent):
    def __init__(self, min_pattern_length=200, repetition_threshold=3):
        self.min_pattern_length = min_pattern_length
        self.repetition_threshold = repetition_threshold
        self.output_history = []

    def detect_loop(self, new_output):
        """Detect if output is repeating."""
        self.output_history.append(new_output)

        # Check for repeated patterns
        for i in range(len(self.output_history) - 1):
            if self.is_similar(self.output_history[i], new_output):
                repetition_count += 1

        if repetition_count >= self.repetition_threshold:
            raise LoopDetectedError("Agent is repeating itself")
```

### 参数

- **min_pattern_length**：纳入检测的最小文本长度（默认 200 字符）
- **repetition_threshold**：触发检测前允许的重复次数（默认 3）

### 有效性

- 可在 token 大量浪费前识别卡住状态
- 保留合理迭代空间，同时阻止无限循环
- 阈值可针对不同智能体类型调优

## 阶段编排

### StageOrchestratorAgent

自定义智能体，负责逐阶段执行：

```python
class StageOrchestratorAgent:
    async def run_stage(self, stage):
        # 1. Implementation Loop
        implementation = await self.run_implementation_loop(stage)

        # 2. Compress events (manual)
        await compress_session_events(self.session)

        # 3. Check Success Criteria
        criteria_update = await self.check_criteria()

        # 4. Reflect and Adapt
        adapted_stages = await self.reflect_on_progress()

        return adapted_stages
```

### 阶段流程

1. **实现循环**
   - Coding Agent 实现当前阶段
   - Review Agent 评审实现
   - Review Confirmation 决定继续循环还是退出
   - 直到实现通过评审

2. **事件压缩**
   - 实现循环后手动压缩
   - 防止长实现过程导致上下文溢出
   - 通过摘要保留关键上下文

3. **标准检查**
   - Criteria Checker 检查输出文件
   - 更新已满足的成功标准
   - 提供客观进展证据

4. **反思与调整**
   - Stage Reflector 分析进展
   - 识别剩余工作
   - 调整或扩展后续阶段
   - 返回更新后的阶段列表

### 设计原因

**关注点分离**：每个子智能体职责聚焦。  
**显式压缩**：关键节点手动压缩保证上下文可控。  
**自适应规划**：每阶段反思，支持计划动态调整。  
**客观进度**：成功标准检查提供可度量进展。

## 评审确认逻辑

### 挑战

工作流如何判断何时退出评审循环？

### 方案

使用专门的确认智能体解析评审反馈并做退出决策：

```python
class ReviewConfirmationAgent(LoopDetectionAgent):
    """Decides whether to exit review loop."""

    instruction = """
    Review the feedback and decide:
    - exit: true if approved, false if needs revision
    - reason: explanation for decision

    Output JSON: {"exit": true/false, "reason": "..."}
    """
```

### 运行机制

1. **Plan Review Confirmation**
   - 接收计划与评审反馈
   - 判断计划是否足够完整可继续
   - 输出结构化决策：`{"exit": true, "reason": "..."}`

2. **Implementation Review Confirmation**
   - 接收实现与评审反馈
   - 判断实现是否满足要求
   - 输出结构化决策：`{"exit": true, "reason": "..."}`

### 为什么使用结构化输出？

- 工作流中有清晰决策点
- 批准/拒绝原因明确
- 易于解析与记录
- 避免模糊的循环退出条件

### 退出条件

**退出循环（通过）：**
- 评审反馈为正面
- 要求已满足
- 无重大问题

**继续循环（不通过）：**
- 评审发现问题
- 要求未完全满足
- 需要修订

## 性能考量

### Token 使用

**规划阶段**：10k-50k tokens
- 多轮计划生成与评审
- 结构化输出
- 文件检查

**每阶段实现**：50k-200k tokens
- 多轮实现与评审
- 代码生成与执行
- 文件操作与检查

**复杂分析总量**：500k-1M tokens
- 多阶段累积
- 事件压缩确保不超限
- 为质量提升付出的成本通常值得

### 延迟

**Orchestrated 模式**：复杂分析约 5-30 分钟
- 规划：1-3 分钟
- 实现：每阶段 2-10 分钟
- 评审与验证：每轮 1-2 分钟

**Simple 模式**：30 秒到 5 分钟
- 无规划开销
- 单次实现
- 无验证循环

### 成本

**Orchestrated 模式**：每次分析约 $1-10（随复杂度波动）
- 多次 LLM 调用
- 规划与评审智能体（经 OpenRouter）
- 编码智能体（Claude Sonnet 4.5）

**Simple 模式**：每次任务约 $0.10-1
- 单次编码智能体调用
- 无规划/评审开销

### 成本何时值得

**使用 Orchestrated 模式：**
- 生产分析
- 复杂多阶段任务
- 关键业务决策
- 出错代价高的场景

**使用 Simple 模式：**
- 快速探索
- 简单脚本
- 学习与开发
- 预算敏感任务

## 内存管理

### 工作目录隔离

每个会话都有独立工作目录：
- 智能体仅在该目录内活动
- 会话间不能互访文件
- 临时目录自动清理
- 项目目录可保留

### 会话状态

状态保存在 ADK session：
```python
session.state = {
    'high_level_plan': "...",
    'high_level_stages': [...],
    'high_level_success_criteria': [...],
    'stage_implementations': [...],
    'current_stage': {...},
}
```

所有智能体共享该状态进行协作。

### 事件队列

事件保存在 session 中：
- 消息事件（智能体输出）
- 函数调用事件（工具使用）
- 函数响应事件（工具结果）
- 用量事件（token 统计）

通过压缩机制维持在限制内。

## 错误处理

### 升级策略

**NonEscalatingLoopAgent**：捕获拒绝但不升级
- 支持迭代改进
- 防止单次拒绝导致整条工作流失败

**LoopDetectionAgent**：在卡住时升级
- 检测无限循环
- 避免资源浪费

**StageOrchestrator**：优雅处理阶段失败
- 记录错误
- 可重试或跳过阶段
- 提供可理解的错误信息

### 恢复机制

1. **重试逻辑**：失败阶段可重试
2. **优雅降级**：失败时尽量返回部分结果
3. **错误上下文**：完整错误细节写入日志，便于调试
4. **状态保留**：会话状态保留，便于后续分析

## 设计原则

1. **关注点分离**：每个智能体只有一个清晰职责
2. **显式优于隐式**：决策点与状态迁移清晰可见
3. **迭代优化**：多轮迭代提升质量
4. **客观验证**：成功标准可度量
5. **上下文管理**：长流程使用激进压缩策略
6. **循环快速失败**：尽快识别并终止无限循环
7. **阶段优雅失败**：出错时尽量不丢失整体进展

## 未来改进方向

潜在优化包括：

**并行阶段执行**：并发运行互相独立的阶段  
**流式压缩**：向用户流式输出同时进行压缩  
**自适应阈值**：按 token 使用动态调整压缩参数  
**阶段检查点**：支持从任意阶段保存/恢复  
**成本优化**：按子任务选择更低成本模型
