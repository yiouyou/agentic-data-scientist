# Agentic Data Scientist 快速开始

本指南将帮助你理解并使用 Agentic Data Scientist 的多智能体工作流。

## 安装

```bash
# 使用 uv 从 PyPI 安装
uv tool install agentic-data-scientist

# 或使用 uvx（无需安装）
uvx agentic-data-scientist "your query here"
```

## 前置条件

- Python 3.12 或更高版本
- 至少安装一种编码执行主体 CLI：
  - Claude Code（需要 Node.js）
  - Codex CLI（`codex`）
  - OpenCode CLI（`opencode`）
- API 密钥：
  - 与所选 coding executor profile 对应的 key（例如 `claude_code` 对应 `ANTHROPIC_API_KEY`）
  - 在 `configs/llm_routing.yaml` 中启用的 profile 对应 provider key

## 快速上手

### 1. 配置环境变量

先在项目根目录基于模板创建 `.env`：

```bash
cp .env.example .env
```

再按需填写：

```bash
# 核心必需（以下为 claude_code 示例）
ANTHROPIC_API_KEY=your_anthropic_key_here

# 仅当对应 profile 在 configs/llm_routing.yaml 中启用时才需要
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
# DASHSCOPE_API_KEY=your_dashscope_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here

# 可选覆盖项
# DEFAULT_MODEL=gemini-3.1-pro-preview
# CODING_MODEL=claude-sonnet-4-6
# REVIEW_MODEL=gemini-3.1-pro-preview
# DEFAULT_CODING_EXECUTOR=claude_code
# DIRECT_CODING_EXECUTOR=claude_code
# LLM_ROUTING_CONFIG_PATH=configs/llm_routing.yaml
# DISABLE_NETWORK_ACCESS=true
# LLM_CIRCUIT_BREAKER_ENABLED=true
# LLM_CIRCUIT_BREAKER_FAILURE_THRESHOLD=2
# LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS=120
# LLM_CIRCUIT_BREAKER_MAX_COOLDOWN_SECONDS=1800
# ADS_HISTORY_ENABLED=true
# ADS_HISTORY_DB_PATH=.agentic_ds_history.sqlite3
# ADS_LEARNING_ADVICE_ENABLED=true
# ADS_LEARNING_TOPK=3
# ADS_LEARNING_RECENT_RUNS=200
# ADS_PLAN_SELECTOR_ENABLED=false
# ADS_PLAN_SELECTOR_ROLLOUT_PERCENT=100
# ADS_PLAN_SELECTOR_INTENT_REGEXES=rna-?seq,variant,wgs
# ADS_PLAN_SELECTOR_ROLLOUT_SALT=
# ADS_PLAN_RANK_MIN_SWITCH_MARGIN=0.12
# ADS_PLAN_ONLY=false
# CODEX_COMMAND_TEMPLATE="codex exec --model {model}"
# OPENCODE_COMMAND_TEMPLATE="opencode run --model {model}"
```

获取 API 密钥：
- Anthropic：https://console.anthropic.com/
- OpenAI：https://platform.openai.com/api-keys
- Google：https://aistudio.google.com/app/apikey
- DashScope（Qwen）：https://dashscope.console.aliyun.com/
- DeepSeek：https://platform.deepseek.com/api_keys

可选启动预检：
```bash
agentic-data-scientist --llm-preflight --llm-config configs/llm_routing.yaml
```

离线规划策略回放：
```bash
agentic-data-scientist --history-replay --history-replay-limit 200
```

可使用 `ADS_PLAN_SELECTOR_INTENT_REGEXES` 将学习型计划选择限制在特定任务意图/领域上，实现按任务类型灰度。

### 2. 运行第一条查询

**重要：** 你必须指定 `--mode` 来选择执行策略。

```bash
# 复杂分析：完整工作流
agentic-data-scientist "Perform differential expression analysis" --mode orchestrated --files data.csv

# 快速脚本任务
agentic-data-scientist "Write a Python script to parse CSV" --mode simple

# 问答任务
agentic-data-scientist "Explain gradient boosting" --mode simple
```

### 3. 工作目录选项

默认情况下，文件会保存到 `./agentic_output/`，并在任务完成后保留：

```bash
# 默认行为（保留文件）
agentic-data-scientist "Analyze data" --mode orchestrated --files data.csv

# 临时目录（自动清理）
agentic-data-scientist "Quick exploration" --mode simple --files data.csv --temp-dir

# 自定义目录
agentic-data-scientist "Project analysis" --mode orchestrated --files data.csv --working-dir ./my_project
```

## 理解工作流

提交查询后，Agentic Data Scientist 会执行一个多阶段工作流，以输出高质量、可验证的结果。

### 一次查询会发生什么

```
USER QUERY: "Analyze customer churn patterns in this dataset"
     |
     v
┌────────────────────────────────────────────────────────┐
│ PHASE 1: PLANNING (Iterative)                         │
├────────────────────────────────────────────────────────┤
│ 1. Plan Maker creates comprehensive analysis plan     │
│    - Breaks down task into logical stages             │
│    - Defines clear success criteria                   │
│    - Recommends appropriate methodologies             │
│                                                        │
│ 2. Plan Reviewer validates the plan                   │
│    - Checks completeness                              │
│    - Verifies all requirements are addressed          │
│    - Provides feedback if improvements needed         │
│                                                        │
│ 3. Loop repeats until plan is approved                │
│                                                        │
│ 4. Plan Parser structures it for execution            │
│    - Converts to executable stages                    │
│    - Sets up success criteria tracking                │
│                                                        │
│ RESULT: Validated, comprehensive execution plan       │
└────────────────────────────────────────────────────────┘
     |
     v
┌────────────────────────────────────────────────────────┐
│ PHASE 2: EXECUTION (Stage by Stage)                   │
├────────────────────────────────────────────────────────┤
│ For each stage in the plan:                           │
│                                                        │
│ A. IMPLEMENTATION LOOP (Iterative)                    │
│    1. Coding Agent implements the stage               │
│       - Has access to 380+ scientific Skills          │
│       - Can read/write files, run code                │
│       - Creates scripts, analyses, visualizations     │
│                                                        │
│    2. Review Agent validates implementation           │
│       - Checks code quality and correctness           │
│       - Verifies stage requirements are met           │
│       - Provides specific feedback if issues found    │
│                                                        │
│    3. Loop repeats until approved                     │
│                                                        │
│ B. PROGRESS VALIDATION                                │
│    4. Criteria Checker updates progress               │
│       - Inspects generated files and results          │
│       - Updates which success criteria are now met    │
│       - Provides objective evidence                   │
│                                                        │
│ C. ADAPTIVE REPLANNING                                │
│    5. Stage Reflector adapts remaining work           │
│       - Considers what's been accomplished            │
│       - Identifies what still needs to be done        │
│       - Modifies or extends remaining stages          │
│                                                        │
│ Then proceeds to next stage...                        │
│                                                        │
│ RESULT: All stages implemented and validated          │
└────────────────────────────────────────────────────────┘
     |
     v
┌────────────────────────────────────────────────────────┐
│ PHASE 3: SUMMARY                                       │
├────────────────────────────────────────────────────────┤
│ Summary Agent creates final report                     │
│ - Synthesizes all work performed                      │
│ - Documents key findings and results                  │
│ - Lists all generated files and outputs               │
│ - Provides comprehensive analysis narrative           │
│                                                        │
│ RESULT: Publication-ready comprehensive report        │
└────────────────────────────────────────────────────────┘
```

### 工作流关键特性

**迭代式优化**
- 执行前先评审并打磨计划
- 每个阶段通过验证后再进入下一阶段
- 多次机会在早期发现并修复问题

**自适应执行**
- 实施阶段的新发现会影响后续阶段
- 计划会依据实际进展与发现动态调整
- 能处理预期外洞见

**持续验证**
- 执行全程客观跟踪成功标准
- 清晰展示“已完成”与“待完成”
- 为每条标准提供客观证据

**关注点分离**
- 规划智能体只关注策略，不做实现
- 编码智能体只关注实现，不承担规划
- 评审智能体提供独立校验

## Python API 用法

### 基础用法

```python
from agentic_data_scientist import DataScientist

# 创建实例并运行查询
with DataScientist() as ds:
    result = ds.run("What is data science?")
    print(result.response)

# 访问结果
print(f"Status: {result.status}")
print(f"Duration: {result.duration}s")
print(f"Files created: {result.files_created}")
```

### 上传文件

```python
from agentic_data_scientist import DataScientist

with DataScientist() as ds:
    result = ds.run(
        "Analyze trends in this time series data",
        files=[
            ("sales.csv", open("sales.csv", "rb").read()),
            ("inventory.csv", open("inventory.csv", "rb").read()),
        ]
    )
    print(result.response)
    print(f"Working directory: {ds.working_dir}")
```

### 带流式输出的异步调用

```python
import asyncio
from agentic_data_scientist import DataScientist

async def analyze_data():
    async with DataScientist() as ds:
        async for event in await ds.run_async(
            "Perform differential expression analysis",
            files=[("data.csv", open("data.csv", "rb").read())],
            stream=True
        ):
            # 实时观察工作流
            if event['type'] == 'message':
                author = event['author']
                content = event['content']
                print(f"[{author}] {content}")
            elif event['type'] == 'completed':
                print(f"Completed in {event['duration']}s")

asyncio.run(analyze_data())
```

### 多轮对话

```python
import asyncio
from agentic_data_scientist import DataScientist

async def chat():
    async with DataScientist() as ds:
        context = {}

        # 第一轮
        result1 = await ds.run_async(
            "What are the main techniques for dimensionality reduction?",
            context=context
        )
        print("AI:", result1.response)

        # 第二轮（保留上下文）
        result2 = await ds.run_async(
            "Which one would you recommend for high-dimensional gene expression data?",
            context=context
        )
        print("AI:", result2.response)

asyncio.run(chat())
```

## 理解流式事件

使用 `stream=True` 时，工作流推进过程中会持续返回事件：

```python
async for event in await ds.run_async("Your query", stream=True):
    event_type = event['type']

    if event_type == 'message':
        # 智能体输出的普通文本
        print(f"[{event['author']}] {event['content']}")

    elif event_type == 'function_call':
        # 智能体调用工具
        print(f"Calling {event['name']}...")

    elif event_type == 'function_response':
        # 工具返回结果
        print(f"Tool {event['name']} completed")

    elif event_type == 'usage':
        # Token 使用信息
        tokens = event['usage']
        print(f"Tokens: {tokens['total_input_tokens']} in, {tokens['output_tokens']} out")

    elif event_type == 'completed':
        # 工作流完成
        print(f"Done in {event['duration']}s")
        print(f"Created {len(event['files_created'])} files")
```

## 执行模式

### Orchestrated 模式（推荐）

完整多智能体工作流，包含规划、验证和自适应执行。

**适用场景：**
- 复杂数据分析
- 多步骤工作流
- 需要质量验证的任务
- 生产级分析

**示例：**
```bash
agentic-data-scientist "Perform DEG analysis comparing treatment vs control" \
  --mode orchestrated \
  --files treatment.csv --files control.csv
```

### Simple 模式

直接编码执行，不走规划开销。

**适用场景：**
- 快速脚本
- 代码生成
- 问答
- 快速原型

**示例：**
```bash
agentic-data-scientist "Write a function to merge CSV files" --mode simple
```

## 下一步

更多 API、CLI、定制化和技术架构内容，请查看 `docs/` 目录下的其他文档。

## 故障排查

### 常见问题

**ImportError: No module named 'agentic_data_scientist'**
- 安装包：`pip install agentic-data-scientist` 或 `uv sync`

**API Key Errors**
- 确认 `.env` 文件位置正确
- 确认 API key 有效且可用
- 确认账户余额/额度足够
- 确认已为 `configs/llm_routing.yaml` 中启用的 profile 配置对应 key
- 运行预检定位问题：`agentic-data-scientist --llm-preflight --llm-config configs/llm_routing.yaml`

**Node.js Issues**
- 确认已安装 Node.js：`node --version`
- 当 coding executor 为 `claude_code` 时需要 Node.js
- 安装后重启终端

**Workflow Seems Stuck**
- 开启流式输出查看进度：`--stream` 或 `stream=True`
- 查看日志中的错误信息
- 复杂计算可能执行较久

### 获取帮助

- 查看 `docs/` 目录中的完整文档
- 在 [GitHub](https://github.com/K-Dense-AI/agentic-data-scientist/issues) 提交 issue
