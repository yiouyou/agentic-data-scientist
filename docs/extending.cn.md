# 扩展 Agentic Data Scientist

本指南说明如何自定义和扩展 Agentic Data Scientist 的多智能体工作流。

## 目录

- [理解智能体层级](#理解智能体层级)
- [自定义提示词](#自定义提示词)
- [自定义智能体](#自定义智能体)
- [自定义 MCP 工具集](#自定义-mcp-工具集)
- [自定义事件处理器](#自定义事件处理器)
- [集成示例](#集成示例)

## 理解智能体层级

ADK 工作流由多个专用智能体组成，并按阶段组织：

```
Workflow Root (SequentialAgent)
├── Planning Loop (NonEscalatingLoopAgent)
│   ├── Plan Maker (LoopDetectionAgent)
│   ├── Plan Reviewer (LoopDetectionAgent)
│   └── Review Confirmation (LoopDetectionAgent)
├── Plan Parser (LoopDetectionAgent)
├── Stage Orchestrator (Custom Agent)
│   └── For each stage:
│       ├── Implementation Loop (NonEscalatingLoopAgent)
│       │   ├── Coding Agent (ClaudeCodeAgent)
│       │   ├── Review Agent (LoopDetectionAgent)
│       │   └── Review Confirmation (LoopDetectionAgent)
│       ├── Criteria Checker (LoopDetectionAgent)
│       └── Stage Reflector (LoopDetectionAgent)
└── Summary Agent (LoopDetectionAgent)
```

### 关键智能体类型

**LoopDetectionAgent**：扩展 ADK 的 LlmAgent，具备循环检测能力，防止无限生成。  
**ClaudeCodeAgent**：对 Claude Code SDK 的封装，用于带工具访问的实现任务。  
**NonEscalatingLoopAgent**：管理迭代过程，不向上层传播升级（escalation）标志。  
**StageOrchestratorAgent**：自定义编排器，负责按阶段执行。

## 自定义提示词

工作流中的每个智能体都由提示词模板驱动。你可以通过修改模板来改变智能体行为。

### 提示词结构

提示词存放于 `src/agentic_data_scientist/prompts/`：

```
prompts/
├── base/
│   ├── plan_maker.md               # 生成分析计划
│   ├── plan_reviewer.md            # 评审计划完整性
│   ├── plan_review_confirmation.md # 判断计划是否通过
│   ├── plan_parser.md              # 将计划结构化为阶段
│   ├── coding_review.md            # 评审实现结果
│   ├── implementation_review_confirmation.md  # 判断实现是否通过
│   ├── criteria_checker.md         # 检查成功标准
│   ├── stage_reflector.md          # 调整剩余阶段
│   ├── summary.md                  # 生成最终报告
│   └── global_preamble.md          # 所有智能体共享上下文
└── domain/
    └── bioinformatics/             # 领域定制
        ├── science_methodology.md
        └── interactive_base.md
```

### 加载自定义提示词

```python
from agentic_data_scientist.prompts import load_prompt

# 加载基础提示词
plan_maker_prompt = load_prompt("plan_maker")

# 加载领域提示词
bio_prompt = load_prompt("science_methodology", domain="bioinformatics")
```

### 创建自定义提示词

1. **在 `prompts/base/` 或 `prompts/domain/your_domain/` 新建提示词文件**

示例：`prompts/base/custom_plan_maker.md`

```markdown
$global_preamble

You are a specialized planning agent for [your domain].

# Your Role

Create detailed analysis plans for [specific task type].

# Output Format

Provide structured plans containing:
1. **Analysis Stages** - Step-by-step breakdown
2. **Success Criteria** - How to verify completion
3. **Recommended Approaches** - Domain-specific methods

# Domain Knowledge

[Include specific expertise, methodologies, or considerations]

# Context

**User Request:**
{original_user_input?}
```

2. **加载并使用自定义提示词**

```python
from agentic_data_scientist.prompts import load_prompt

custom_prompt = load_prompt("custom_plan_maker")
```

### 自定义特定智能体提示词

要定制某个智能体行为，直接修改其对应提示词。

**示例：为金融分析定制 Plan Maker**

创建 `prompts/domain/finance/plan_maker.md`：

```markdown
$global_preamble

You are a financial data science strategist specializing in quantitative analysis.

# Your Role

Transform financial analysis requests into comprehensive, risk-aware plans.

# Financial Analysis Stages

Focus on:
1. Data quality and compliance verification
2. Risk assessment and statistical validation
3. Regulatory compliance checks
4. Backtesting and validation strategies

# Success Criteria Requirements

Every plan must include:
- Data quality thresholds
- Statistical significance requirements
- Risk metrics and controls
- Audit trail requirements

[... rest of customized prompt ...]
```

然后加载：

```python
financial_prompt = load_prompt("plan_maker", domain="finance")
```

**说明**：模型通过环境变量（`DEFAULT_MODEL`、`CODING_MODEL`）和 `configs/llm_routing.yaml` 配置，支持多提供商路由。

### 提示词变量

提示词支持运行时插值变量：

- `{original_user_input?}`：用户查询
- `{high_level_plan?}`：当前计划
- `{high_level_stages?}`：阶段列表
- `{high_level_success_criteria?}`：成功标准
- `{stage_implementations?}`：已完成阶段摘要
- `{current_stage?}`：当前执行阶段
- `{implementation_summary?}`：实现输出摘要
- `{review_feedback?}`：评审反馈

## 自定义智能体

### 扩展现有智能体

你可以通过修改版本来定制现有角色：

```python
from google.adk.agents import LlmAgent
from google.genai import types
from agentic_data_scientist.agents.adk.loop_detection import LoopDetectionAgent
from agentic_data_scientist.agents.adk.utils import DEFAULT_MODEL, get_generate_content_config
from agentic_data_scientist.prompts import load_prompt

def create_custom_plan_maker(tools):
    """Create a custom plan maker with specialized behavior."""

    # Load custom prompt
    custom_instructions = load_prompt("custom_plan_maker", domain="finance")

    # DEFAULT_MODEL is a LiteLLM model instance configured to use OpenRouter
    return LoopDetectionAgent(
        name="custom_plan_maker",
        model=DEFAULT_MODEL,  # Automatically routed through OpenRouter
        description="Custom financial planning agent",
        instruction=custom_instructions,
        tools=tools,
        output_key="high_level_plan",
        generate_content_config=get_generate_content_config(temperature=0.4),
        # Custom loop detection thresholds
        min_pattern_length=300,
        repetition_threshold=4,
    )
```

### 创建新的智能体角色

你也可以向工作流新增全新智能体：

```python
from google.adk.agents import InvocationContext
from google.adk.events import Event
from google.genai import types
from typing import AsyncGenerator

class ValidationAgent(LoopDetectionAgent):
    """Custom validation agent for specific checks."""

    def __init__(self, validation_rules, **kwargs):
        super().__init__(**kwargs)
        self.validation_rules = validation_rules

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """Custom validation logic."""
        state = ctx.session.state

        # Get implementation results
        implementation = state.get("implementation_summary", "")

        # Apply custom validation rules
        validation_results = []
        for rule_name, rule_fn in self.validation_rules.items():
            passed = rule_fn(implementation)
            validation_results.append({
                'rule': rule_name,
                'passed': passed
            })

        # Store results
        state["validation_results"] = validation_results

        # Yield results as event
        summary = f"Validation: {sum(r['passed'] for r in validation_results)}/{len(validation_results)} checks passed"
        yield Event(
            author=self.name,
            content=types.Content(
                role="model",
                parts=[types.Part(text=summary)]
            ),
        )
```

### 修改工作流

如需将自定义智能体接入工作流，需要调整 agent 工厂：

```python
from agentic_data_scientist.agents.adk.agent import create_agent
import logging

logger = logging.getLogger(__name__)

def create_custom_workflow(working_dir, mcp_servers=None):
    """Create workflow with custom agents."""

    # Get standard agents
    from agentic_data_scientist.agents.adk.agent import (
        create_agent as base_create_agent
    )

    # Create base workflow
    workflow = base_create_agent(working_dir, mcp_servers)

    # Or build custom workflow from scratch
    from google.adk.agents import SequentialAgent

    custom_workflow = SequentialAgent(
        name="custom_workflow",
        description="Workflow with custom agents",
        sub_agents=[
            # Your custom agent composition
        ]
    )

    return custom_workflow
```

## 自定义工具

工具为智能体提供能力。你可以通过简单的 Python 函数创建自定义工具。

### 创建自定义工具

自定义工具本质是遵循固定签名模式的普通 Python 函数：

```python
from functools import partial
from pathlib import Path

def custom_data_analysis(
    query: str,
    working_dir: str,
) -> str:
    """
    Perform custom data analysis.

    Parameters
    ----------
    query : str
        Analysis query
    working_dir : str
        Working directory for security validation

    Returns
    -------
    str
        Analysis results or error message
    """
    try:
        # Your custom logic here
        # Validate paths against working_dir for security
        work_path = Path(working_dir).resolve()

        # Perform analysis
        result = f"Analysis for: {query}"
        return result
    except Exception as e:
        return f"Error: {e}"

def fetch_custom_api(endpoint: str, timeout: int = 30) -> str:
    """
    Fetch data from a custom API.

    Parameters
    ----------
    endpoint : str
        API endpoint path
    timeout : int, optional
        Request timeout in seconds

    Returns
    -------
    str
        API response or error message
    """
    import requests

    try:
        base_url = "https://api.example.com"
        response = requests.get(f"{base_url}/{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error: {e}"
```

### 将自定义工具添加给智能体

修改智能体创建逻辑，加入你的工具：

```python
from functools import partial
from agentic_data_scientist.agents.adk.agent import create_agent
from agentic_data_scientist.tools import (
    read_file,
    list_directory,
    fetch_url,
)

def create_agent_with_custom_tools(working_dir: str):
    """Create agent with custom tools."""

    # Import your custom tools
    from my_tools import custom_data_analysis, fetch_custom_api

    # Create tools list with working_dir bound
    tools = [
        # Standard file tools
        partial(read_file, working_dir=working_dir),
        partial(list_directory, working_dir=working_dir),

        # Custom tools
        partial(custom_data_analysis, working_dir=working_dir),

        # Web tools (no working_dir needed)
        fetch_url,
        fetch_custom_api,
    ]

    # Create agent with custom tools
    # Note: You'll need to modify agent.py to accept tools parameter
    # or directly instantiate agents with your tools list
    return tools
```

### 工具设计最佳实践

1. **返回字符串结果**：为兼容 ADK，工具应返回字符串结果或错误信息
2. **包含安全参数**：文件操作工具应包含 `working_dir` 参数
3. **优雅处理错误**：以字符串返回错误，不直接抛异常
4. **使用类型注解**：为参数和返回值添加类型注解
5. **编写 Docstring**：使用 NumPy 风格文档字符串
6. **保持单一职责**：每个工具只做好一件事

### 示例：自定义数据库工具

```python
from functools import partial
import sqlite3
from pathlib import Path

def query_database(
    query: str,
    working_dir: str,
    db_name: str = "data.db",
) -> str:
    """
    Execute a read-only SQL query on a database.

    Parameters
    ----------
    query : str
        SQL query (SELECT only)
    working_dir : str
        Working directory containing the database
    db_name : str, optional
        Database filename, default "data.db"

    Returns
    -------
    str
        Query results as formatted string
    """
    try:
        # Security: Validate database is in working_dir
        work_path = Path(working_dir).resolve()
        db_path = (work_path / db_name).resolve()

        if not db_path.is_relative_to(work_path):
            return f"Error: Database must be in working directory"

        # Security: Only allow SELECT queries
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries allowed"

        # Execute query
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(query)
        results = cursor.fetchall()
        conn.close()

        # Format results
        if not results:
            return "No results found"

        # Simple formatting
        return "\n".join(str(row) for row in results)

    except Exception as e:
        return f"Error executing query: {e}"

# Usage in agent configuration
tools = [
    partial(query_database, working_dir="/path/to/session"),
]
```

## 自定义事件处理器

### 处理流式事件

创建自定义处理器来处理工作流事件：

```python
async def custom_event_processor(ds, query):
    """Custom event processing with metrics."""

    metrics = {
        'plan_iterations': 0,
        'implementation_iterations': 0,
        'stages_completed': 0,
        'tools_used': set(),
    }

    async for event in await ds.run_async(query, stream=True):
        event_type = event.get('type')
        author = event.get('author', '')

        # Track metrics
        if 'plan_maker' in author:
            metrics['plan_iterations'] += 1
        elif 'coding_agent' in author:
            metrics['implementation_iterations'] += 1
        elif 'Stage' in event.get('content', ''):
            metrics['stages_completed'] += 1

        if event_type == 'function_call':
            metrics['tools_used'].add(event['name'])

        # Custom handling
        if event_type == 'message':
            # Filter or transform messages
            content = event['content']
            if 'ERROR' in content:
                logger.error(f"Error in {author}: {content}")

        elif event_type == 'completed':
            # Log metrics
            logger.info(f"Workflow Metrics: {metrics}")
            print(f"\nWorkflow completed with:")
            print(f"  - {metrics['plan_iterations']} planning iterations")
            print(f"  - {metrics['implementation_iterations']} implementation iterations")
            print(f"  - {metrics['stages_completed']} stages completed")
            print(f"  - {len(metrics['tools_used'])} unique tools used")
```

### 自定义事件转换

在处理前先转换事件：

```python
from agentic_data_scientist.core.events import event_to_dict

def transform_event(event):
    """Add custom fields to events."""
    event_dict = event_to_dict(event)

    # Add custom metadata
    event_dict['processed_at'] = time.time()
    event_dict['workflow_phase'] = detect_phase(event_dict['author'])

    # Enhance with additional info
    if event_dict['type'] == 'message':
        event_dict['word_count'] = len(event_dict['content'].split())

    return event_dict

def detect_phase(author):
    """Detect which workflow phase an event belongs to."""
    if 'plan_maker' in author or 'plan_reviewer' in author:
        return 'planning'
    elif 'stage_orchestrator' in author or 'coding_agent' in author:
        return 'execution'
    elif 'summary' in author:
        return 'summary'
    return 'unknown'
```

## 集成示例

### 与 FastAPI 集成

```python
from fastapi import FastAPI, WebSocket, HTTPException
from agentic_data_scientist import DataScientist
import asyncio
import json

app = FastAPI()

@app.websocket("/ws/analyze")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis."""
    await websocket.accept()

    try:
        # Receive request
        data = await websocket.receive_json()
        query = data.get('query')
        files = data.get('files', [])

        # Run workflow with streaming
        async with DataScientist() as ds:
            async for event in await ds.run_async(
                query,
                files=[(f['name'], f['content']) for f in files],
                stream=True
            ):
                # Send events to client
                await websocket.send_json(event)

    except Exception as e:
        await websocket.send_json({
            'type': 'error',
            'content': str(e)
        })
    finally:
        await websocket.close()

@app.post("/api/analyze")
async def analyze_endpoint(query: str, files: list = None):
    """REST endpoint for analysis."""
    async with DataScientist() as ds:
        result = await ds.run_async(query, files=files)

        if result.status == "error":
            raise HTTPException(status_code=500, detail=result.error)

        return {
            'response': result.response,
            'files_created': result.files_created,
            'duration': result.duration
        }
```

### 与 Jupyter Notebook 集成

```python
from agentic_data_scientist import DataScientist
from IPython.display import display, Markdown, HTML
import asyncio

async def notebook_analysis(query, files=None):
    """Run analysis in Jupyter with rich formatting."""
    display(Markdown(f"## Analysis Request\n\n{query}"))

    async with DataScientist() as ds:
        display(Markdown("### Workflow Progress"))

        current_phase = None
        async for event in await ds.run_async(
            query,
            files=files,
            stream=True
        ):
            author = event.get('author', '')

            # Track phase changes
            if 'plan_maker' in author and current_phase != 'Planning':
                current_phase = 'Planning'
                display(Markdown(f"**Phase: {current_phase}**"))
            elif 'coding_agent' in author and current_phase != 'Execution':
                current_phase = 'Execution'
                display(Markdown(f"**Phase: {current_phase}**"))
            elif 'summary' in author and current_phase != 'Summary':
                current_phase = 'Summary'
                display(Markdown(f"**Phase: {current_phase}**"))

            if event['type'] == 'message':
                content = event['content']
                # Display formatted messages
                if len(content) < 200:
                    display(Markdown(f"*{author}*: {content}"))

            elif event['type'] == 'completed':
                files = event['files_created']
                display(Markdown(f"### Results\n\n**Files Created:**"))
                for f in files:
                    display(Markdown(f"- `{f}`"))

# Usage in notebook
await notebook_analysis("Analyze customer churn", files=[('data.csv', data)])
```

### 自定义会话管理

```python
from agentic_data_scientist import DataScientist
import json
from pathlib import Path

class PersistentDataScientist:
    """DataScientist with session persistence."""

    def __init__(self, session_dir="./sessions"):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.ds = None
        self.session_id = None

    async def start_session(self, session_id=None):
        """Start or resume a session."""
        self.ds = DataScientist()
        await self.ds.__aenter__()

        if session_id:
            # Resume existing session
            self.session_id = session_id
            context = self.load_context(session_id)
        else:
            # New session
            self.session_id = self.ds.session_id
            context = {}

        return context

    def load_context(self, session_id):
        """Load session context."""
        context_file = self.session_dir / f"{session_id}.json"
        if context_file.exists():
            with open(context_file) as f:
                return json.load(f)
        return {}

    def save_context(self, context):
        """Save session context."""
        context_file = self.session_dir / f"{self.session_id}.json"
        with open(context_file, 'w') as f:
            json.dump(context, f, indent=2)

    async def run(self, query, context=None):
        """Run query with persistent context."""
        if context is None:
            context = self.load_context(self.session_id)

        result = await self.ds.run_async(query, context=context)
        self.save_context(context)

        return result

    async def close(self):
        """Close session."""
        if self.ds:
            await self.ds.__aexit__(None, None, None)

# Usage
pds = PersistentDataScientist()
context = await pds.start_session()

# Run queries
result1 = await pds.run("Analyze this dataset", context)
result2 = await pds.run("What are the key trends?", context)

await pds.close()
```

## 环境配置

扩展系统时，请留意以下环境变量：

**核心必需：**
- `ANTHROPIC_API_KEY`：Claude Code 编码智能体

**按路由配置启用时必需：**
- `OPENAI_API_KEY`、`GOOGLE_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`：仅当对应配置在 `configs/llm_routing.yaml` 中启用时需要

**可选：**
- `DEFAULT_MODEL`：规划/评审模型（默认：`gemini-3.1-pro-preview`）
- `REVIEW_MODEL`：评审模型（默认：与 DEFAULT_MODEL 相同）
- `CODING_MODEL`：编码模型（默认：`claude-sonnet-4-6`）
- `OPENROUTER_API_KEY`：仅用于 OpenRouter 路由调用（可选/旧版兼容）

## 最佳实践

1. **充分测试自定义提示词**：用多样查询验证提示词改动
2. **使用类型注解**：自定义代码统一补齐类型注解
3. **处理错误**：在自定义智能体中实现完整错误处理
4. **记录定制行为**：用 docstring 说明自定义逻辑
5. **保持提示词模块化**：复杂提示词拆成可复用组件
6. **版本管理提示词**：像管理代码一样追踪提示词变更
7. **监控智能体行为**：开发时记录并分析输出
8. **模型配置外置**：通过环境变量配置模型，避免硬编码

## 另请参阅

更多入门、API、CLI、工具配置与技术架构文档，请查看 `docs/` 目录。
