# API 参考

Agentic Data Scientist 的完整 API 参考。

## 核心 API

### `DataScientist`

与 Agentic Data Scientist 多智能体工作流交互的主类。

```python
from agentic_data_scientist import DataScientist

ds = DataScientist(
    agent_type="adk",           # "adk"（推荐）或 "claude_code"（直接模式）
    mcp_servers=None,           # 可选：MCP 服务器列表
)
```

#### 参数

- **agent_type**（str，默认 `"adk"`）：使用的智能体类型
  - `"adk"`：**（推荐）** 完整多智能体工作流，含规划、验证和自适应执行
  - `"claude_code"`：直接模式，绕过工作流，适用于简单脚本任务

- **mcp_servers**（list，可选）：启用的 MCP 服务器列表（当前未使用，见 `tools_configuration.md`）

**说明**：多智能体 ADK 工作流（`agent_type="adk"`）是主要模式，适用于大多数场景。直接模式仅建议用于不需要规划与验证的简单任务。

**模型配置**：模型通过环境变量配置，支持多提供商路由（参见 `configs/llm_routing.yaml`）：
- ADK 智能体：`DEFAULT_MODEL`（默认：`gemini-3.1-pro-preview`）
- 编码智能体：`CODING_MODEL`（默认：`claude-sonnet-4-6`）
- 路由配置支持按角色指定主/备模型和提供商

#### 属性

- **session_id**（str）：唯一会话 ID
- **working_dir**（Path）：会话临时工作目录
- **config**（SessionConfig）：会话配置

#### 方法

##### `run(message, files=None, **kwargs) -> Result`

同步执行查询。

**参数：**
- **message**（str）：用户查询或指令
- **files**（list[tuple]，可选）：`(filename, content)` 元组列表
- **kwargs**：其他附加参数

**返回：**
- `Result` 对象，包含 `response`、`files_created`、`duration` 等信息

**示例：**
```python
with DataScientist() as ds:
    result = ds.run("Analyze trends in this data", files=[("data.csv", data)])
    print(result.response)
    print(f"Status: {result.status}")  # "completed" or "error"
```

##### `run_async(message, files=None, stream=False, context=None) -> Union[Result, AsyncGenerator]`

异步执行查询。

**参数：**
- **message**（str）：用户查询或指令
- **files**（list[tuple]，可选）：`(filename, content)` 元组列表
- **stream**（bool，默认 `False`）：若为 `True`，返回用于流式事件的异步生成器
- **context**（dict，可选）：多轮对话的上下文

**返回：**
- 若 `stream=False`：返回 `Result`
- 若 `stream=True`：返回产生事件字典的 `AsyncGenerator`

**示例（非流式）：**
```python
import asyncio

async def main():
    async with DataScientist() as ds:
        result = await ds.run_async("Explain gradient boosting")
        print(result.response)

asyncio.run(main())
```

**示例（流式）：**
```python
async def stream_example():
    async with DataScientist() as ds:
        async for event in await ds.run_async(
            "Analyze this dataset",
            files=[("data.csv", data)],
            stream=True
        ):
            if event['type'] == 'message':
                print(f"[{event['author']}] {event['content']}")

asyncio.run(stream_example())
```

##### `save_files(files) -> List[FileInfo]`

将文件保存到工作目录。

**参数：**
- **files**（list[tuple]）：`(filename, content)` 元组列表

**返回：**
- `FileInfo` 对象列表（包含名称、路径、大小）

##### `prepare_prompt(message, file_info=None) -> str`

构造提示词，可附加文件信息。

**参数：**
- **message**（str）：用户消息
- **file_info**（list[FileInfo]，可选）：已上传文件列表

**返回：**
- 完整提示词字符串

##### `cleanup()`

清理临时工作目录。

## 数据类

### `SessionConfig`

智能体会话配置。

```python
from agentic_data_scientist.core.api import SessionConfig

config = SessionConfig(
    agent_type="adk",
    mcp_servers=["filesystem", "fetch"],
    max_llm_calls=1024,
    session_id=None,
    working_dir=None,
)
```

#### 属性

- **agent_type**（str）：`"adk"` 或 `"claude_code"`
- **mcp_servers**（list，可选）：MCP 服务器列表（当前未使用）
- **max_llm_calls**（int）：每个会话最大 LLM 调用次数
- **session_id**（str，可选）：自定义会话 ID
- **working_dir**（str，可选）：自定义工作目录
- **auto_cleanup**（bool）：完成后是否自动清理工作目录

**说明**：模型通过环境变量（`OPENROUTER_API_KEY`、`DEFAULT_MODEL`、`CODING_MODEL`）配置，不在 `SessionConfig` 中设置。

### `Result`

工作流运行结果。

```python
result = ds.run("Query")

# 访问结果属性
print(result.session_id)       # 会话 ID
print(result.status)           # "completed" or "error"
print(result.response)         # 智能体返回文本
print(result.error)            # 错误信息（status="error" 时）
print(result.files_created)    # 创建文件列表
print(result.duration)         # 执行时长（秒）
print(result.events_count)     # 处理事件数
```

### `FileInfo`

上传文件信息。

```python
file_info = FileInfo(
    name="data.csv",
    path="/path/to/data.csv",
    size_kb=10.5
)
```

## 事件系统

启用流式模式（`stream=True`）时，工作流会在执行过程中持续发出事件。

### 工作流事件顺序

对于 ADK 多智能体工作流，事件大致顺序如下：

```
Planning Phase:
  plan_maker_agent → plan_reviewer_agent → plan_review_confirmation_agent →
  high_level_plan_parser

Execution Phase (repeated for each stage):
  stage_orchestrator → coding_agent → review_agent →
  implementation_review_confirmation_agent → success_criteria_checker →
  stage_reflector

Summary Phase:
  summary_agent
```

### 事件类型

#### MessageEvent

智能体的普通文本输出。

```python
{
    'type': 'message',
    'content': 'Text content',
    'author': 'plan_maker_agent',  # 事件来源智能体
    'timestamp': '12:34:56.789',
    'is_thought': False,            # 内部推理 / 对外输出
    'is_partial': False,            # 流式分片 / 完整内容
    'event_number': 1
}
```

**工作流中的常见 Author：**
- `plan_maker_agent`：生成分析计划
- `plan_reviewer_agent`：评审计划
- `plan_review_confirmation_agent`：判断计划是否通过
- `high_level_plan_parser`：结构化计划
- `stage_orchestrator`：管理阶段执行
- `coding_agent`：实现阶段内容
- `review_agent`：评审实现
- `implementation_review_confirmation_agent`：判断实现是否通过
- `success_criteria_checker`：更新进展
- `stage_reflector`：调整剩余阶段
- `summary_agent`：生成最终报告

#### FunctionCallEvent

智能体正在调用工具。

```python
{
    'type': 'function_call',
    'name': 'read_file',
    'arguments': {'path': 'data.csv'},
    'author': 'review_agent',
    'timestamp': '12:34:56.789',
    'event_number': 2
}
```

#### FunctionResponseEvent

工具返回结果。

```python
{
    'type': 'function_response',
    'name': 'read_file',
    'response': {'content': '...file contents...'},
    'author': 'review_agent',
    'timestamp': '12:34:56.789',
    'event_number': 3
}
```

#### UsageEvent

Token 用量信息。

```python
{
    'type': 'usage',
    'usage': {
        'total_input_tokens': 1500,
        'cached_input_tokens': 200,
        'output_tokens': 500
    },
    'timestamp': '12:34:56.789'
}
```

#### ErrorEvent

执行过程中发生错误。

```python
{
    'type': 'error',
    'content': 'Error message describing what went wrong',
    'timestamp': '12:34:56.789'
}
```

#### CompletedEvent

工作流成功完成。

```python
{
    'type': 'completed',
    'session_id': 'session_123',
    'duration': 45.2,
    'total_events': 150,
    'files_created': ['results.csv', 'plot.png', 'summary.md'],
    'files_count': 3,
    'timestamp': '12:34:56.789'
}
```

### 工作流特定事件

#### 阶段切换事件

当编排器在阶段间切换时：

```python
{
    'type': 'message',
    'author': 'stage_orchestrator',
    'content': '### Stage 2: Data Preprocessing\n\nBeginning implementation...',
    # ...
}
```

#### 成功标准更新事件

Criteria Checker 运行后：

```python
{
    'type': 'message',
    'author': 'success_criteria_checker',
    'content': '{...JSON with criteria updates...}',
    # checker 输出结构化 JSON
}
```

#### 规划循环事件

在计划迭代优化期间：

```python
# 生成计划
{'author': 'plan_maker_agent', 'content': '### Analysis Stages:\n1. ...'}

# 评审反馈
{'author': 'plan_reviewer_agent', 'content': 'This plan looks good...'}

# 决策
{'author': 'plan_review_confirmation_agent', 'content': '{"exit": true, "reason": "..."}'}
```

### 示例：处理事件

```python
async def process_workflow_events(ds, query):
    """Track workflow progress through events."""

    current_phase = None
    current_stage = None

    async for event in await ds.run_async(query, stream=True):
        event_type = event.get('type')
        author = event.get('author', '')

        # Track workflow phase
        if 'plan_maker' in author:
            if current_phase != 'planning':
                current_phase = 'planning'
                print("\n=== PLANNING PHASE ===")
        elif 'stage_orchestrator' in author:
            if current_phase != 'execution':
                current_phase = 'execution'
                print("\n=== EXECUTION PHASE ===")
        elif 'summary' in author:
            if current_phase != 'summary':
                current_phase = 'summary'
                print("\n=== SUMMARY PHASE ===")

        # Handle different event types
        if event_type == 'message':
            content = event['content']

            # Track stage transitions
            if 'Stage' in content and 'Beginning implementation' in content:
                print(f"\n-> Starting new stage")

            print(f"[{author}] {content[:100]}...")

        elif event_type == 'function_call':
            tool_name = event['name']
            print(f"  -> Using tool: {tool_name}")

        elif event_type == 'usage':
            usage = event['usage']
            print(f"  Tokens: {usage.get('total_input_tokens', 0)} in, "
                  f"{usage.get('output_tokens', 0)} out")

        elif event_type == 'error':
            error_msg = event['content']
            print(f"  Error: {error_msg}")

        elif event_type == 'completed':
            duration = event['duration']
            files = event['files_created']
            print(f"\nCompleted in {duration:.1f}s")
            print(f"Created {len(files)} files: {', '.join(files)}")
```

## CLI 用法

完整 CLI 文档（所有参数、工作目录行为、更多示例）请查看 `cli_reference.md`。

## 环境变量

### 核心必需

- **ANTHROPIC_API_KEY**：Claude（编码智能体）所需 Anthropic API key

### 按路由配置启用时必需

- **OPENAI_API_KEY**、**GOOGLE_API_KEY**、**DASHSCOPE_API_KEY**、**DEEPSEEK_API_KEY**：仅当对应配置文件在 `configs/llm_routing.yaml` 中启用时需要

### 可选

- **DEFAULT_MODEL**：规划与评审模型（默认：`gemini-3.1-pro-preview`）
- **REVIEW_MODEL**：评审模型（默认：与 DEFAULT_MODEL 相同）
- **CODING_MODEL**：编码模型（默认：`claude-sonnet-4-6`）
- **OPENROUTER_API_KEY**：仅用于 OpenRouter 路由调用（可选/旧版兼容）
- **OPENROUTER_API_BASE**：OpenRouter API 地址（默认：`https://openrouter.ai/api/v1`）

## 错误处理

```python
from agentic_data_scientist import DataScientist

with DataScientist() as ds:
    result = ds.run("Query")

    if result.status == "error":
        print(f"Error occurred: {result.error}")
        # 在这里处理错误
    else:
        print(f"Success: {result.response}")
        print(f"Created files: {result.files_created}")
```

## 最佳实践

1. **使用上下文管理器**确保资源清理：
   ```python
   with DataScientist() as ds:
       # your code
   ```

2. **优雅处理错误**：
   ```python
   result = ds.run("Query")
   if result.status != "error":
       # process result
   ```

3. **长任务使用流式模式**监控进度：
   ```python
   async for event in await ds.run_async("Task", stream=True):
       # 实时处理事件
   ```

4. **多轮对话传入 context**：
   ```python
   context = {}
   result1 = await ds.run_async("First query", context=context)
   result2 = await ds.run_async("Follow-up", context=context)
   ```

5. **复杂任务使用 ADK 工作流**：
   ```python
   # 推荐用于大多数场景
   with DataScientist(agent_type="adk") as ds:
       result = ds.run("Complex analysis task")
   ```

6. **直接模式仅用于简单任务**：
   ```python
   # 仅用于简单脚本
   with DataScientist(agent_type="claude_code") as ds:
       result = ds.run("Write a simple function")
   ```

## 另请参阅

更多入门、CLI、定制化和技术架构文档，请查看 `docs/` 目录。
