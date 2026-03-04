# 工具配置

本指南说明 Agentic Data Scientist 智能体可用的工具，以及这些工具的工作方式。

## 概览

ADK 多智能体工作流使用本地 Python 函数工具，为智能体提供文件系统与网页访问能力。不同智能体拥有不同的工具集：

- **规划与评审智能体**（ADK）：使用本地文件读取工具和网页抓取工具来读取文件、获取网页内容
- **编码智能体**（Claude Code）：可使用 Context7 MCP（可选）获取库文档，并可使用 380+ 科学技能（自动加载）处理专业任务

## 本地工具

所有 ADK 智能体都可以访问一组经过筛选的本地 Python 工具，这些工具内置了安全控制。

### 安全模型

所有文件操作均为**只读**，并强制执行**工作目录沙箱**：

- 智能体只能访问分配给它的 `working_dir` 内的文件
- 若路径穿越（例如 `../`）试图逃逸工作目录，会被阻止
- 拒绝访问工作目录之外的绝对路径
- 不提供写入、删除、编辑操作

这可确保智能体无法修改文件，也无法访问敏感系统数据。

### 文件操作工具

#### read_file

读取文本文件内容，并可选限制头部/尾部行数。

**参数：**
- `path`（str）：文件路径（相对于 working_dir，或位于 working_dir 内的绝对路径）
- `head`（int，可选）：仅读取前 N 行
- `tail`（int，可选）：仅读取后 N 行

**用法：**
```python
# 智能体会自动使用该工具
result = read_file("data/results.csv")
```

#### read_media_file

读取二进制/媒体文件（图像、音频等），并以 base64 编码返回。

**参数：**
- `path`（str）：媒体文件路径

**返回：**
- JSON 字符串，包含 `data`（base64 编码）与 `mimeType` 字段

**用法：**
```python
# 智能体可读取并处理图像
result = read_media_file("plot.png")
```

#### list_directory

列出目录内容，可选显示文件大小并排序。

**参数：**
- `path`（str，可选）：目录路径，默认 `"."`
- `show_sizes`（bool，可选）：是否显示文件大小，默认 `False`
- `sort_by`（str，可选）：排序方式：`"name"` 或 `"size"`，默认 `"name"`

**用法：**
```python
# 列出当前目录文件
result = list_directory(".", show_sizes=True)
```

#### directory_tree

递归生成目录树视图，可配置排除模式。

**参数：**
- `path`（str，可选）：目录路径，默认 `"."`
- `exclude_patterns`（list[str]，可选）：排除模式（例如 `["*.pyc", "__pycache__"]`）

**返回：**
- 表示目录树结构的 JSON 字符串

**用法：**
```python
# 获取完整目录树
result = directory_tree(".", exclude_patterns=["*.pyc"])
```

#### search_files

按 glob 模式搜索文件。

**参数：**
- `pattern`（str）：glob 模式（例如 `"*.py"`、`"test_*.txt"`）
- `path`（str，可选）：搜索目录，默认 `"."`
- `exclude_patterns`（list[str]，可选）：结果排除模式

**用法：**
```python
# 查找所有 Python 文件
result = search_files("*.py", path="src/")
```

#### get_file_info

获取文件的详细元数据。

**参数：**
- `path`（str）：文件路径

**返回：**
- 包含大小、类型、修改时间、访问时间、权限的格式化字符串

**用法：**
```python
# 获取文件元数据
result = get_file_info("data.csv")
```

### Web 操作工具

#### fetch_url

通过 HTTP GET 获取 URL 内容。

**参数：**
- `url`（str）：要抓取的 URL
- `timeout`（int，可选）：请求超时（秒），默认 30
- `user_agent`（str，可选）：自定义 User-Agent 请求头

**返回：**
- 响应文本内容或错误信息

**用法：**
```python
# 智能体可抓取网页内容
result = fetch_url("https://api.example.com/data")
```

## 在工作流中的使用

### 基础文件访问

```python
from agentic_data_scientist import DataScientist

# 上传的文件存储在会话工作目录中
# 规划与评审智能体可通过本地工具读取这些文件
with DataScientist() as ds:
    result = ds.run(
        "Analyze trends in this data",
        files=[("data.csv", open("data.csv", "rb").read())]
    )
```

### 抓取网页内容

```python
# 智能体可在分析过程中自动抓取网页内容
with DataScientist() as ds:
    result = ds.run("Summarize the latest research on transformer architectures from ArXiv")
```

### 处理多个文件

```python
with DataScientist() as ds:
    result = ds.run(
        "Compare the metrics across all JSON files",
        files=[
            ("metrics_v1.json", open("metrics_v1.json", "rb").read()),
            ("metrics_v2.json", open("metrics_v2.json", "rb").read()),
            ("metrics_v3.json", open("metrics_v3.json", "rb").read()),
        ]
    )
```

## Claude Code 智能体工具

Claude Code 智能体使用另一套工具，配置位于 `.claude/settings.json`。

### Context7 MCP（可选）

为各类库和框架提供文档与上下文检索能力。

**状态：** 可选。编码智能体即使不启用 Context7 也可工作，大多数文档需求可由 Skills 满足。

**可用工具：**
- `resolve-library-id`：将包名解析为 Context7 兼容的库 ID
- `get-library-docs`：获取某个库的最新文档

**配置方式：**

若要启用 Context7，可在项目根目录的 `.claude/settings.json` 中配置：

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"],
      "env": {
        "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"
      }
    }
  }
}
```

**环境变量：**

```bash
# 写入 .env（可选）
CONTEXT7_API_KEY=your-api-key-here
```

### Claude Scientific Skills（自动加载）

编码智能体可自动访问来自 [claude-scientific-skills](https://github.com/K-Dense-AI/claude-scientific-skills) 的 380+ 科学技能。

**状态：** 自动，无需配置。编码智能体启动时会自动加载技能。

**可用技能类别：**

**科学数据库：**
- UniProt、PubChem、PDB、KEGG、PubMed
- COSMIC、ClinVar、GEO、ENA、Ensembl
- STRING、Reactome、DrugBank、ChEMBL
- 以及更多……

**科学软件包：**
- BioPython、RDKit、PyDESeq2、scanpy、anndata
- MDAnalysis、scikit-learn、PyTorch、TensorFlow
- statsmodels、matplotlib、seaborn、polars
- 以及更多……

**Skills 工作机制：**

1. **自动加载**：编码智能体启动时，Skills 会自动克隆到 `.claude/skills/`
2. **智能体发现**：编码智能体在运行时发现可用 Skills
3. **自主使用**：智能体根据任务自动决定使用哪些 Skills
4. **零配置**：无需环境变量或额外初始化

## 工具实现细节

### 添加自定义工具

你可以通过定义 Python 函数并将其加入智能体配置来添加自定义工具：

```python
from functools import partial
from agentic_data_scientist.agents.adk import create_agent

# 定义你的自定义工具
def custom_analysis_tool(data: str, working_dir: str) -> str:
    """
    带 working_dir 参数的自定义分析工具。

    Parameters
    ----------
    data : str
        输入数据
    working_dir : str
        用于安全校验的工作目录

    Returns
    -------
    str
        分析结果
    """
    # 在此实现你的逻辑
    return f"Analyzed: {data}"

# 创建带自定义工具的智能体
agent = create_agent(working_dir="/tmp/session")

# 若要添加自定义工具，可修改 agent.py 里的 tools 列表
# 追加绑定了 working_dir 的 partial 函数：
# tools.append(partial(custom_analysis_tool, working_dir=str(working_dir)))
```

### 工具函数签名

所有文件操作工具遵循如下模式：

```python
def tool_function(
    # 工具特有参数
    path: str,
    # 其他参数...
    # 安全参数（文件操作始终需要）
    working_dir: str,
) -> str:
    """
    工具说明。

    Returns
    -------
    str
        结果或错误信息
    """
    # 实现
    pass
```

在为智能体配置工具时，`working_dir` 参数会通过 `functools.partial` 自动绑定。

## 环境变量

**框架必需：**
- `OPENROUTER_API_KEY`：规划/评审智能体必需
- `ANTHROPIC_API_KEY`：编码智能体必需

**工具可选：**
- `CONTEXT7_API_KEY`：仅在启用 Context7 MCP 时需要

**以下功能不需要环境变量：**
- 本地文件操作工具（开箱即用）
- 网页抓取工具（开箱即用）
- Claude Scientific Skills（自动加载）

## 安全注意事项

### 文件访问安全

- **只读**：不提供写入、删除、编辑操作
- **沙箱化**：所有路径都会对照 working_dir 校验
- **防路径穿越**：阻止逃逸 working_dir 的尝试
- **符号链接解析**：符号链接会被解析并校验

### 网页抓取安全

- **协议限制**：仅允许 HTTP 和 HTTPS
- **超时保护**：默认 30 秒超时，避免请求挂起
- **错误处理**：网络错误返回安全错误信息

### 最佳实践

1. **最小权限**：智能体仅拥有各自工作目录的读取权限
2. **会话隔离**：每个会话拥有独立工作目录
3. **职责分离**：规划智能体不能修改实现文件
4. **审计轨迹**：所有工具调用都会被记录并可审查

## 故障排查

### “Access denied: outside working directory”

该错误表示工具尝试访问工作目录之外的文件。请确保路径：
- 相对于工作目录，或
- 是工作目录内部的绝对路径

### “File does not exist”

指定文件不存在。请检查：
- 文件路径拼写与大小写
- 文件是否已正确上传到会话
- 目录结构是否正确

### “Request timed out”

网页抓取请求超时。可尝试：
- 提高超时参数：`fetch_url(url, timeout=60)`
- 检查 URL 是否可访问
- 验证网络连接

## 总结

**内置工具（规划/评审智能体）：**
- 文件操作：只读，并沙箱到工作目录
- Web 操作：带超时保护的 HTTP 抓取
- 无需额外配置

**Claude Code 工具（编码智能体）：**
- Context7 MCP：可选，用于库文档
- Scientific Skills：自动加载，380+ 科学计算技能
- Skills 无需配置

**环境变量：**
- 必需：`OPENROUTER_API_KEY`、`ANTHROPIC_API_KEY`
- 可选：`CONTEXT7_API_KEY`（仅使用 Context7 时）
