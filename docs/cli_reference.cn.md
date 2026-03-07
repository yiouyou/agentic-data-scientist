# CLI 参考

Agentic Data Scientist 的完整命令行接口参考。

## 基础用法

```bash
agentic-data-scientist [OPTIONS] QUERY
```

CLI 提供简洁接口，可选择完整多智能体工作流或直接编码模式来执行数据科学任务。

## 必需选项

### `--mode`（必需）

每次查询都必须指定执行模式，以确保你明确了解复杂度与 API 成本。

**可选值：**
- `orchestrated`：完整多智能体工作流，包含规划、验证和自适应执行（复杂分析推荐）
- `simple`：直接编码模式，无规划开销（适合快速脚本与简单任务）

**示例：**
```bash
# 复杂分析：完整工作流
agentic-data-scientist "Perform differential expression analysis" --mode orchestrated --files data.csv

# 快速脚本任务
agentic-data-scientist "Write a Python function to parse JSON" --mode simple
```

## 可选参数

### `--files, -f`

上传文件或目录并纳入分析。可重复指定以上传多个文件。

**行为：**
- 文件会上传到工作目录
- 目录会递归上传
- 上传的所有文件均可被智能体访问

**示例：**
```bash
# 单文件
agentic-data-scientist "Analyze this data" --mode orchestrated --files data.csv

# 多文件
agentic-data-scientist "Compare datasets" --mode orchestrated -f data1.csv -f data2.csv

# 目录上传（递归）
agentic-data-scientist "Analyze all files" --mode orchestrated --files ./data_folder/
```

### `--working-dir, -w`

为会话指定自定义工作目录。

**默认值：** 当前目录下的 `./agentic_output/`

**行为：**
- 文件保存到该目录
- 完成后保留目录（除非使用 `--temp-dir`）
- 目录不存在会自动创建

**示例：**
```bash
# 自定义目录
agentic-data-scientist "Analyze data" --mode orchestrated --files data.csv --working-dir ./my_analysis

# 绝对路径
agentic-data-scientist "Process data" --mode orchestrated --files data.csv -w /tmp/analysis_2024
```

### `--temp-dir`

在 `/tmp` 下使用临时目录，任务结束后自动清理。

**行为：**
- 创建唯一临时目录
- 会话结束后自动删除
- 若同时指定 `--working-dir`，`--temp-dir` 优先
- 适用于无需保留文件的快速分析

**示例：**
```bash
# 临时分析
agentic-data-scientist "Quick exploration" --mode simple --files data.csv --temp-dir

# 问答（无需保留文件）
agentic-data-scientist "Explain gradient boosting" --mode simple --temp-dir
```

### `--keep-files`

显式要求任务完成后保留工作目录。

**默认行为：** 使用 `--working-dir` 或默认目录时，文件默认会保留。

**注意：** 使用 `--temp-dir` 时该参数无效（临时目录总会被清理）。

**示例：**
```bash
# 显式保留文件
agentic-data-scientist "Generate report" --mode orchestrated --files data.csv --keep-files
```

### `--log-file`

指定日志文件路径。

**默认值：** 工作目录下 `.agentic_ds.log`

**示例：**
```bash
# 自定义日志位置
agentic-data-scientist "Analyze data" --mode orchestrated --files data.csv --log-file ./analysis.log

# 绝对路径
agentic-data-scientist "Process data" --mode simple --log-file /var/log/agentic_analysis.log
```

### `--verbose, -v`

开启详细日志，用于调试。

**行为：**
- 显示详细执行日志
- 展示智能体内部通信
- 便于问题定位

**示例：**
```bash
# 详细输出
agentic-data-scientist "Debug issue" --mode simple --files data.csv --verbose

# 与其他参数组合
agentic-data-scientist "Complex analysis" --mode orchestrated --files data.csv --verbose --log-file debug.log
```

## 工作目录行为

理解工作目录机制有助于管理分析输出文件。

### 默认行为

未指定目录参数时：
- 在当前目录创建 `./agentic_output/`
- 任务完成后保留所有文件
- 智能体可在此目录读写文件

```bash
agentic-data-scientist "Analyze data" --mode orchestrated --files data.csv
# Files saved to: ./agentic_output/
# Preserved: Yes
```

### 临时目录

使用 `--temp-dir` 时：
- 在 `/tmp` 下创建唯一目录
- 完成后自动删除
- 适合不需要留存文件的快速分析

```bash
agentic-data-scientist "Quick test" --mode simple --files data.csv --temp-dir
# Files saved to: /tmp/agentic_ds_XXXXXX/
# Preserved: No (auto-cleanup)
```

### 自定义目录

使用 `--working-dir` 时：
- 使用你指定的目录
- 完成后保留文件
- 目录不存在会自动创建

```bash
agentic-data-scientist "Project analysis" --mode orchestrated --files data.csv --working-dir ./my_project
# Files saved to: ./my_project/
# Preserved: Yes
```

## 执行模式

### Orchestrated 模式（推荐）

完整多智能体工作流，包含规划、验证和自适应执行。

**适用场景：**
- 复杂数据分析
- 多步骤工作流
- 需要验证和质量保障的任务
- 规划能显著提升效果的场景
- 执行中需求可能变化的任务

**执行流程：**
1. Plan Maker 生成完整计划
2. Plan Reviewer 评审计划
3. 对每个阶段：
   - Coding Agent 实现阶段内容
   - Review Agent 评审实现结果
   - Criteria Checker 跟踪进度
   - Stage Reflector 调整剩余工作
4. Summary Agent 生成最终报告

**示例：**
```bash
# 差异表达分析
agentic-data-scientist "Perform DEG analysis comparing treatment vs control" \
  --mode orchestrated \
  --files treatment_data.csv \
  --files control_data.csv

# 完整分析管线
agentic-data-scientist "Analyze customer churn, create predictive model, and generate report" \
  --mode orchestrated \
  --files customers.csv \
  --working-dir ./churn_analysis

# 多文件处理
agentic-data-scientist "Analyze all CSV files and create summary statistics" \
  --mode orchestrated \
  --files ./raw_data/
```

### Simple 模式

直接编码执行，不包含规划与验证循环。

**适用场景：**
- 快速脚本任务
- 简单代码生成
- 问答
- 快速原型
- 无需规划开销的任务

**执行特点：**
- 由 Claude Code 智能体直接执行
- 无规划阶段
- 无评审或验证循环
- 更快，但不提供质量保障

**示例：**
```bash
# 生成工具脚本
agentic-data-scientist "Write a Python script to merge CSV files by common column" \
  --mode simple

# 技术问答
agentic-data-scientist "Explain the difference between Random Forest and Gradient Boosting" \
  --mode simple

# 快速分析
agentic-data-scientist "Create a basic scatter plot from this data" \
  --mode simple \
  --files data.csv \
  --temp-dir
```

## 常见使用模式

### 多文件分析

```bash
# 对比多个数据集
agentic-data-scientist "Compare these datasets and identify trends" \
  --mode orchestrated \
  -f dataset1.csv \
  -f dataset2.csv \
  -f dataset3.csv
```

### 目录处理

```bash
# 处理整个目录
agentic-data-scientist "Analyze all JSON files and create consolidated report" \
  --mode orchestrated \
  --files ./data_directory/
```

### 临时分析

```bash
# 快速探索且不保留文件
agentic-data-scientist "Explore data distributions" \
  --mode simple \
  --files data.csv \
  --temp-dir
```

### 项目化分析

```bash
# 组织化项目结构
agentic-data-scientist "Complete statistical analysis with visualizations" \
  --mode orchestrated \
  --files raw_data.csv \
  --working-dir ./projects/analysis_2024 \
  --log-file ./projects/analysis_2024.log
```

### 调试与开发

```bash
# 调试时输出详细日志
agentic-data-scientist "Debug data processing issue" \
  --mode simple \
  --files problematic_data.csv \
  --verbose \
  --log-file debug.log
```

## 输入方式

### 命令行参数

最常见方式：把查询直接作为命令行参数。

```bash
agentic-data-scientist "Your query here" --mode orchestrated
```

### 标准输入管道（stdin）

从其他命令或文件通过管道传入：

```bash
# 来自 echo
echo "Analyze this dataset" | agentic-data-scientist --mode simple --files data.csv

# 来自文件
cat query.txt | agentic-data-scientist --mode orchestrated --files data.csv
```

## 退出码

- `0`：成功
- `1`：错误（参数非法、运行时错误等）

## 环境变量

CLI 会读取以下环境变量（可在 `.env` 或 shell 中设置）：

**核心必需：**
- `ANTHROPIC_API_KEY`：编码智能体所需 Anthropic API key

**按路由配置启用时必需：**
- `OPENAI_API_KEY`、`GOOGLE_API_KEY`、`DASHSCOPE_API_KEY`、`DEEPSEEK_API_KEY`：仅当对应配置在 `configs/llm_routing.yaml` 中启用时需要

**可选：**
- `DEFAULT_MODEL`：规划/评审模型（默认：`gemini-3.1-pro-preview`）
- `CODING_MODEL`：编码模型（默认：`claude-sonnet-4-6`）
- `OPENROUTER_API_KEY`：仅用于 OpenRouter 路由调用（可选/旧版兼容）

## 输出与日志

### 控制台输出

CLI 会显示：
- 智能体活动与进度
- 关键决策与里程碑
- 文件创建通知
- 完成摘要

### 日志文件

详细日志写入：
- 默认：工作目录下 `.agentic_ds.log`
- 自定义：`--log-file` 指定路径

日志内容包括：
- 完整智能体对话
- 工具调用与响应
- 错误信息与堆栈
- Token 使用统计

## 故障排查

### “Error: No query provided”

你没有提供查询。可选：
- 命令行参数方式：`agentic-data-scientist "query" --mode orchestrated`
- stdin 管道方式：`echo "query" | agentic-data-scientist --mode orchestrated`

### “Error: Missing option '--mode'”

`--mode` 是必需参数。必须指定其一：
- `--mode orchestrated`：完整多智能体工作流
- `--mode simple`：直接编码模式

### “File not found” 错误

请检查：
- 路径是否正确，文件是否存在
- 是否有读权限
- 路径中是否包含需要转义的特殊字符

### “API Key Not Found” 错误

请确保：
- 已在环境变量或 `.env` 设置 `OPENROUTER_API_KEY`
- 已在环境变量或 `.env` 设置 `ANTHROPIC_API_KEY`
- API key 有效且余额/额度足够

### 工作目录权限错误

请确保：
- 对工作目录有写权限
- 磁盘空间足够
- 父目录存在（或可自动创建）

### 内存不足错误

处理大任务时建议：
- 使用 `--temp-dir` 保证自动清理
- 分批处理文件
- 使用 simple 模式降低开销

## 最佳实践

1. **重要任务使用 Orchestrated 模式**
   - 规划可提前发现问题
   - 验证可保障质量
   - 生产任务通常值得额外 API 成本

2. **快速任务使用 Simple 模式**
   - 开发期快速迭代
   - 技术问答
   - 简单脚本生成

3. **组织好工作目录**
   - 用 `--working-dir` 管理项目输出
   - 用 `--temp-dir` 处理临时探索
   - 相关分析放在独立目录

4. **按需开启详细日志**
   - 调试时使用 `--verbose`
   - 使用 `--log-file` 持久保存日志
   - 通过日志复盘智能体行为

5. **管理文件生命周期**
   - 一次性分析用 `--temp-dir`
   - 重要任务用自定义 `--working-dir`
   - 定期清理旧工作目录

## 按场景示例

### 数据科学工作流

```bash
# 初步探索（临时）
agentic-data-scientist "Explore data distributions and missing values" \
  --mode simple --files data.csv --temp-dir

# 完整分析（保留）
agentic-data-scientist "Perform complete statistical analysis with visualizations" \
  --mode orchestrated --files data.csv --working-dir ./analysis_results

# 模型构建
agentic-data-scientist "Build and evaluate multiple regression models" \
  --mode orchestrated --files train.csv --files test.csv \
  --working-dir ./models
```

### 生物信息学

```bash
# 差异表达
agentic-data-scientist "Perform DESeq2 differential expression analysis" \
  --mode orchestrated \
  --files counts.csv --files metadata.csv \
  --working-dir ./deg_analysis

# 通路分析
agentic-data-scientist "Run GSEA pathway enrichment on DEGs" \
  --mode orchestrated --files deg_results.csv \
  --working-dir ./pathway_analysis
```

### 脚本与自动化

```bash
# 生成工具脚本
agentic-data-scientist "Write Python script to merge CSV files" \
  --mode simple --working-dir ./scripts

# 批处理脚本
agentic-data-scientist "Create script to process multiple data files" \
  --mode simple --files sample_data.csv --working-dir ./scripts
```

### 学习与探索

```bash
# 技术问答
agentic-data-scientist "Explain PCA and when to use it" --mode simple --temp-dir

# 代码示例
agentic-data-scientist "Show me how to use pandas groupby with multiple aggregations" \
  --mode simple --temp-dir
```
