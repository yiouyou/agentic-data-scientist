# Getting Started with Agentic Data Scientist

This guide will help you understand and use the Agentic Data Scientist multi-agent workflow.

## Installation

```bash
# Install from PyPI using uv
uv tool install agentic-data-scientist

# Or use with uvx (no installation needed)
uvx agentic-data-scientist "your query here"
```

## Prerequisites

- Python 3.12 or later
- At least one coding executor CLI:
  - Claude Code (requires Node.js)
  - Codex CLI (`codex`)
  - OpenCode CLI (`opencode`)
- API keys:
  - Key for your selected coding executor profile (for example `ANTHROPIC_API_KEY` for `claude_code`)
  - Provider keys for any profiles enabled in `configs/llm_routing.yaml`

## Quick Start

### 1. Set up environment variables

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Then fill required values:

```bash
# Core required (example for claude_code executor)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required only if enabled in configs/llm_routing.yaml
# OPENAI_API_KEY=your_openai_key_here
# GOOGLE_API_KEY=your_google_key_here
# DASHSCOPE_API_KEY=your_dashscope_key_here
# DEEPSEEK_API_KEY=your_deepseek_key_here

# Optional: Model configuration
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
# ADS_LOCAL_SKILLS_SOURCE="scientific-skills"
# ADS_SKILLS_SCOPE_NAME="scientific-skills"
# CODEX_COMMAND_TEMPLATE="codex exec --model {model}"
# OPENCODE_COMMAND_TEMPLATE="opencode run --model {model}"
```

Get your API keys:
- Anthropic: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys
- Google: https://aistudio.google.com/app/apikey
- DashScope (Qwen): https://dashscope.console.aliyun.com/
- DeepSeek: https://platform.deepseek.com/api_keys

Optional startup check:
```bash
agentic-data-scientist --llm-preflight --llm-config configs/llm_routing.yaml
```

Offline planner policy replay:
```bash
agentic-data-scientist --history-replay --history-replay-limit 200
```

Use `ADS_PLAN_SELECTOR_INTENT_REGEXES` to restrict learning-based plan selection to specific task intents/domains.

### 2. Run your first query

**Important:** You must specify `--mode` to choose your execution strategy.

```bash
# Complex analysis with full workflow
agentic-data-scientist "Perform differential expression analysis" --mode orchestrated --files data.csv

# Quick scripting task
agentic-data-scientist "Write a Python script to parse CSV" --mode simple

# Question answering
agentic-data-scientist "Explain gradient boosting" --mode simple
```

### 3. Working Directory Options

By default, files are saved to `./agentic_output/` and preserved after completion:

```bash
# Default behavior (files preserved)
agentic-data-scientist "Analyze data" --mode orchestrated --files data.csv

# Temporary directory (auto-cleanup)
agentic-data-scientist "Quick exploration" --mode simple --files data.csv --temp-dir

# Custom location
agentic-data-scientist "Project analysis" --mode orchestrated --files data.csv --working-dir ./my_project
```

## Understanding the Workflow

When you submit a query, Agentic Data Scientist goes through a multi-phase workflow designed to produce high-quality, validated results.

### What Happens When You Run a Query

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

### Key Workflow Characteristics

**Iterative Refinement**
- Plans are reviewed and refined before execution begins
- Implementations are validated before proceeding to the next stage
- Multiple opportunities to catch and fix issues early

**Adaptive Execution**
- Discoveries during implementation inform subsequent stages
- Plan adapts based on actual progress and findings
- Flexible enough to handle unexpected insights

**Continuous Validation**
- Success criteria tracked objectively throughout execution
- Clear visibility into what's been accomplished vs. what remains
- Objective evidence for each criterion's status

**Separation of Concerns**
- Planning agents focus on strategy without implementation details
- Coding agent focuses on implementation without planning burden
- Review agents provide independent validation

## Python API Usage

### Basic Usage

```python
from agentic_data_scientist import DataScientist

# Create an instance and run a query
with DataScientist() as ds:
    result = ds.run("What is data science?")
    print(result.response)
    
# Access results
print(f"Status: {result.status}")
print(f"Duration: {result.duration}s")
print(f"Files created: {result.files_created}")
```

### With File Upload

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

### Async Usage with Streaming

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
            # Watch the workflow in real-time
            if event['type'] == 'message':
                author = event['author']
                content = event['content']
                print(f"[{author}] {content}")
            elif event['type'] == 'completed':
                print(f"✓ Completed in {event['duration']}s")

asyncio.run(analyze_data())
```

### Multi-turn Conversation

```python
import asyncio
from agentic_data_scientist import DataScientist

async def chat():
    async with DataScientist() as ds:
        context = {}
        
        # First turn
        result1 = await ds.run_async(
            "What are the main techniques for dimensionality reduction?",
            context=context
        )
        print("AI:", result1.response)
        
        # Second turn (maintains context)
        result2 = await ds.run_async(
            "Which one would you recommend for high-dimensional gene expression data?",
            context=context
        )
        print("AI:", result2.response)

asyncio.run(chat())
```

## Understanding Streaming Events

When using `stream=True`, you'll receive events as the workflow progresses:

```python
async for event in await ds.run_async("Your query", stream=True):
    event_type = event['type']
    
    if event_type == 'message':
        # Regular text output from agents
        print(f"[{event['author']}] {event['content']}")
        
    elif event_type == 'function_call':
        # Agent is using a tool
        print(f"Calling {event['name']}...")
        
    elif event_type == 'function_response':
        # Tool returned a result
        print(f"Tool {event['name']} completed")
        
    elif event_type == 'usage':
        # Token usage information
        tokens = event['usage']
        print(f"Tokens: {tokens['total_input_tokens']} in, {tokens['output_tokens']} out")
        
    elif event_type == 'completed':
        # Workflow finished
        print(f"Done in {event['duration']}s")
        print(f"Created {len(event['files_created'])} files")
```

## Execution Modes

### Orchestrated Mode (Recommended)

Full multi-agent workflow with planning, validation, and adaptive execution.

**When to use:**
- Complex data analyses
- Multi-step workflows  
- Tasks requiring validation
- Production analyses

**Example:**
```bash
agentic-data-scientist "Perform DEG analysis comparing treatment vs control" \
  --mode orchestrated \
  --files treatment.csv --files control.csv
```

### Simple Mode

Direct coding without planning overhead.

**When to use:**
- Quick scripts
- Code generation
- Question answering
- Rapid prototyping

**Example:**
```bash
agentic-data-scientist "Write a function to merge CSV files" --mode simple
```

## Next Steps

See the `docs/` folder for additional guides on API usage, CLI options, customization, and technical architecture.

## Troubleshooting

### Common Issues

**ImportError: No module named 'agentic_data_scientist'**
- Install the package: `pip install agentic-data-scientist` or `uv sync`

**API Key Errors**
- Ensure your `.env` file is in the correct location
- Verify API keys are valid and active
- Check that keys have sufficient credits
- Ensure keys exist for profiles enabled in `configs/llm_routing.yaml`
- Run preflight: `agentic-data-scientist --llm-preflight --llm-config configs/llm_routing.yaml`

**Node.js Issues**
- Ensure Node.js is installed: `node --version`
- Required when your coding executor is `claude_code`
- Restart terminal after installing Node.js

**Workflow Seems Stuck**
- Enable streaming to see progress: `--stream` or `stream=True`
- Check logs for error messages
- Workflow may be running long computations - be patient

### Getting Help

- Check the full documentation in the `docs/` folder
- Open an issue on [GitHub](https://github.com/K-Dense-AI/agentic-data-scientist/issues)
