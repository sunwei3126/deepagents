# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`deepagents` is a Python package that implements "Deep Agents" - LLM agents with planning tools, sub-agents, file system access, and detailed prompts. It's built on LangGraph and heavily inspired by Claude Code's architecture.

## Development Commands

### Installation
```bash
pip install deepagents
```

### Package Installation from Source
```bash
pip install -e .
```

### Running Examples
```bash
# Research agent example (requires TAVILY_API_KEY environment variable)
pip install tavily-python
python examples/research/research_agent.py
```

## Architecture Overview

### Core Components

1. **Main Agent Factory** (`src/deepagents/graph.py`): 
   - `create_deep_agent()` is the main entry point that creates LangGraph agents
   - Combines user tools with built-in tools (todos, file operations, sub-agent spawning)
   - Uses LangGraph's `create_react_agent` internally

2. **Built-in Tools** (`src/deepagents/tools.py`):
   - `write_todos`: Task planning and tracking
   - `write_file`, `read_file`, `edit_file`, `ls`: Mock file system operations
   - File operations use LangGraph state, not actual filesystem

3. **Sub-Agent System** (`src/deepagents/sub_agent.py`):
   - `task` tool allows spawning specialized sub-agents
   - Sub-agents can have custom prompts, tools, and model configurations
   - Includes built-in "general-purpose" sub-agent

4. **State Management** (`src/deepagents/state.py`):
   - `DeepAgentState` extends LangGraph's `AgentState`
   - Tracks todos and virtual file system state
   - Uses reducers for proper state merging

5. **Prompting** (`src/deepagents/prompts.py`):
   - Contains detailed system prompts inspired by Claude Code
   - Includes instructions for todo management, file operations, and sub-agent usage

### Key Design Patterns

- **Virtual File System**: File operations don't touch real filesystem, they use LangGraph state
- **Todo-Driven Planning**: Built-in todo system for task tracking and decomposition  
- **Context Quarantine**: Sub-agents prevent context pollution in main agent
- **Composable Tools**: User tools combined with built-in capabilities seamlessly

## Agent Creation Patterns

### Basic Agent
```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[your_tools],
    instructions="Your system prompt instructions"
)
```

### With Sub-agents
```python
subagents = [{
    "name": "researcher",
    "description": "Conducts detailed research",
    "prompt": "You are a research specialist...",
    "tools": ["internet_search"],  # Optional: subset of tools
    "model_settings": {"temperature": 0}  # Optional: per-agent config
}]

agent = create_deep_agent(
    tools=[internet_search],
    instructions="You are an expert...",
    subagents=subagents
)
```

### With Custom Model
```python
from langchain.chat_models import init_chat_model

model = init_chat_model("ollama:gpt-oss:20b")
agent = create_deep_agent(tools=tools, instructions=instructions, model=model)
```

### With Tool Interrupts
```python
from langgraph.prebuilt.interrupt import HumanInterruptConfig

agent = create_deep_agent(
    tools=tools,
    instructions=instructions,
    interrupt_config={
        "write_file": HumanInterruptConfig(allow_accept=True, allow_edit=False)
    }
)
```

## State Access Patterns

### Passing Files to Agent
```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "Process these files"}],
    "files": {"config.json": '{"setting": "value"}', "data.txt": "content"}
})
```

### Accessing Agent Output Files
```python
result = agent.invoke({"messages": [...]})
output_files = result["files"]  # Dict of filename -> content
todos_created = result.get("todos", [])  # List of Todo objects
```

## Working with This Codebase

- **No test framework**: Project has no test configuration or test files
- **No development scripts**: No Makefile, npm scripts, or development automation
- **Simple build**: Uses standard setuptools with pyproject.toml
- **Dependencies**: Core deps are langgraph, langchain-anthropic, langchain
- **Examples**: See `examples/research/` for comprehensive usage patterns

## Important Implementation Details

- Default model is "claude-sonnet-4-20250514" if none specified
- Agents have recursion_limit that may need adjustment for complex tasks
- Sub-agents inherit main agent's tools unless explicitly restricted
- File system is single-level (no subdirectories in virtual filesystem)
- All LangGraph features (streaming, checkpointing, human-in-the-loop) work with created agents