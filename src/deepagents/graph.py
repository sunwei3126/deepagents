from deepagents.sub_agent import _create_task_tool, SubAgent
from deepagents.model import get_default_model
from deepagents.tools import write_todos, write_file, read_file, ls, edit_file
from deepagents.state import DeepAgentState
from deepagents.compression import CompressionConfig, create_compression_pre_hook
from typing import Sequence, Union, Callable, Any, List, TypeVar, Type, Optional
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import LanguageModelLike
from deepagents.interrupt import create_interrupt_hook, ToolInterruptConfig
from langgraph.types import Checkpointer
from langgraph.prebuilt import create_react_agent


StateSchema = TypeVar("StateSchema", bound=DeepAgentState)
StateSchemaType = Type[StateSchema]

base_prompt = """You have access to a number of standard tools

## `write_todos`

You have access to the `write_todos` tools to help you manage and plan tasks. Use these tools VERY frequently to ensure that you are tracking your tasks and giving the user visibility into your progress.
These tools are also EXTREMELY helpful for planning tasks, and for breaking down larger complex tasks into smaller steps. If you do not use this tool when planning, you may forget to do important tasks - and that is unacceptable.

It is critical that you mark todos as completed as soon as you are done with a task. Do not batch up multiple tasks before marking them as completed.
## `task`

- When doing web search, prefer to use the `task` tool in order to reduce context usage."""


def chain_hooks(hooks: List[Callable]) -> Callable:
    """Chain multiple hooks together.
    
    Args:
        hooks: List of hook functions that take state and return updates
    
    Returns:
        Combined hook function that applies all hooks in sequence
    """
    def combined_hook(state):
        updates = {}
        for hook in hooks:
            hook_result = hook(state)
            if hook_result:
                updates.update(hook_result)
                # Apply updates to state for next hook
                state = {**state, **hook_result}
        return updates if updates else None
    
    return combined_hook


def create_deep_agent(
    tools: Sequence[Union[BaseTool, Callable, dict[str, Any]]],
    instructions: str,
    model: Optional[Union[str, LanguageModelLike]] = None,
    subagents: list[SubAgent] = None,
    state_schema: Optional[StateSchemaType] = None,
    builtin_tools: Optional[list[str]] = None,
    interrupt_config: Optional[ToolInterruptConfig] = None,
    compression_config: Optional[CompressionConfig] = None,
    config_schema: Optional[Type[Any]] = None,
    checkpointer: Optional[Checkpointer] = None,
    post_model_hook: Optional[Callable] = None,
):
    """Create a deep agent.

    This agent will by default have access to a tool to write todos (write_todos),
    and then four file editing tools: write_file, ls, read_file, edit_file.

    Args:
        tools: The additional tools the agent should have access to.
        instructions: The additional instructions the agent should have. Will go in
            the system prompt.
        model: The model to use.
        subagents: The subagents to use. Each subagent should be a dictionary with the
            following keys:
                - `name`
                - `description` (used by the main agent to decide whether to call the sub agent)
                - `prompt` (used as the system prompt in the subagent)
                - (optional) `tools`
                - (optional) `model` (either a LanguageModelLike instance or dict settings)
        state_schema: The schema of the deep agent. Should subclass from DeepAgentState
        builtin_tools: If not provided, all built-in tools are included. If provided, 
            only the specified built-in tools are included.
        interrupt_config: Optional Dict[str, HumanInterruptConfig] mapping tool names to interrupt configs.

        config_schema: The schema of the deep agent.
        checkpointer: Optional checkpointer for persisting agent state between runs.
        post_model_hook: Optional custom post model hook. Can be used with compression_config.
    """
    
    prompt = instructions + base_prompt
    
    all_builtin_tools = [write_todos, write_file, read_file, ls, edit_file]
    
    if builtin_tools is not None:
        tools_by_name = {}
        for tool_ in all_builtin_tools:
            if not isinstance(tool_, BaseTool):
                tool_ = tool(tool_)
            tools_by_name[tool_.name] = tool_
        # Only include built-in tools whose names are in the specified list
        built_in_tools = [ tools_by_name[_tool] for _tool in builtin_tools        ]
    else:
        built_in_tools = all_builtin_tools
    
    if model is None:
        model = get_default_model()
    state_schema = state_schema or DeepAgentState
    task_tool = _create_task_tool(
        list(tools) + built_in_tools,
        instructions,
        subagents or [],
        model,
        state_schema
    )
    all_tools = built_in_tools + list(tools) + [task_tool]
    
    # Set up compression pre-model hook if configured
    compression_pre_hook = None
    if compression_config is not None:
        compression_pre_hook = create_compression_pre_hook(compression_config)
    
    # Set up post-model hooks (interrupt + custom)
    post_hooks = []
    if interrupt_config is not None:
        post_hooks.append(create_interrupt_hook(interrupt_config))
    if post_model_hook is not None:
        post_hooks.append(post_model_hook)
    
    selected_post_model_hook = chain_hooks(post_hooks) if post_hooks else None
    
    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        pre_model_hook=compression_pre_hook,
        post_model_hook=selected_post_model_hook,
        config_schema=config_schema,
        checkpointer=checkpointer,
    )
