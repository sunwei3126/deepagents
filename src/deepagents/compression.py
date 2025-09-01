"""Context compression functionality for deep agents using LangGraph utilities."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal, Union
from langchain_core.messages import SystemMessage
from langchain_core.messages.utils import trim_messages, count_tokens_approximately


@dataclass
class CompressionConfig:
    """Configuration for LangGraph-native context compression.
    
    Args:
        max_tokens: Maximum number of tokens to keep in context
        strategy: Message selection strategy - "first" (keep earliest) or "last" (keep latest)
        start_on: Type of message to start counting from when trimming
        end_on: Tuple of message types that are valid endpoints when trimming
        include_system: Whether to always preserve system messages
        compress_files: Whether to compress large files in virtual filesystem  
        max_file_size: Maximum file size in characters before compression
        destructive: If True, overwrites message history; if False, uses llm_input_messages
    """
    max_tokens: int = 8000
    strategy: Literal["first", "last"] = "last"
    start_on: Literal["human", "ai", "tool"] = "human"
    end_on: Union[Literal["human", "ai", "tool"], tuple] = ("human", "tool")
    include_system: bool = True
    compress_files: bool = True
    max_file_size: int = 10000
    destructive: bool = False


def create_compression_pre_hook(config: CompressionConfig) -> callable:
    """Create a pre-model hook that compresses context using LangGraph utilities.
    
    Args:
        config: CompressionConfig object with compression parameters
    """
    
    def compression_hook(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pre-model hook that trims messages using LangGraph utilities."""
        messages = state.get("messages", [])
        if not messages:
            return _compress_files_if_needed(state, config)
        
        # Count current tokens to see if compression is needed
        current_tokens = count_tokens_approximately(messages)
        if current_tokens <= config.max_tokens:
            return _compress_files_if_needed(state, config)
        
        # Use LangGraph's trim_messages for reliable compression
        try:
            trimmed_messages = trim_messages(
                messages,
                strategy=config.strategy,
                token_counter=count_tokens_approximately,
                max_tokens=config.max_tokens,
                start_on=config.start_on,
                end_on=config.end_on,
                include_system=config.include_system,
            )
            
            # Add compression notification
            compression_msg = SystemMessage(
                content=f"[SYSTEM: Context compressed using LangGraph - reduced from {len(messages)} to {len(trimmed_messages)} messages, "
                       f"~{current_tokens} to ~{count_tokens_approximately(trimmed_messages)} tokens using '{config.strategy}' strategy.]"
            )
            trimmed_messages.append(compression_msg)
            
            # Prepare updates
            updates = {}
            
            if config.destructive:
                # Overwrite message history completely
                from langchain_core.messages import RemoveMessage
                from langgraph.graph.message import REMOVE_ALL_MESSAGES
                updates["messages"] = [RemoveMessage(REMOVE_ALL_MESSAGES)] + trimmed_messages
            else:
                # Non-destructive: use llm_input_messages
                updates["llm_input_messages"] = trimmed_messages
            
            # Handle file compression separately
            file_updates = _compress_files_if_needed(state, config)
            if file_updates:
                updates.update(file_updates)
            
            return updates if updates else None
            
        except Exception as e:
            print(f"[WARNING: Compression failed: {e}]")
            return _compress_files_if_needed(state, config)
    
    return compression_hook


# Backward compatibility alias
def create_compression_hook(config: CompressionConfig) -> callable:
    """Legacy alias for create_compression_pre_hook."""
    return create_compression_pre_hook(config)


def _compress_files(files: Dict[str, str], config: CompressionConfig) -> Dict[str, str]:
    """Compress files that exceed size limits."""
    compressed = {}
    
    for filename, content in files.items():
        if len(content) > config.max_file_size:
            # For large files, keep beginning and end with truncation notice
            keep_size = config.max_file_size // 3
            truncated_content = (
                content[:keep_size] + 
                f"\n\n[... Content truncated - {len(content) - 2*keep_size} characters omitted ...]\n\n" +
                content[-keep_size:]
            )
            compressed[filename] = truncated_content
        else:
            compressed[filename] = content
    
    return compressed




def _compress_files_if_needed(state: Dict[str, Any], config: CompressionConfig) -> Optional[Dict[str, Any]]:
    """Compress only files if needed, even when messages don't need compression."""
    if not config.compress_files:
        return None
    
    files = state.get("files", {})
    if not files:
        return None
    
    compressed_files = _compress_files(files, config)
    
    # Only return update if files actually changed
    if compressed_files != files:
        # Add file compression notification
        compressed_file_names = [name for name, content in files.items() 
                               if len(content) > config.max_file_size]
        
        if compressed_file_names:
            # Create notification message
            file_msg = SystemMessage(
                content=f"[SYSTEM: Files compressed - {', '.join(compressed_file_names)} truncated due to size limit ({config.max_file_size} chars).]"
            )
            
            if config.destructive:
                # Add to messages if destructive mode
                messages = state.get("messages", [])
                return {"files": compressed_files, "messages": messages + [file_msg]}
            else:
                # Add to llm_input_messages if non-destructive
                llm_messages = state.get("llm_input_messages", state.get("messages", []))
                return {"files": compressed_files, "llm_input_messages": llm_messages + [file_msg]}
        
        return {"files": compressed_files}
    
    return None