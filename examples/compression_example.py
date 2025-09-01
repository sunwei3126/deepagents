"""Example demonstrating LangGraph-native context compression in deepagents."""

from deepagents import create_deep_agent, CompressionConfig
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os

load_dotenv()

def simple_tool(message: str) -> str:
    """A simple tool that returns the message with a prefix."""
    return f"Tool processed: {message}"


def main():
    print("=== DeepAgents LangGraph Compression Example ===\n")
    api_key=os.environ["DEEPSEEK_API_KEY"]
    model = ChatDeepSeek(model="deepseek-chat", api_key=api_key)
    
    # Create LangGraph-native compression config
    compression_config = CompressionConfig(
        max_tokens=1000,        # Winodw Size for model
        strategy="last",        # Keep most recent messages
        start_on="human",       # Start counting from human messages
        end_on=("human", "tool"), # Valid endpoints
        include_system=True,    # Always preserve system messages
        destructive=True       # Use llm_input_messages (non-destructive)
    )
    
    # Create agent with compression
    instructions = """You are a helpful assistant. You have access to a simple tool.
    
When asked to demonstrate compression, create a long conversation with many tool calls
to show how compression works."""
    
    agent = create_deep_agent(
        tools=[simple_tool],
        model=model,
        instructions=instructions,
        compression_config=compression_config
    )
    
    print("Agent created with LangGraph compression config:")
    print(f"- Max tokens: {compression_config.max_tokens}")
    print(f"- Strategy: {compression_config.strategy}")
    print(f"- Start on: {compression_config.start_on}")
    print(f"- End on: {compression_config.end_on}")
    print(f"- Include system: {compression_config.include_system}")
    print(f"- Destructive: {compression_config.destructive}")
    print()
    
    # Create a long conversation to trigger compression
    messages = [
        "Hello! Please use the simple_tool with message 'test1'",
        "Now use the simple_tool with message 'test2'", 
        "Use the simple_tool with message 'test3'",
        "Use the simple_tool with message 'test4'",
        "Use the simple_tool with message 'test5'",
        "Please use the simple_tool with message 'test6'",
        "Use the simple_tool one more time with message 'test7'",
        "Finally, use the simple_tool with message 'test8'",
        "What was the very first tool call you made?",  # This should show compression working
        "Use the simple_tool with message 'test9'",
    ]
    
    state = {"messages": []}
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Message {i}: {message} ---")
        
        # Add user message
        from langchain_core.messages import HumanMessage
        state["messages"].append(HumanMessage(content=message))
        
        print(f"Messages before processing: {len(state['messages'])}")
        
        # Process with agent (this will trigger compression if needed)
        try:
            result = agent.invoke(state)
            state = result
            
            print(f"Messages after processing: {len(state['messages'])}")
            
            # Show the latest response
            if state["messages"]:
                latest = state["messages"][-1]
                print(f"Agent response: {latest.content[:200]}...")
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    print(f"\n=== Final Conversation Summary ===")
    print(f"Total messages in final state: {len(state.get('messages', []))}")
    print("\nFinal message contents:")
    for i, msg in enumerate(state.get("messages", [])[:], 1):  # Show last 5
        msg_type = type(msg).__name__
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. {msg_type}: {content}")


if __name__ == "__main__":
    main()