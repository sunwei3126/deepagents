"""Research agent example with context compression enabled."""

import os
from typing import Literal

from tavily import TavilyClient
from langchain_deepseek import ChatDeepSeek

from deepagents import create_deep_agent, SubAgent, CompressionConfig
 
 
# It's best practice to initialize the client once and reuse it.
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs


sub_research_prompt = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

research_sub_agent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": sub_research_prompt,
    "tools": ["internet_search"],
}

sub_critique_prompt = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

critique_sub_agent = {
    "name": "critique-agent", 
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": sub_critique_prompt,
}


# Prompt prefix to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

[Rest of prompt omitted for brevity - same as original research agent]"""

model = ChatDeepSeek(model="deepseek-chat") 

# Create LangGraph-native compression configuration for long research sessions
compression_config = CompressionConfig(
    max_tokens=15000,         # Allow larger context for research work
    strategy="last",          # Keep most recent messages for research continuity
    start_on="human",         # Start counting from user queries
    end_on=("human", "tool"), # Preserve complete tool interactions
    include_system=True,      # Always keep system instructions
    compress_files=True,      # Compress large research files
    max_file_size=8000,       # Compress files over 8k chars
    destructive=False         # Non-destructive to preserve research history
)

# Create the agent with compression enabled
agent = create_deep_agent(
    [internet_search],
    research_instructions,
    model=model,
    subagents=[critique_sub_agent, research_sub_agent],
    compression_config=compression_config,  # Enable compression
).with_config({"recursion_limit": 1000})

def main():
    """Run the research agent with LangGraph compression enabled."""
    print("=== Research Agent with LangGraph Context Compression ===")
    print("Compression settings:")
    print(f"- Max tokens: {compression_config.max_tokens}")
    print(f"- Strategy: {compression_config.strategy}")
    print(f"- Start on: {compression_config.start_on}")
    print(f"- Include system: {compression_config.include_system}")
    print(f"- File compression: {compression_config.compress_files}")
    print(f"- Destructive mode: {compression_config.destructive}")
    print()
    
    # Example research query that will generate lots of context
    query = "Research the current state of artificial intelligence in 2024, including major breakthroughs, key companies, and future trends. Create a comprehensive report."
    
    result = agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    
    print(f"Research completed!")
    print(f"Final message count: {len(result.get('messages', []))}")
    print(f"Files created: {list(result.get('files', {}).keys())}")
    
    # Show final report if created
    if 'final_report.md' in result.get('files', {}):
        report = result['files']['final_report.md']
        print(f"Final report length: {len(report)} characters")
        print("Report preview:")
        print(report[:500] + "..." if len(report) > 500 else report)


if __name__ == "__main__":
    main()