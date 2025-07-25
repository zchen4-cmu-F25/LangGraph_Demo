import os
import json
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from dotenv import load_dotenv
load_dotenv()
GITHUB_PAT = os.environ.get("GITHUB_PAT")
GITHUB_USER = os.environ.get("GITHUB_USER")

from langchain_community.chat_models import QianfanChatEndpoint
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-128K",
    temperature=0.6,
    timeout=30
)

client = MultiServerMCPClient(
    {
        "github": {
            "url": "https://api.githubcopilot.com/mcp/",
            "transport": "streamable_http",
            "headers": {
                "Authorization": f"Bearer {GITHUB_PAT}"
            }
        }
    }
)



# Helper: serialize for JSON
def safe_serialize(obj):
    if isinstance(obj, dict):
        return {k: safe_serialize(v) for k, v in obj.items() if not callable(v)}
    elif isinstance(obj, list):
        return [safe_serialize(v) for v in obj]
    elif hasattr(obj, "model_dump"):
        return safe_serialize(obj.model_dump())
    elif hasattr(obj, "__dict__"):
        return safe_serialize(obj.__dict__)
    elif callable(obj):
        return str(obj)
    else:
        return obj

# Helper: filter tools by name
def filter_tools(tools, needed_names):
    return [t for t in tools if getattr(t, "name", None) in needed_names]

# Helper: truncate tool descriptions
def truncate_tool_descriptions(tools, max_length=120):
    for t in tools:
        if hasattr(t, "description") and isinstance(t.description, str):
            t.description = t.description[:max_length]
    return tools

# Specify only the tools you need
NEEDED_TOOL_NAMES = {
    "list_commits"
}

def prepare_tools(tools, needed_tool_names=NEEDED_TOOL_NAMES, desc_length=120):
    filtered = filter_tools(tools, needed_tool_names)
    filtered = truncate_tool_descriptions(filtered, desc_length)
    return filtered

def store_tools_json(tools, filename="available_tools.json"):
    tools_json = json.dumps([safe_serialize(t) for t in tools], indent=2, ensure_ascii=False)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tools_json)

def call_model_factory(filtered_tools):
    def call_model(state: MessagesState):
        # Save the prompt being sent to the model into a JSON file
        with open("last_prompt.json", "w", encoding="utf-8") as f:
            json.dump(safe_serialize(state["messages"]), f, ensure_ascii=False, indent=2)
        response = qianfan_chat.bind_tools(filtered_tools).invoke(state["messages"])
        return {"messages": response}
    return call_model

async def main():
    tools = await client.get_tools()
    filtered_tools = prepare_tools(tools)
    store_tools_json(filtered_tools)

    builder = StateGraph(MessagesState)
    builder.add_node(call_model_factory(filtered_tools))
    builder.add_node(ToolNode(filtered_tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model",
        tools_condition,
    )
    builder.add_edge("tools", "call_model")
    graph = builder.compile()
    github_response = await graph.ainvoke({"messages": f"List the latest one commit of the repository {GITHUB_USER}/LangGraph_Demo."})
    # Shorten output: only show thoughts, conversation, and error if present
    messages = github_response.get("messages", [])
    # Store the entire message in output.json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(safe_serialize(messages), f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    # Run the main function in an event loop
    if not asyncio.get_event_loop().is_running():
        asyncio.run(main())
    else:
        print("Event loop is already running, cannot run main() directly.")
