# 构建一个 意图识别node：根据用户 query，判断出用户的意图，并调用相应的工具，意图有四类：查询版本号、打包、查询负责人，其他意图
# 四类意图，分别构建构建一个 node，根据意图 print 下即可
# 验证下 tools_condition 逻辑

import os
import json
import asyncio
from typing import Any
from langgraph.graph import StateGraph, MessagesState, START, END

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the necessary chat model
from langchain_community.chat_models import QianfanChatEndpoint
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-128K",
    temperature=0.6,
    timeout=30
)


# Helper: serialize for JSON
def safe_serialize(obj: Any) -> json:
    """Recursively serialize an object to a JSON-compatible format."""
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


# Helper: call the model
async def call_model(state: MessagesState):
        # Save the prompt being sent to the model into a JSON file in outputs folder
        last_prompt_path = os.path.join("outputs", "last_prompt.json")
        with open(last_prompt_path, "w", encoding="utf-8") as f:
            json.dump(safe_serialize(state["messages"]), f, ensure_ascii=False, indent=2)

        # Call the model with the prompt
        response = await qianfan_chat.invoke(state["messages"])
        return {"messages": response}


# Helper: intention recognition logic
def intention_recognition(state: MessagesState):
    """Recognize the user's intention by llm response."""
    pass


# Main function to run the agent
async def main():
    # Create the state graph with MessagesState
    builder = StateGraph(MessagesState)
    # Add the model call node
    builder.add_node(
        "call_model", 
        call_model
    )
    # Add the four nodes for different intents
    builder.add_node("query_version", lambda state: print("Querying version..."))
    builder.add_node("pack", lambda state: print("Packing..."))
    builder.add_node("query_owner", lambda state: print("Querying owner..."))
    builder.add_node("other_intent", lambda state: print("Handling other intent..."))
    # Define the edges of the graph
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", 
        intention_recognition,
        {
        "query_version": "query_version",
        "pack": "pack",
        "query_owner": "query_owner",
        "other_intent": "other_intent"
        }
    )
    builder.add_edge("query_version", END)
    builder.add_edge("pack", END)
    builder.add_edge("query_owner", END)
    builder.add_edge("other_intent", END)
    # Compile the graph
    graph = builder.compile()

    # Save the graph as a PNG file
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_png)
    
    # Example usage: async invoke the agent with a GitHub query
    github_response = await graph.ainvoke({"messages": f"List the latest one commit of the repository /LangGraph_Demo."})

    # Store the entire message in output.json in outputs folder
    messages = github_response.get("messages", [])
    output_json_path = os.path.join("outputs", "output.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(safe_serialize(messages), f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    asyncio.run(main())
