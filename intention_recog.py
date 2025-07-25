# intention_recog.py
# Author: chenzixuan06@baidu.com
# Date: 2025-07-25
# Description: This script builds an intention recognition node that determines the user's intent based on their query, using the llm.
# It constructs a node for each of the four intents: query version, pack, query owner, and other intent.
#
# Original Requirement:
# 构建一个 意图识别node：根据用户 query，判断出用户的意图，并调用相应的工具，意图有四类：查询版本号、打包、查询负责人，其他意图
# 四类意图，分别构建构建一个 node，根据意图 print 下即可
# 验证下 tools_condition 逻辑



import os
import json
import asyncio
from typing import Any
from typing import Annotated, Sequence, TypedDict

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import langchain and langgraph libraries
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Import the necessary chat model
from langchain_community.chat_models import QianfanChatEndpoint
qianfan_chat = QianfanChatEndpoint(
    model="ERNIE-3.5-128K",
    temperature=0.6,
    timeout=30
)


# Define AgentState with intention
class AgentState(TypedDict):
    """State of the agent including messages and intention."""
    messages: Annotated[Sequence[Any], add_messages]
    intention: str


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


# Helper: intention recognition logic (merged with model call)
async def intention_recognition(state: AgentState):
    """Recognize the user's intention by LLM response and save prompt."""
    # Add a system message to instruct the LLM to classify the intent using SystemMessage
    system_prompt = SystemMessage(
        content=
        "You are an intention recognition agent. "
        "Classify the user's intent from their message into one of the following categories: "
        "'query_version', 'pack', 'query_owner', or 'other_intent'. "
        "Return only the category name as your answer."
    )
    messages = [system_prompt] + state["messages"]

    # Save the prompt being sent to the model into a JSON file in intention_outs folder
    last_prompt_path = os.path.join("intention_outs", "prompt.json")
    with open(last_prompt_path, "w", encoding="utf-8") as f:
        json.dump(safe_serialize(messages), f, ensure_ascii=False, indent=2)

    # Call the model with the prompt
    response = await qianfan_chat.ainvoke(messages)
    # Extract intention from response (assuming response is a string or has .content)
    if hasattr(response, "content"):
        intention = response.content
    else:
        intention = str(response)
    return {"messages": response, "intention": intention}


def decide_next_node(state: AgentState) -> str:
    """This node will select the next node from the call_model node."""
    intention = state.get("intention", "")
    if "version" in intention:
        return "query_version"
    elif "pack" in intention:
        return "pack"
    elif "owner" in intention:
        return "query_owner"
    else:
        return "other_intent"


# Function to build the intention StateGraph
def build_intention_graph() -> StateGraph:
    """Build the intention recognition graph with four nodes for different intents."""
    builder = StateGraph(AgentState)
    # Add the intention recognition node (merged)
    builder.add_node(
        "call_model", 
        intention_recognition
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
        decide_next_node,
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
    # Compile and return the graph
    return builder.compile()


# Main function to run the agent
async def main():
    """Main function to run the intention recognition agent."""
    graph = build_intention_graph()

    # Save the graph as a PNG file
    graph_png = graph.get_graph().draw_mermaid_png()
    with open("intention_graph.png", "wb") as f:
        f.write(graph_png)
    
    # Example usage: check the version of the software (add a typo here intentionally)
    intention_response = await graph.ainvoke({"messages": "I want to check the preson onw of the software."})

    # Store the entire message in output.json in intention_outs folder
    output_json_path = os.path.join("intention_outs", "output.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(safe_serialize(intention_response), f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    asyncio.run(main())
