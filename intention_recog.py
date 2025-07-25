# 构建一个 意图识别node：根据用户 query，判断出用户的意图，并调用相应的工具，意图有四类：查询版本号、打包、查询负责人，其他意图
# 四类意图，分别构建构建一个 node，根据意图 print 下即可
# 验证下 tools_condition 逻辑

import os
import json
import asyncio
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

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
