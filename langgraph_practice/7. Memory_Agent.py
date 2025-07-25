import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input."""
    response = llm.invoke(state['messages'])
    state['messages'].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []

user_input = input("You: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        "messages": conversation_history
    })
    conversation_history = result['messages']
    user_input = input("You: ")

with open("logging.txt", "w") as log_file:
    log_file.write("Your Conversation Log:\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            log_file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            log_file.write(f"AI: {message.content}\n\n")
    log_file.write("End of Conversation Log.\n")

print("Conversation logged to 'logging.txt'.")
