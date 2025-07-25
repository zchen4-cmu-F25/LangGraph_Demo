
# LangGraph Demo

This repository demonstrates the use of [LangGraph](https://github.com/langchain-ai/langgraph) for building stateful, multi-actor applications with LLMs. It includes various agent and workflow examples, tool integrations, and intention recognition logic for experimentation and learning.

## Project Structure

- `requirements.txt` — Python dependencies for the project
- `.venv/` — Python virtual environment (not tracked by git)
- `.env` — Environment variable configuration (API keys, etc.)
- `Agent_Bot.py` — Example agent bot using LangGraph
- `langGraphMCP.py` — LangGraph agent with MCP tool integration
- `langGraph_mcp_demo.py` — Demo for LangGraph MCP integration
- `Memory_Agent.py` — Agent with memory/stateful logic
- `ReAct.py` — ReAct agent with tool usage and streaming
- `intention_recog.py` — Intention recognition agent and graph
- `output.json`, `available_tools.json`, etc. — Output and tool metadata files
- `Hello_World.ipynb`, `Looping.ipynb`, `Multiple_Inputs.ipynb`, `Sequential_Agent.ipynb`, `Conditional_Agent.ipynb` — Jupyter notebooks with LangGraph examples
- `README.md` — This file

## Setup

1. Create and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root to set environment variables (such as API keys) required by some scripts. Example:
    ```env
    GITHUB_PAT=your_github_pat
    GITHUB_USER=your_github_username
    QIANFAN_AK=your_qianfan_ak
    QIANFAN_SK=your_qianfan_sk
    ...
    ```

## Usage

Run any of the main Python scripts, for example:
```bash
python Agent_Bot.py
python langGraphMCP.py
python Memory_Agent.py
python intention_recog.py
```

Or open and run the Jupyter notebooks for interactive demos:
```bash
jupyter notebook
```

## License

MIT
