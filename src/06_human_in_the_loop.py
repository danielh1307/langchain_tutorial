from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from config import set_environment
from typing import Dict, List
from langchain_core.runnables import Runnable, RunnablePassthrough
import json

set_environment()

llm = ChatOpenAI(model="gpt-4o")

@tool
def count_emails(last_n_days: int) -> int:
    """Count emails which I received in the last n days."""
    return last_n_days * 2

@tool
def send_email(message: str, recipient: str) -> str:
    """Send an email with a specific message to a specific recipient."""
    return f"Successfully sent email to {recipient}"


def human_approval(msg: AIMessage) -> Runnable:
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no."
    )

    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise ValueError(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg


## without human approval
tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper"""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools
print(chain.invoke("how many emails did i get in the last 5 days?"))


## with human approval
chain = llm_with_tools | human_approval | call_tools
print(chain.invoke("how many emails did i get in the last 39 days?"))
print(chain.invoke("Send sally@gmail.com an email saying 'What's up homie?'"))