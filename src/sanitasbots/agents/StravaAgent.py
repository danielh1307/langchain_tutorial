import openai
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.globals import set_verbose
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langsmith.wrappers import wrap_openai

from config import set_environment

# load the necessary keys ...
set_environment()

# set up langsmith
client = wrap_openai(openai.Client())
set_verbose(True)


class Route(BaseModel):
    name: str = Field(description="Name of the route")


@tool
def add_route(route: Route):
    """Allows the user to upload a new route to the Strava profile"""
    print("*** This is our new route: ", route)


@tool
def get_route(id: str) -> Route:
    """Load the route with a specific id"""
    print("*** Here I am loading a route with id: ", id)
    route = Route(name="Hugo Hase Route")
    return route


tools = [add_route, get_route]

MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant to handle my Strava profile. "
            "Strava is an online social network for sports events. "
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
chat_history = []

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    user_msg = input("You: ")
    if user_msg.lower() == 'quit':
        break
    result = agent_executor.invoke({"input": user_msg, "chat_history": chat_history})
    chat_history.extend(
        [
            HumanMessage(content=user_msg),
            AIMessage(content=result["output"]),
        ]
    )