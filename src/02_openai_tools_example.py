from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import Tool
from langchain.agents import create_openai_tools_agent
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

from config import set_environment

set_environment()

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-tools-agent")
prompt.messages[0].prompt.template = "You are a coding assistant which translates the user input to Python code " \
                                     "which prints the answer. You return this code, and nothing else."
print(prompt.messages[0])

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
                "If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
tools = [repl_tool]

# Choose the LLM that will drive the agent
# Only certain models support this
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what is 2+2`?"})
agent_executor.invoke({"input": "what is the 7th prime number?"})
