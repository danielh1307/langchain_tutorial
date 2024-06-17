from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain_community.llms import FakeListLLM
from langchain_experimental.utilities import PythonREPL

responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
llm = FakeListLLM(responses=responses)

python_repl = PythonREPL()

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
                "If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run
)

agent = initialize_agent(
    [repl_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

agent.invoke("what is 2+2")