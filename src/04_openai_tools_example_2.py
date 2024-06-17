from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from config import set_environment


@tool
def multiply(first_int: int, second_int: int) -> int:
    """Multiply two integers together"""
    return first_int * second_int


set_environment()

print(multiply.name)
print(multiply.description)
print(multiply.args)

result = multiply.invoke({"first_int": 4, "second_int": 5})
print(result)

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([multiply])

msg = llm_with_tools.invoke("whats 5 times forty two")
print(msg.tool_calls)

chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]) | multiply
print(chain.invoke("what's four times 23?"))

