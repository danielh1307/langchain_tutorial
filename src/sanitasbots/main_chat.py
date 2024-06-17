from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import set_environment
from src.sanitasbots.tools.address_change import address_change
from src.sanitasbots.tools.address_change import get_coverage
from src.sanitasbots.tools.call_strava import add_route

import json

set_environment()

# hier erstellen wir das LLM
llm = ChatOpenAI(model="gpt-4o")

# jetzt geben wir dem LLM die Tools mit, die es braucht
tools = [address_change, add_route, get_coverage]
llm_with_tools = llm.bind_tools(tools)
tool_map = {tool.name: tool for tool in tools}

main_system_prompt = """
Du bist ein digitaler Assistent der Schweizer Krankenversicherung Sanitas.
Ich bin ein Kunde von Sanitas. Sei freundlich und hilfsbereit zu mir.
Wenn ich die Unterhaltung beende, antworte nur mit einem Wort: QUIT.
"""
main_prompt = ChatPromptTemplate.from_template(
    main_system_prompt +
    """
    Conversation history:
    {history}
    
    Human: {input}
    AI: 
    """
)

output_parser = StrOutputParser()
memory = ConversationBufferMemory(return_messages=True)
chain = main_prompt | llm_with_tools

while True:
    user_msg = input("You: ")
    if user_msg.lower() == 'quit':
        break
    llm_response = chain.invoke({"input": user_msg, "history": memory.load_memory_variables({})})
    bot_response_text = output_parser.parse(llm_response.content)
    # save context
    memory.save_context({"input": user_msg}, {"output": bot_response_text})
    if bot_response_text.lower() == "quit":
        break
    if llm_response.tool_calls:
        print("Tool call: ", llm_response.tool_calls)
        for tool_call in llm_response.tool_calls:
            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            print(json.dumps(result, indent=4))
            # Update the conversation history with the tool call result
            memory.save_context({"input": f"Tool call: {tool_call['name']} with args {tool_call['args']}"}, {"output": str(result)})

            # Get a new response from the LLM including the tool result
            llm_response = chain.invoke({"input": f"Tool call result: {str(result)}", "history": memory.load_memory_variables({})})
            bot_response_text = output_parser.parse(llm_response.content)
            print("Sanitas Bot: ", bot_response_text)

            # Save the context after getting the new response
            memory.save_context({"input": f"Tool call result: {str(result)}"}, {"output": bot_response_text})
    else:
        print("Sanitas Bot: ", bot_response_text)


