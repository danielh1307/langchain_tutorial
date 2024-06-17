from langchain.llms import HuggingFaceHub

from config import set_environment

set_environment()

llm = HuggingFaceHub(
    model_kwargs={"temperature": 0.5, "max_length": 64},
    repo_id="google/flan-t5-xxl"
)
prompt = "When was Barack Obama born?"
completion = llm.invoke(prompt)
print(completion)
