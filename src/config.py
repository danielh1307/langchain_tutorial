import os

#OPENAI_API_KEY = "a"
#HUGGINGFACEHUB_API_TOKEN = "a"
#SANITAS_TOKEN = "a"


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key or "TOKEN" in key:
            os.environ[key] = value
