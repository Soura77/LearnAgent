import os
from my_config import MyConfig
from langfuse.langchain import CallbackHandler

my_config = MyConfig()

def setup_langfuse_tracer():
    os.environ["LANGFUSE_PUBLIC_KEY"] = my_config.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = my_config.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = my_config.LANGFUSE_HOST
    # Initialize Langfuse CallbackHandler for LangGraph/Langchain (tracing)
    return CallbackHandler()

# def hf_login():
#     from huggingface_hub import login
#     login(token=my_config.HF_TOKEN)