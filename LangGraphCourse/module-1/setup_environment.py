
from my_config import MyConfig

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

my_config = MyConfig()

def setup_langfuse():
    langfuse = Langfuse(
        public_key = my_config.LANGFUSE_PUBLIC_KEY,
        secret_key = my_config.LANGFUSE_SECRET_KEY,
        host = my_config.LANGFUSE_HOST
    )
    langfuse_handler1 = CallbackHandler()
    return langfuse, langfuse_handler1
