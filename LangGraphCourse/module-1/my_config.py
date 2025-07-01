import os
import base64
from dotenv import load_dotenv
from os.path import join, dirname
from dataclasses import dataclass

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

@dataclass
class MyConfig:
    OLLAMA_INFERENCE_NODE_IP = os.environ.get("OLLAMA_INFERENCE_NODE_IP")
    OLLAMA_INFERENCE_RUNPOD_ID = os.environ.get("OLLAMA_INFERENCE_RUNPOD_ID")
    VLLM_EMBEDDING_MODEL_INFERENCE_NODE_IP = os.environ.get("VLLM_EMBEDDING_MODEL_INFERENCE_NODE_IP")
    VLLM_LLM_INFERENCE_NODE_IP = os.environ.get("VLLM_LLM_INFERENCE_NODE_IP")
    LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    # OTEL_EXPORTER_OTLP_ENDPOINT = "https://cloud.langfuse.com/api/public/otel" # EU data region
    # OTEL_EXPORTER_OTLP_ENDPOINT = "https://us.cloud.langfuse.com/api/public/otel" # US data region
    LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST")
    SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    def __post_init__(self):
        self.LANGFUSE_AUTH = base64.b64encode(f"{self.LANGFUSE_PUBLIC_KEY}:{self.LANGFUSE_SECRET_KEY}".encode()).decode()