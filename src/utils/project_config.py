import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

class Config:
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    # Chroma DB stored at project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
    CHROMA_PERSIST_DIRECTORY = str(PROJECT_ROOT / "chroma_db")

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # Hugging Face models
    LLM_SCAN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_REASONING_MODEL = "Qwen/Qwen2.5-7B-Instruct"
    LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

    @classmethod
    def validate_api_key(cls):
        if not cls.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing in .env file")
        if cls.HUGGINGFACEHUB_API_TOKEN.startswith("your_") or len(cls.HUGGINGFACEHUB_API_TOKEN) < 10:
            raise ValueError(f"Invalid HUGGINGFACEHUB_API_TOKEN. Please check your .env file.")
