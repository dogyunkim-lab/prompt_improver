import os
from dotenv import load_dotenv

load_dotenv()

# 폐쇄망 gpt-oss-120B API (OpenAI-compatible)
GPT_API_BASE    = os.getenv("GPT_API_BASE", "http://내부서버주소/v1")
GPT_API_KEY     = os.getenv("GPT_API_KEY", "")
GPT_MODEL       = os.getenv("GPT_MODEL", "gpt-oss-120b-26")
GPT_REASONING   = os.getenv("GPT_REASONING", "high")   # low / medium / high

# Dify 워크플로우 API
DIFY_BASE_URL   = os.getenv("DIFY_BASE_URL", "http://내부서버주소/v1")
DIFY_API_KEY    = os.getenv("DIFY_API_KEY", "")

# 서버
HOST = "0.0.0.0"
PORT = 8000
DB_PATH = "data/improver.db"
