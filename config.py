import os
from dotenv import load_dotenv

load_dotenv()

# 폐쇄망 gpt-oss-120B API (OpenAI-compatible)
GPT_API_BASE    = os.getenv("GPT_API_BASE", "http://내부서버주소/v1")
GPT_API_KEY     = os.getenv("GPT_API_KEY", "")
GPT_MODEL       = os.getenv("GPT_MODEL", "gpt-oss-120b-26")
GPT_REASONING   = os.getenv("GPT_REASONING", "high")   # low / medium / high

# Dify (CLIENT_ID/SECRET/BASE_URL은 고정값, object_id는 Phase 3 UI에서 워크플로우별 입력)
DIFY_CLIENT_ID     = os.getenv("DIFY_CLIENT_ID", "")
DIFY_CLIENT_SECRET = os.getenv("DIFY_CLIENT_SECRET", "")
DIFY_BASE_URL      = os.getenv("DIFY_BASE_URL", "http://내부서버주소/v1")

# 서버
HOST = "0.0.0.0"
PORT = 8000
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "improver.db")
