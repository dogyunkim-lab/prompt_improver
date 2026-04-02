import logging
from openai import AsyncOpenAI
from config import GPT_API_BASE, GPT_API_KEY, GPT_MODEL

logger = logging.getLogger(__name__)

# api_key가 비어있으면 openai 클라이언트가 오류를 내므로 기본값 설정
_client = AsyncOpenAI(
    base_url=GPT_API_BASE,
    api_key=GPT_API_KEY if GPT_API_KEY else "none",
)


async def call_gpt(messages: list, reasoning: str = "high", timeout: float = 180.0) -> str:
    logger.info(f"[GPT] POST {GPT_API_BASE}/chat/completions  model={GPT_MODEL}  reasoning={reasoning}")
    try:
        response = await _client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            extra_body={"reasoning_effort": reasoning},
            timeout=timeout,
        )
        return response.choices[0].message.content
    except Exception as e:
        msg = f"GPT 호출 오류 — {GPT_API_BASE}  {type(e).__name__}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
