import httpx
import json
import logging
from config import GPT_API_BASE, GPT_API_KEY, GPT_MODEL

logger = logging.getLogger(__name__)


async def call_gpt(messages: list, reasoning: str = "high", timeout: float = 180.0) -> str:
    url = f"{GPT_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning,
    }
    logger.info(f"[GPT] POST {url}  model={GPT_MODEL}  reasoning={reasoning}")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.ConnectError as e:
        msg = f"GPT 연결 실패 — URL: {url}  오류: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except httpx.HTTPStatusError as e:
        msg = f"GPT HTTP 오류 {e.response.status_code} — URL: {url}  응답: {e.response.text[:200]}"
        logger.error(msg)
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f"GPT 호출 오류 — URL: {url}  {type(e).__name__}: {e}"
        logger.error(msg)
        raise
