import httpx
import json
from config import GPT_API_BASE, GPT_API_KEY, GPT_MODEL


async def call_gpt(messages: list, reasoning: str = "high", timeout: float = 180.0) -> str:
    headers = {
        "Authorization": f"Bearer {GPT_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "reasoning_effort": reasoning,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{GPT_API_BASE}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
