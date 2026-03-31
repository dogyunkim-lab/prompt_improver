import time
from config import DIFY_CLIENT_ID, DIFY_CLIENT_SECRET

# object_id별 토큰 캐시: {object_id: {"token": str, "expires_at": float}}
_token_cache: dict[str, dict] = {}


async def get_dify_token(object_id: str) -> str:
    """object_id에 대한 유효한 토큰을 반환. 만료 60초 전에 자동 재발급."""
    cached = _token_cache.get(object_id)
    if cached and time.time() < cached["expires_at"] - 60:
        return cached["token"]

    token, expires_in = await _issue_token(DIFY_CLIENT_ID, DIFY_CLIENT_SECRET, object_id)
    _token_cache[object_id] = {
        "token": token,
        "expires_at": time.time() + expires_in
    }
    return token


async def _issue_token(client_id: str, client_secret: str, object_id: str) -> tuple[str, int]:
    """
    토큰 발급 함수 — 사용자가 제공한 함수로 교체하세요.
    반환: (token_string, expires_in_seconds)
    """
    raise NotImplementedError(
        "토큰 발급 함수를 구현해 주세요. "
        "_issue_token(client_id, client_secret, object_id) -> (token, expires_in)"
    )
