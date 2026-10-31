import json
from typing import Any

import redis.asyncio as redis

from app.config import settings


# ============================================================
# GLOBAL REDIS CLIENT
# ============================================================

_redis_client: redis.Redis | None = None


# ============================================================
# GET CLIENT
# ============================================================

def get_redis() -> redis.Redis:
    """
    Return the shared asynchronous Redis client.

    Client creation is lazy so importing this module does not
    immediately establish a network connection.
    """

    global _redis_client

    if _redis_client is None:

        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=10,
            health_check_interval=30,
        )

    return _redis_client


# ============================================================
# INITIALIZE REDIS
# ============================================================

async def init_redis() -> None:
    """
    Verify Redis connectivity during application startup.
    """

    client = get_redis()

    print(
        "[REDIS] Checking connection..."
    )

    try:

        await client.ping()

        print(
            "[REDIS] Connection ready."
        )

    except Exception as exc:

        print(
            "[REDIS] Connection failed:"
        )

        print(
            repr(exc)
        )

        raise


# ============================================================
# CLOSE REDIS
# ============================================================

async def close_redis() -> None:
    """
    Close Redis client during application shutdown.
    """

    global _redis_client

    if _redis_client is None:
        return

    try:

        print(
            "[REDIS] Closing connection..."
        )

        await _redis_client.aclose()

        print(
            "[REDIS] Connection closed."
        )

    finally:

        _redis_client = None


# ============================================================
# HEALTH CHECK
# ============================================================

async def redis_health() -> bool:

    try:

        client = get_redis()

        return bool(
            await client.ping()
        )

    except Exception:

        return False


# ============================================================
# SET JSON
# ============================================================

async def set_json(
    key: str,
    value: Any,
    ttl: int | None = None,
) -> None:

    client = get_redis()

    payload = json.dumps(
        value,
        default=str,
    )

    if ttl is not None:

        await client.set(
            key,
            payload,
            ex=ttl,
        )

    else:

        await client.set(
            key,
            payload,
        )


# ============================================================
# GET JSON
# ============================================================

async def get_json(
    key: str,
) -> Any | None:

    client = get_redis()

    payload = await client.get(
        key
    )

    if payload is None:
        return None

    try:

        return json.loads(
            payload
        )

    except json.JSONDecodeError:

        return payload


# ============================================================
# DELETE KEY
# ============================================================

async def delete_key(
    key: str,
) -> bool:

    client = get_redis()

    count = await client.delete(
        key
    )

    return bool(
        count
    )


# ============================================================
# CHECK KEY
# ============================================================

async def key_exists(
    key: str,
) -> bool:

    client = get_redis()

    result = await client.exists(
        key
    )

    return bool(
        result
    )


# ============================================================
# EXPIRE KEY
# ============================================================

async def expire_key(
    key: str,
    ttl: int,
) -> bool:

    client = get_redis()

    result = await client.expire(
        key,
        ttl,
    )

    return bool(
        result
    )


# ============================================================
# SESSION STATE CACHE
# ============================================================

async def cache_session_state(
    session_id: str,
    state: dict,
    ttl: int = 3600,
) -> None:

    key = (
        f"research:session:"
        f"{session_id}"
    )

    await set_json(
        key=key,
        value=state,
        ttl=ttl,
    )


async def get_cached_session_state(
    session_id: str,
) -> dict | None:

    key = (
        f"research:session:"
        f"{session_id}"
    )

    value = await get_json(
        key
    )

    if isinstance(
        value,
        dict,
    ):

        return value

    return None


async def delete_cached_session_state(
    session_id: str,
) -> bool:

    key = (
        f"research:session:"
        f"{session_id}"
    )

    return await delete_key(
        key
    )


# ============================================================
# RETRIEVAL CACHE
# ============================================================

async def cache_retrieval(
    tenant_id: str,
    query_hash: str,
    results: list[dict],
    ttl: int = 600,
) -> None:

    key = (
        f"retrieval:"
        f"{tenant_id}:"
        f"{query_hash}"
    )

    await set_json(
        key=key,
        value=results,
        ttl=ttl,
    )


async def get_cached_retrieval(
    tenant_id: str,
    query_hash: str,
) -> list[dict] | None:

    key = (
        f"retrieval:"
        f"{tenant_id}:"
        f"{query_hash}"
    )

    result = await get_json(
        key
    )

    if isinstance(
        result,
        list,
    ):

        return result

    return None


# ============================================================
# GENERIC CACHE
# ============================================================

async def cache_value(
    namespace: str,
    key: str,
    value: Any,
    ttl: int = 600,
) -> None:

    redis_key = (
        f"{namespace}:{key}"
    )

    await set_json(
        key=redis_key,
        value=value,
        ttl=ttl,
    )


async def get_cached_value(
    namespace: str,
    key: str,
) -> Any | None:

    redis_key = (
        f"{namespace}:{key}"
    )

    return await get_json(
        redis_key
    )


# ============================================================
# LIGHTWEIGHT LOCK
# ============================================================

async def acquire_lock(
    name: str,
    ttl: int = 60,
) -> bool:
    """
    Attempt to acquire a simple Redis lock.

    Returns True only if the lock was successfully acquired.
    """

    client = get_redis()

    key = (
        f"lock:{name}"
    )

    result = await client.set(
        key,
        "1",
        ex=ttl,
        nx=True,
    )

    return bool(
        result
    )


async def release_lock(
    name: str,
) -> None:

    client = get_redis()

    key = (
        f"lock:{name}"
    )

    await client.delete(
        key
    )