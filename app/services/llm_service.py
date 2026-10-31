# app/services/llm_service.py

from __future__ import annotations

import ast
import inspect
import json
import os
import re
from typing import Any

from groq import AsyncGroq, RateLimitError

from app.config import settings


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_FALLBACK_MODEL = ""

# Normal text generation
DEFAULT_TEXT_MAX_TOKENS = 800
MIN_TEXT_TOKENS = 128
MAX_TEXT_TOKENS = 1200

# Structured JSON generation
#
# IMPORTANT:
# 256 tokens was too small for Planner JSON and caused
# truncated responses such as:
#
# {"intent": "...", "research_plan": [...], "sub_queries": [
#     "How do RAG and ...
#
DEFAULT_JSON_MAX_TOKENS = 800
MIN_JSON_TOKENS = 700
MAX_JSON_TOKENS = 1200

# JSON repair
DEFAULT_JSON_REPAIR_MAX_TOKENS = 1000
MAX_JSON_REPAIR_TOKENS = 1400

# Prompt protection
DEFAULT_CONTEXT_MAX_CHARS = 8000

# GPT-OSS reasoning
DEFAULT_TEXT_REASONING = "medium"
DEFAULT_JSON_REASONING = "low"


# ============================================================
# GLOBAL CLIENT
# ============================================================

_groq_client: AsyncGroq | None = None


# ============================================================
# EXCEPTIONS
# ============================================================


class LLMServiceError(RuntimeError):
    """
    Base LLM exception.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str = "groq",
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code


class LLMRateLimitError(LLMServiceError):
    """
    Normalized Groq HTTP 429 error.
    """

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        quota_type: str = "unknown",
        model: str | None = None,
    ) -> None:
        super().__init__(
            message,
            provider="groq",
            model=model,
            status_code=429,
        )

        self.retry_after_seconds = retry_after_seconds
        self.quota_type = quota_type


class LLMCreditError(LLMServiceError):
    """
    Compatibility only.

    Existing main.py may still import this from the
    previous OpenRouter implementation.
    """

    pass


# ============================================================
# SETTINGS HELPERS
# ============================================================


def _setting(
    name: str,
    default: Any = None,
) -> Any:
    try:
        return getattr(
            settings,
            name,
            default,
        )
    except Exception:
        return default


def _string_value(
    value: Any,
) -> str:
    if value is None:
        return ""

    get_secret_value = getattr(
        value,
        "get_secret_value",
        None,
    )

    if callable(get_secret_value):
        try:
            return str(
                get_secret_value()
            ).strip()
        except Exception:
            pass

    return str(value).strip()


# ============================================================
# API KEY
# ============================================================


def _read_api_key() -> str:
    key = _string_value(
        _setting(
            "groq_api_key",
            "",
        )
    )

    if not key:
        key = (
            os.getenv(
                "GROQ_API_KEY",
                "",
            )
            .strip()
        )

    return key


def _get_api_key() -> str:
    key = _read_api_key()

    if not key:
        raise LLMServiceError(
            (
                "GROQ_API_KEY is missing. "
                "Add GROQ_API_KEY to your project .env."
            ),
            provider="groq",
        )

    return key


# ============================================================
# MODEL CONFIGURATION
# ============================================================


def _get_model() -> str:
    value = _string_value(
        _setting(
            "groq_model",
            DEFAULT_MODEL,
        )
    )

    if not value:
        value = (
            os.getenv(
                "GROQ_MODEL",
                "",
            )
            .strip()
        )

    return value or DEFAULT_MODEL


def _get_fallback_model() -> str:
    value = _string_value(
        _setting(
            "groq_fallback_model",
            "",
        )
    )

    if not value:
        value = (
            os.getenv(
                "GROQ_FALLBACK_MODEL",
                "",
            )
            .strip()
        )

    return (
        value
        or
        DEFAULT_FALLBACK_MODEL
    )


def _candidate_models(
    requested_model: str | None = None,
) -> list[str]:
    models: list[str] = []

    if requested_model:
        primary = requested_model.strip()
    else:
        primary = _get_model()

    fallback = _get_fallback_model()

    if primary:
        models.append(primary)

    if (
        fallback
        and
        fallback != primary
    ):
        models.append(fallback)

    return models


def _has_next_model(
    current_index: int,
    models: list[str],
) -> bool:
    return (
        current_index + 1
        <
        len(models)
    )


# ============================================================
# CLIENT
# ============================================================


def get_llm_client() -> AsyncGroq:
    global _groq_client

    if _groq_client is not None:
        return _groq_client

    try:
        timeout = float(
            _setting(
                "llm_request_timeout_seconds",
                180,
            )
        )
    except Exception:
        timeout = 180.0

    _groq_client = AsyncGroq(
        api_key=_get_api_key(),
        timeout=timeout,

        # Do not let SDK automatically retry 429 responses.
        # Our application handles fallback itself.
        max_retries=0,
    )

    return _groq_client


# ============================================================
# CLOSE CLIENT
# ============================================================


async def close_llm_client() -> None:
    global _groq_client

    client = _groq_client
    _groq_client = None

    if client is None:
        return

    close_method = getattr(
        client,
        "close",
        None,
    )

    if not callable(close_method):
        return

    try:
        result = close_method()

        if inspect.isawaitable(result):
            await result

    except Exception as exc:
        print(
            "[LLM] Groq client close warning:",
            repr(exc),
        )


# ============================================================
# STATUS CODE
# ============================================================


def _status_code(
    exc: Exception | None,
) -> int | None:
    if exc is None:
        return None

    status = getattr(
        exc,
        "status_code",
        None,
    )

    if status is not None:
        try:
            return int(status)
        except Exception:
            pass

    response = getattr(
        exc,
        "response",
        None,
    )

    if response is not None:
        status = getattr(
            response,
            "status_code",
            None,
        )

        if status is not None:
            try:
                return int(status)
            except Exception:
                pass

    return None


# ============================================================
# RATE LIMIT HELPERS
# ============================================================


def _retry_after(
    exc: Exception,
) -> float | None:
    response = getattr(
        exc,
        "response",
        None,
    )

    # --------------------------------------------------------
    # Headers
    # --------------------------------------------------------

    if response is not None:
        headers = getattr(
            response,
            "headers",
            None,
        )

        if headers:
            try:
                value = (
                    headers.get("retry-after")
                    or
                    headers.get("Retry-After")
                )
            except Exception:
                value = None

            if value:
                try:
                    return float(value)
                except Exception:
                    pass

    # --------------------------------------------------------
    # Provider message
    # --------------------------------------------------------

    text = str(exc)

    patterns = [
        r"try again in\s+([0-9.]+)\s*s",
        r"try again in\s+([0-9.]+)\s*seconds?",
        r"retry after\s+([0-9.]+)",
        r"retry-after['\"]?\s*[:=]\s*['\"]?([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        try:
            return float(
                match.group(1)
            )
        except Exception:
            pass

    return None


def _quota_type(
    message: str,
) -> str:
    text = (
        message
        or ""
    ).lower()

    if (
        "tokens per day" in text
        or "tpd" in text
    ):
        return "TPD"

    if (
        "input tokens per minute" in text
        or "itpm" in text
    ):
        return "ITPM"

    if (
        "output tokens per minute" in text
        or "otpm" in text
    ):
        return "OTPM"

    if (
        "tokens per minute" in text
        or "tpm" in text
    ):
        return "TPM"

    if (
        "requests per day" in text
        or "rpd" in text
    ):
        return "RPD"

    if (
        "requests per minute" in text
        or "rpm" in text
    ):
        return "RPM"

    return "unknown"


def convert_groq_rate_limit(
    exc: Exception,
    *,
    model: str | None = None,
) -> LLMRateLimitError:
    message = str(exc)

    return LLMRateLimitError(
        message=message,
        retry_after_seconds=_retry_after(
            exc
        ),
        quota_type=_quota_type(
            message
        ),
        model=model,
    )


# Compatibility with main.py from earlier OpenRouter migration.
def convert_openrouter_rate_limit(
    exc: Exception,
    *,
    model: str | None = None,
) -> LLMRateLimitError:
    return convert_groq_rate_limit(
        exc,
        model=model,
    )


# ============================================================
# LOGGING
# ============================================================


def _log_rate_limit(
    error: LLMRateLimitError,
) -> None:
    print()
    print(
        "[LLM] Groq rate limit reached."
    )

    print(
        "[LLM] Model:",
        error.model,
    )

    print(
        "[LLM] Quota type:",
        error.quota_type,
    )

    print(
        "[LLM] Retry after:",
        error.retry_after_seconds,
    )


def _log_model_fallback(
    current_model: str,
    next_model: str,
) -> None:
    print()

    print(
        (
            "[LLM] Switching fallback model: "
            f"{current_model} -> {next_model}"
        )
    )


# ============================================================
# PROMPT LIMIT
# ============================================================


def _context_max_chars() -> int:
    value = _setting(
        "llm_context_max_chars",
        DEFAULT_CONTEXT_MAX_CHARS,
    )

    try:
        value = int(value)
    except Exception:
        value = DEFAULT_CONTEXT_MAX_CHARS

    return max(
        2000,
        value,
    )


def _limit_prompt(
    text: str,
) -> str:
    value = (
        text
        or ""
    ).strip()

    limit = (
        _context_max_chars()
    )

    if len(value) <= limit:
        return value

    head_size = int(
        limit * 0.70
    )

    tail_size = (
        limit
        -
        head_size
    )

    return (
        value[:head_size]
        +
        "\n\n"
        "[... context truncated to control token usage ...]"
        "\n\n"
        +
        value[-tail_size:]
    )


# ============================================================
# TOKEN BUDGET HELPERS
# ============================================================


def _requested_token_budget(
    max_completion_tokens: int | None,
    max_tokens: int | None,
    default: int,
) -> int:
    value = (
        max_completion_tokens
        if max_completion_tokens is not None
        else
        max_tokens
    )

    if value is None:
        value = _setting(
            "llm_max_completion_tokens",
            default,
        )

    try:
        return int(value)
    except Exception:
        return default


def _text_token_budget(
    max_completion_tokens: int | None,
    max_tokens: int | None,
) -> int:
    requested = (
        _requested_token_budget(
            max_completion_tokens,
            max_tokens,
            DEFAULT_TEXT_MAX_TOKENS,
        )
    )

    return max(
        MIN_TEXT_TOKENS,
        min(
            requested,
            MAX_TEXT_TOKENS,
        ),
    )


def _json_token_budget(
    max_completion_tokens: int | None,
    max_tokens: int | None,
) -> int:
    """
    Never allow planner / claim JSON to drop below
    MIN_JSON_TOKENS.

    Therefore a caller asking for max_tokens=256 becomes 700.
    """

    requested = (
        _requested_token_budget(
            max_completion_tokens,
            max_tokens,
            DEFAULT_JSON_MAX_TOKENS,
        )
    )

    return max(
        MIN_JSON_TOKENS,
        min(
            requested,
            MAX_JSON_TOKENS,
        ),
    )


# ============================================================
# RESPONSE CONTENT
# ============================================================


def _extract_content(
    response: Any,
) -> str:
    choices = getattr(
        response,
        "choices",
        None,
    )

    if not choices:
        return ""

    message = getattr(
        choices[0],
        "message",
        None,
    )

    if message is None:
        return ""

    content = getattr(
        message,
        "content",
        None,
    )

    if content is None:
        return ""

    if isinstance(
        content,
        str,
    ):
        return content.strip()

    # --------------------------------------------------------
    # Content parts compatibility
    # --------------------------------------------------------

    if isinstance(
        content,
        list,
    ):
        parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                parts.append(item)
                continue

            if isinstance(
                item,
                dict,
            ):
                value = (
                    item.get("text")
                    or
                    item.get("content")
                )

                if value:
                    parts.append(
                        str(value)
                    )

                continue

            value = getattr(
                item,
                "text",
                None,
            )

            if value:
                parts.append(
                    str(value)
                )

        return "\n".join(
            parts
        ).strip()

    return str(
        content
    ).strip()


# ============================================================
# SDK SIGNATURE
# ============================================================


def _supported_parameters(
    function: Any,
) -> set[str]:
    try:
        signature = inspect.signature(
            function
        )

        return set(
            signature.parameters.keys()
        )
    except Exception:
        return set()


# ============================================================
# GROQ COMPLETION
# ============================================================


async def _chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    response_format: dict[str, Any] | None = None,
) -> Any:
    client = get_llm_client()

    create_method = (
        client
        .chat
        .completions
        .create
    )

    supported = (
        _supported_parameters(
            create_method
        )
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    # ========================================================
    # TOKEN PARAMETER
    #
    # User's currently installed Groq SDK expects max_tokens.
    # Newer SDK versions may expose max_completion_tokens.
    # ========================================================

    if (
        "max_completion_tokens"
        in supported
    ):
        kwargs[
            "max_completion_tokens"
        ] = max_completion_tokens

        token_parameter = (
            "max_completion_tokens"
        )

    else:
        kwargs[
            "max_tokens"
        ] = max_completion_tokens

        token_parameter = (
            "max_tokens"
        )

    # ========================================================
    # TEMPERATURE
    # ========================================================

    if temperature is not None:
        kwargs[
            "temperature"
        ] = temperature

    # ========================================================
    # GPT-OSS REASONING
    # ========================================================

    if reasoning_effort:
        effort = (
            str(reasoning_effort)
            .strip()
            .lower()
        )

        if effort not in {
            "low",
            "medium",
            "high",
        }:
            effort = "low"

        if (
            "reasoning_effort"
            in supported
        ):
            kwargs[
                "reasoning_effort"
            ] = effort

    # ========================================================
    # STRUCTURED OUTPUT
    # ========================================================

    if response_format is not None:
        if (
            not supported
            or
            "response_format"
            in supported
        ):
            kwargs[
                "response_format"
            ] = response_format

    print(
        "[LLM] Groq token parameter:",
        token_parameter,
    )

    print(
        "[LLM] Max output tokens:",
        max_completion_tokens,
    )

    # ========================================================
    # REQUEST
    # ========================================================

    try:
        return await create_method(
            **kwargs
        )

    # ========================================================
    # SDK COMPATIBILITY FALLBACKS
    # ========================================================

    except TypeError as exc:
        error_text = (
            str(exc)
            .lower()
        )

        # ----------------------------------------------------
        # max_completion_tokens unsupported
        # ----------------------------------------------------

        if (
            "max_completion_tokens"
            in error_text
        ):
            kwargs.pop(
                "max_completion_tokens",
                None,
            )

            kwargs[
                "max_tokens"
            ] = max_completion_tokens

            print(
                (
                    "[LLM] Groq SDK does not support "
                    "max_completion_tokens. "
                    "Retrying with max_tokens."
                )
            )

            return await create_method(
                **kwargs
            )

        # ----------------------------------------------------
        # max_tokens unsupported
        # ----------------------------------------------------

        if (
            "max_tokens"
            in error_text
        ):
            kwargs.pop(
                "max_tokens",
                None,
            )

            kwargs[
                "max_completion_tokens"
            ] = max_completion_tokens

            print(
                (
                    "[LLM] Groq SDK does not support "
                    "max_tokens. Retrying with "
                    "max_completion_tokens."
                )
            )

            return await create_method(
                **kwargs
            )

        # ----------------------------------------------------
        # reasoning_effort unsupported
        # ----------------------------------------------------

        if (
            "reasoning_effort"
            in error_text
        ):
            kwargs.pop(
                "reasoning_effort",
                None,
            )

            print(
                (
                    "[LLM] Groq SDK does not support "
                    "reasoning_effort. Retrying without it."
                )
            )

            return await create_method(
                **kwargs
            )

        # ----------------------------------------------------
        # response_format unsupported
        # ----------------------------------------------------

        if (
            "response_format"
            in error_text
        ):
            kwargs.pop(
                "response_format",
                None,
            )

            print(
                (
                    "[LLM] Groq SDK does not support "
                    "response_format. Retrying without it."
                )
            )

            return await create_method(
                **kwargs
            )

        raise


# ============================================================
# TEXT MESSAGES
# ============================================================


def _text_messages(
    *,
    user_prompt: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                system_prompt
                or
                (
                    "You are a precise research assistant. "
                    "Use supplied evidence when available. "
                    "Do not invent unsupported factual claims. "
                    "Be concise."
                )
            ),
        },
        {
            "role": "user",
            "content": _limit_prompt(
                user_prompt
            ),
        },
    ]


# ============================================================
# JSON MESSAGES
# ============================================================


def _json_messages(
    *,
    user_prompt: str,
    system_prompt: str | None,
    json_schema: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    base = (
        system_prompt
        or
        (
            "You are a precise structured-data "
            "generation assistant."
        )
    )

    rules = """
Return exactly one complete JSON object.

Rules:
- Output JSON only.
- Do not output Markdown.
- Do not use code fences.
- Do not add explanations.
- Use double quotes for keys.
- Use double quotes for string values.
- Do not use trailing commas.
- Close every array.
- Close every object.
- Follow the requested structure exactly.
- Keep values concise where possible.
"""

    schema_text = ""

    if json_schema is not None:
        try:
            schema_text = (
                "\n\nRequired JSON Schema:\n"
                +
                json.dumps(
                    json_schema,
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                )
            )
        except Exception:
            schema_text = ""

    return [
        {
            "role": "system",
            "content": (
                base
                +
                "\n\n"
                +
                rules.strip()
                +
                schema_text
            ),
        },
        {
            "role": "user",
            "content": (
                _limit_prompt(
                    user_prompt
                )
                +
                "\n\n"
                "Return the complete JSON object now."
            ),
        },
    ]


# ============================================================
# JSON CLEANING
# ============================================================


def _strip_code_fences(
    content: str,
) -> str:
    text = (
        content
        or ""
    ).strip()

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    return text.strip()


# ============================================================
# BALANCED JSON EXTRACTION
# ============================================================


def _extract_balanced_json(
    content: str,
) -> str | None:
    text = (
        content
        or ""
    )

    start: int | None = None

    for index, char in enumerate(
        text
    ):
        if char in {
            "{",
            "[",
        }:
            start = index
            break

    if start is None:
        return None

    pairs = {
        "{": "}",
        "[": "]",
    }

    stack: list[str] = []

    inside_string = False
    escaped = False

    for index in range(
        start,
        len(text),
    ):
        char = text[index]

        if inside_string:
            if escaped:
                escaped = False
                continue

            if char == "\\":
                escaped = True
                continue

            if char == '"':
                inside_string = False

            continue

        if char == '"':
            inside_string = True
            continue

        if char in pairs:
            stack.append(char)
            continue

        if char in {
            "}",
            "]",
        }:
            if not stack:
                continue

            expected = pairs[
                stack[-1]
            ]

            if char != expected:
                return None

            stack.pop()

            if not stack:
                return text[
                    start:
                    index + 1
                ]

    return None


# ============================================================
# JSON PARSER
# ============================================================


def _parse_json(
    content: str,
) -> Any:
    cleaned = (
        _strip_code_fences(
            content
        )
    )

    if not cleaned:
        raise ValueError(
            (
                "GPT-OSS returned an empty "
                "JSON response."
            )
        )

    # ========================================================
    # DIRECT JSON
    # ========================================================

    try:
        return json.loads(
            cleaned
        )
    except json.JSONDecodeError:
        pass

    # ========================================================
    # EXTRACT EMBEDDED JSON
    # ========================================================

    candidate = (
        _extract_balanced_json(
            cleaned
        )
        or
        cleaned
    )

    try:
        return json.loads(
            candidate
        )
    except json.JSONDecodeError:
        pass

    # ========================================================
    # TRAILING COMMA REPAIR
    # ========================================================

    repaired = re.sub(
        r",\s*([}\]])",
        r"\1",
        candidate,
    )

    try:
        return json.loads(
            repaired
        )
    except json.JSONDecodeError:
        pass

    # ========================================================
    # PYTHON DICT/LIST FALLBACK
    # ========================================================

    try:
        value = ast.literal_eval(
            repaired
        )

        if isinstance(
            value,
            (
                dict,
                list,
            ),
        ):
            return value

    except Exception:
        pass

    raise ValueError(
        (
            "Unable to parse GPT-OSS output as JSON. "
            f"Preview: {cleaned[:500]!r}"
        )
    )


# ============================================================
# JSON SCHEMA FORMAT
# ============================================================


def _json_schema_format(
    json_schema: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    return {
        "type": "json_schema",

        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
        },
    }


# ============================================================
# GENERATE TEXT
# ============================================================


async def generate_text(
    user_prompt: str | None = None,
    *,
    prompt: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    reasoning_effort: str | None = DEFAULT_TEXT_REASONING,
    **_: Any,
) -> str:
    final_prompt = (
        user_prompt
        if user_prompt is not None
        else
        prompt
    )

    final_prompt = str(
        final_prompt
        or ""
    ).strip()

    if not final_prompt:
        raise ValueError(
            (
                "generate_text requires "
                "a non-empty user_prompt."
            )
        )

    if temperature is None:
        try:
            temperature = float(
                _setting(
                    "llm_temperature",
                    0.2,
                )
            )
        except Exception:
            temperature = 0.2

    token_budget = (
        _text_token_budget(
            max_completion_tokens,
            max_tokens,
        )
    )

    messages = (
        _text_messages(
            user_prompt=final_prompt,
            system_prompt=system_prompt,
        )
    )

    models = (
        _candidate_models(
            model
        )
    )

    last_error: Exception | None = None

    for index, model_name in enumerate(
        models
    ):
        print()
        print(
            (
                "[LLM] Groq | text | "
                f"{model_name}"
            )
        )

        try:
            response = (
                await _chat_completion(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=(
                        token_budget
                    ),
                    temperature=temperature,
                    reasoning_effort=(
                        reasoning_effort
                    ),
                    response_format=None,
                )
            )

            result = (
                _extract_content(
                    response
                )
            )

            if not result:
                raise LLMServiceError(
                    (
                        "GPT-OSS returned an "
                        "empty text response."
                    ),
                    model=model_name,
                )

            print(
                "[LLM] Text generation succeeded."
            )

            return result

        # ====================================================
        # RATE LIMIT
        # ====================================================

        except RateLimitError as exc:
            converted = (
                convert_groq_rate_limit(
                    exc,
                    model=model_name,
                )
            )

            _log_rate_limit(
                converted
            )

            last_error = converted

            if _has_next_model(
                index,
                models,
            ):
                _log_model_fallback(
                    model_name,
                    models[
                        index + 1
                    ],
                )

                continue

            raise converted from exc

        except LLMRateLimitError as exc:
            last_error = exc

            if _has_next_model(
                index,
                models,
            ):
                _log_model_fallback(
                    model_name,
                    models[
                        index + 1
                    ],
                )

                continue

            raise

        # ====================================================
        # OTHER MODEL ERROR
        # ====================================================

        except Exception as exc:
            last_error = exc

            print(
                (
                    "[LLM] Text generation failed | "
                    f"{model_name} | {exc}"
                )
            )

            if _has_next_model(
                index,
                models,
            ):
                _log_model_fallback(
                    model_name,
                    models[
                        index + 1
                    ],
                )

                continue

    if isinstance(
        last_error,
        LLMRateLimitError,
    ):
        raise last_error

    raise LLMServiceError(
        (
            "Groq GPT-OSS text generation failed: "
            f"{last_error}"
        ),
        model=(
            models[-1]
            if models
            else
            _get_model()
        ),
        status_code=(
            _status_code(
                last_error
            )
        ),
    ) from last_error


# ============================================================
# GENERATE JSON
# ============================================================


async def generate_json(
    user_prompt: str | None = None,
    *,
    prompt: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = 0.0,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    reasoning_effort: str | None = DEFAULT_JSON_REASONING,
    json_schema: dict[str, Any] | None = None,
    schema_name: str = "research_response",
    **_: Any,
) -> Any:
    """
    GPT-OSS JSON strategy:

    1. Minimum output budget is 700 tokens.
    2. If a real schema exists, try strict schema mode.
    3. Otherwise use normal completion with JSON instructions.
    4. Parse JSON locally.
    5. Repair malformed/truncated JSON when useful.
    6. If primary model is rate-limited, try fallback model.
    """

    final_prompt = (
        user_prompt
        if user_prompt is not None
        else
        prompt
    )

    final_prompt = str(
        final_prompt
        or ""
    ).strip()

    if not final_prompt:
        raise ValueError(
            (
                "generate_json requires "
                "a non-empty user_prompt."
            )
        )

    # ========================================================
    # JSON OUTPUT BUDGET
    #
    # A caller requesting 256 will receive at least 700.
    # ========================================================

    token_budget = (
        _json_token_budget(
            max_completion_tokens,
            max_tokens,
        )
    )

    models = (
        _candidate_models(
            model
        )
    )

    messages = (
        _json_messages(
            user_prompt=final_prompt,
            system_prompt=system_prompt,
            json_schema=json_schema,
        )
    )

    last_error: Exception | None = None

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for index, model_name in enumerate(
        models
    ):
        print()
        print(
            (
                "[LLM] Groq | json | "
                f"{model_name}"
            )
        )

        # ====================================================
        # STRICT JSON SCHEMA
        # ====================================================

        if json_schema is not None:
            try:
                print(
                    (
                        "[LLM] Trying strict "
                        "JSON Schema."
                    )
                )

                response = (
                    await _chat_completion(
                        model=model_name,
                        messages=messages,
                        max_completion_tokens=(
                            token_budget
                        ),
                        temperature=temperature,
                        reasoning_effort=(
                            reasoning_effort
                        ),
                        response_format=(
                            _json_schema_format(
                                json_schema,
                                schema_name,
                            )
                        ),
                    )
                )

                content = (
                    _extract_content(
                        response
                    )
                )

                result = (
                    _parse_json(
                        content
                    )
                )

                print(
                    (
                        "[LLM] Strict JSON "
                        "Schema succeeded."
                    )
                )

                return result

            # ------------------------------------------------
            # RATE LIMIT -> NEXT MODEL
            # ------------------------------------------------

            except RateLimitError as exc:
                converted = (
                    convert_groq_rate_limit(
                        exc,
                        model=model_name,
                    )
                )

                _log_rate_limit(
                    converted
                )

                last_error = converted

                if _has_next_model(
                    index,
                    models,
                ):
                    _log_model_fallback(
                        model_name,
                        models[
                            index + 1
                        ],
                    )

                    continue

                raise converted from exc

            except LLMRateLimitError as exc:
                last_error = exc

                if _has_next_model(
                    index,
                    models,
                ):
                    _log_model_fallback(
                        model_name,
                        models[
                            index + 1
                        ],
                    )

                    continue

                raise

            # ------------------------------------------------
            # Schema mode failed.
            # Use prompt-guided JSON on same model.
            # ------------------------------------------------

            except Exception as exc:
                last_error = exc

                print(
                    (
                        "[LLM] Strict JSON Schema "
                        "failed | "
                        f"{model_name} | {exc}"
                    )
                )

                print(
                    (
                        "[LLM] Falling back to "
                        "prompt-guided JSON."
                    )
                )

        # ====================================================
        # PROMPT-GUIDED JSON
        # ====================================================

        content = ""

        try:
            print(
                (
                    "[LLM] Using prompt-guided "
                    "JSON generation."
                )
            )

            response = (
                await _chat_completion(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=(
                        token_budget
                    ),
                    temperature=0.0,
                    reasoning_effort=(
                        reasoning_effort
                    ),

                    # Critical:
                    #
                    # Generic json_object previously generated
                    # Groq json_validate_failed HTTP 400 errors.
                    response_format=None,
                )
            )

            content = (
                _extract_content(
                    response
                )
            )

            result = (
                _parse_json(
                    content
                )
            )

            print(
                "[LLM] JSON generation succeeded."
            )

            return result

        # ====================================================
        # RATE LIMIT -> NEXT MODEL
        # ====================================================

        except RateLimitError as exc:
            converted = (
                convert_groq_rate_limit(
                    exc,
                    model=model_name,
                )
            )

            _log_rate_limit(
                converted
            )

            last_error = converted

            if _has_next_model(
                index,
                models,
            ):
                _log_model_fallback(
                    model_name,
                    models[
                        index + 1
                    ],
                )

                continue

            raise converted from exc

        except LLMRateLimitError as exc:
            last_error = exc

            if _has_next_model(
                index,
                models,
            ):
                _log_model_fallback(
                    model_name,
                    models[
                        index + 1
                    ],
                )

                continue

            raise

        # ====================================================
        # JSON PARSE / GENERATION FAILURE
        # ====================================================

        except Exception as exc:
            last_error = exc

            print(
                (
                    "[LLM] JSON generation/parse "
                    "failed | "
                    f"{model_name} | {exc}"
                )
            )

        # ====================================================
        # JSON REPAIR
        #
        # Only perform repair if the model actually produced
        # content. If request failed before producing output,
        # don't waste another call.
        # ====================================================

        if content:
            repair_budget = min(
                max(
                    token_budget,
                    DEFAULT_JSON_REPAIR_MAX_TOKENS,
                ),
                MAX_JSON_REPAIR_TOKENS,
            )

            repair_prompt = f"""
Repair the following response into one complete valid JSON object.

ORIGINAL TASK:
{_limit_prompt(final_prompt)}

BROKEN JSON:
{content[:5000]}

Rules:
- Return JSON only.
- No Markdown.
- No code fences.
- No explanation.
- Use double quotes.
- No trailing commas.
- Preserve the intended fields and values.
- Finish any truncated arrays or objects.
- Close every array.
- Close every object.
"""

            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You repair malformed or truncated JSON. "
                        "Return exactly one complete valid JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": repair_prompt,
                },
            ]

            try:
                print(
                    "[LLM] Running JSON repair."
                )

                response = (
                    await _chat_completion(
                        model=model_name,
                        messages=repair_messages,
                        max_completion_tokens=(
                            repair_budget
                        ),
                        temperature=0.0,
                        reasoning_effort="low",
                        response_format=None,
                    )
                )

                repaired_content = (
                    _extract_content(
                        response
                    )
                )

                result = (
                    _parse_json(
                        repaired_content
                    )
                )

                print(
                    "[LLM] JSON repair succeeded."
                )

                return result

            # --------------------------------------------
            # Repair rate limited -> try next model
            # --------------------------------------------

            except RateLimitError as exc:
                converted = (
                    convert_groq_rate_limit(
                        exc,
                        model=model_name,
                    )
                )

                _log_rate_limit(
                    converted
                )

                last_error = converted

                if _has_next_model(
                    index,
                    models,
                ):
                    _log_model_fallback(
                        model_name,
                        models[
                            index + 1
                        ],
                    )

                    continue

                raise converted from exc

            except LLMRateLimitError as exc:
                last_error = exc

                if _has_next_model(
                    index,
                    models,
                ):
                    _log_model_fallback(
                        model_name,
                        models[
                            index + 1
                        ],
                    )

                    continue

                raise

            except Exception as exc:
                last_error = exc

                print(
                    (
                        "[LLM] JSON repair failed | "
                        f"{model_name} | {exc}"
                    )
                )

        # ====================================================
        # CURRENT MODEL FAILED
        # TRY FALLBACK IF AVAILABLE
        # ====================================================

        if _has_next_model(
            index,
            models,
        ):
            _log_model_fallback(
                model_name,
                models[
                    index + 1
                ],
            )

            continue

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    if isinstance(
        last_error,
        LLMRateLimitError,
    ):
        raise last_error

    raise LLMServiceError(
        (
            "Groq GPT-OSS JSON generation failed "
            "after generation, repair and fallback: "
            f"{last_error}"
        ),
        model=(
            models[-1]
            if models
            else
            _get_model()
        ),
        status_code=(
            _status_code(
                last_error
            )
        ),
    ) from last_error


# ============================================================
# GENERIC GENERATE
# ============================================================


async def generate(
    user_prompt: str | None = None,
    *,
    prompt: str | None = None,
    system_prompt: str | None = None,
    json_mode: bool = False,
    json_schema: dict[str, Any] | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_completion_tokens: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    **kwargs: Any,
) -> Any:
    final_prompt = (
        user_prompt
        if user_prompt is not None
        else
        prompt
    )

    if (
        json_mode
        or
        json_schema is not None
    ):
        return await generate_json(
            user_prompt=final_prompt,
            system_prompt=system_prompt,
            json_schema=json_schema,
            model=model,
            reasoning_effort=(
                reasoning_effort
                or
                DEFAULT_JSON_REASONING
            ),
            max_completion_tokens=(
                max_completion_tokens
            ),
            max_tokens=max_tokens,
            temperature=(
                0.0
                if temperature is None
                else
                temperature
            ),
            **kwargs,
        )

    return await generate_text(
        user_prompt=final_prompt,
        system_prompt=system_prompt,
        model=model,
        reasoning_effort=(
            reasoning_effort
            or
            DEFAULT_TEXT_REASONING
        ),
        max_completion_tokens=(
            max_completion_tokens
        ),
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )


# ============================================================
# INFO
# ============================================================


def llm_info() -> dict[str, Any]:
    primary = _get_model()
    fallback = _get_fallback_model()

    return {
        "provider":
            "groq",

        "model":
            primary,

        "fallback_model":
            fallback or None,

        "fallback_enabled":
            bool(fallback),

        "fallback_on_rate_limit":
            bool(fallback),

        "api_key_loaded":
            bool(
                _read_api_key()
            ),

        "json_strategy":
            (
                "strict_schema_when_provided_"
                "otherwise_prompt_guided_json"
            ),

        "local_json_validation":
            True,

        "json_repair":
            True,

        "minimum_json_tokens":
            MIN_JSON_TOKENS,

        "default_json_tokens":
            DEFAULT_JSON_MAX_TOKENS,

        "maximum_json_tokens":
            MAX_JSON_TOKENS,

        "text_reasoning":
            DEFAULT_TEXT_REASONING,

        "json_reasoning":
            DEFAULT_JSON_REASONING,

        "context_max_chars":
            _context_max_chars(),
    }