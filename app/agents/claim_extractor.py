# app/agents/claim_extractor.py

from __future__ import annotations

import inspect
import re
from typing import Any

from app.services.llm_service import (
    generate_json,
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_CLAIMS = 20
MIN_CLAIM_LENGTH = 8


# ============================================================
# MAYBE AWAIT
# ============================================================


async def _maybe_await(
    value: Any,
) -> Any:

    if inspect.isawaitable(
        value
    ):
        return await value

    return value


# ============================================================
# BASIC TEXT CLEANING
# ============================================================


def _clean_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    text = str(
        value
    )

    text = (
        text
        .replace(
            "\r",
            " ",
        )
        .replace(
            "\n",
            " ",
        )
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # --------------------------------------------------------
    # BULLETS
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*[-*•]\s*",
        "",
        text,
    )

    # --------------------------------------------------------
    # NUMBERED LIST
    # --------------------------------------------------------

    text = re.sub(
        r"^\s*\d+[\.\)]\s*",
        "",
        text,
    )

    return text.strip()


# ============================================================
# EXTRACT CLAIM TEXT
# ============================================================


def extract_claim_text(
    value: Any,
) -> str:

    if value is None:
        return ""

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(
        value,
        str,
    ):

        return _clean_text(
            value
        )

    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(
        value,
        dict,
    ):

        for key in (
            "claim",
            "text",
            "statement",
            "sentence",
            "fact",
            "content",
        ):

            candidate = (
                value.get(
                    key
                )
            )

            if candidate is None:
                continue

            text = (
                _clean_text(
                    candidate
                )
            )

            if text:
                return text

    return ""


# ============================================================
# UNWRAP CLAIM RESPONSE
# ============================================================


def unwrap_claims(
    result: Any,
) -> list[Any]:

    if result is None:
        return []

    # --------------------------------------------------------
    # DIRECT LIST
    # --------------------------------------------------------

    if isinstance(
        result,
        list,
    ):

        return result

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(
        result,
        str,
    ):

        text = (
            result.strip()
        )

        return (
            [text]
            if text
            else
            []
        )

    # --------------------------------------------------------
    # DICT
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        for key in (
            "claims",
            "atomic_claims",
            "items",
            "statements",
            "facts",
            "results",
        ):

            value = (
                result.get(
                    key
                )
            )

            if isinstance(
                value,
                list,
            ):

                return value

            if (
                isinstance(
                    value,
                    str,
                )
                and
                value.strip()
            ):

                return [
                    value.strip()
                ]

        # ----------------------------------------------------
        # MAYBE DICT ITSELF IS ONE CLAIM
        # ----------------------------------------------------

        single = (
            extract_claim_text(
                result
            )
        )

        if single:

            return [
                result
            ]

    return []


# ============================================================
# NON FACTUAL FILTER
# ============================================================


def _is_non_factual(
    text: str,
) -> bool:

    if not text:
        return True

    lowered = (
        text
        .lower()
        .strip()
    )

    if len(
        lowered
    ) < MIN_CLAIM_LENGTH:

        return True

    blocked_exact = {
        "introduction",
        "summary",
        "conclusion",
        "overview",
        "key points",
        "key characteristics",
        "benefits",
        "sources",
        "evidence",
        "answer",
    }

    if lowered in blocked_exact:

        return True

    blocked_prefixes = (
        "the answer says",
        "the response says",
        "according to the answer",
        "the provided answer",
        "this answer",
        "the assistant",
    )

    if lowered.startswith(
        blocked_prefixes
    ):

        return True

    return False


# ============================================================
# DEDUPE KEY
# ============================================================


def _dedupe_key(
    text: str,
) -> str:

    normalized = (
        text.lower()
    )

    normalized = re.sub(
        r"[^\w\s]",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


# ============================================================
# NORMALIZE CLAIMS
# ============================================================


def normalize_claims(
    result: Any,
) -> list[dict[str, Any]]:

    raw_claims = (
        unwrap_claims(
            result
        )
    )

    normalized: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen: set[str] = set()

    for raw_claim in raw_claims:

        text = (
            extract_claim_text(
                raw_claim
            )
        )

        if (
            not text
            or
            _is_non_factual(
                text
            )
        ):

            continue

        key = (
            _dedupe_key(
                text
            )
        )

        if (
            not key
            or
            key in seen
        ):

            continue

        seen.add(
            key
        )

        item: dict[
            str,
            Any,
        ] = {

            "id":
                f"claim_{len(normalized) + 1}",

            "claim":
                text,

            "text":
                text,
        }

        # ----------------------------------------------------
        # PRESERVE OPTIONAL METADATA
        # ----------------------------------------------------

        if isinstance(
            raw_claim,
            dict,
        ):

            for optional_key in (
                "source",
                "citation",
                "importance",
                "type",
            ):

                value = (
                    raw_claim.get(
                        optional_key
                    )
                )

                if value is not None:

                    item[
                        optional_key
                    ] = value

        normalized.append(
            item
        )

        if len(
            normalized
        ) >= MAX_CLAIMS:

            break

    return normalized


# ============================================================
# FALLBACK CLAIM EXTRACTION
# ============================================================


def _fallback_claims_from_answer(
    answer: str,
) -> list[dict[str, Any]]:

    answer = (
        answer
        or
        ""
    ).strip()

    if not answer:
        return []

    # --------------------------------------------------------
    # REMOVE MARKDOWN HEADINGS
    # --------------------------------------------------------

    answer = re.sub(
        r"(?m)^#{1,6}\s+",
        "",
        answer,
    )

    # --------------------------------------------------------
    # SPLIT SENTENCES / LINES
    # --------------------------------------------------------

    parts = re.split(
        r"(?<=[.!?])\s+|\n+",
        answer,
    )

    claims: list[
        dict[
            str,
            Any,
        ]
    ] = []

    seen: set[str] = set()

    for part in parts:

        text = (
            _clean_text(
                part
            )
        )

        text = (
            text
            .replace(
                "|",
                " ",
            )
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if (
            not text
            or
            _is_non_factual(
                text
            )
        ):

            continue

        key = (
            _dedupe_key(
                text
            )
        )

        if (
            not key
            or
            key in seen
        ):

            continue

        seen.add(
            key
        )

        claims.append(
            {

                "id":
                    f"claim_{len(claims) + 1}",

                "claim":
                    text,

                "text":
                    text,
            }
        )

        if len(
            claims
        ) >= MAX_CLAIMS:

            break

    return claims


# ============================================================
# BUILD CLAIM PROMPT
# ============================================================


def _build_claim_prompt(
    answer: str,
) -> str:

    return f"""
You are the claim-extraction stage of an evidence-grounded
Retrieval-Augmented Generation system.

GENERATED ANSWER:

{answer}


TASK:

Extract ATOMIC factual claims from the generated answer.


STRICT RULES:

1. Extract factual claims only.

2. Each claim must contain ONE independently verifiable fact.

3. Do not combine multiple benefits, properties, effects,
   actions, or conclusions in one claim.

BAD:

"RAG improves accuracy, reduces hallucinations,
and reduces retraining costs."

GOOD:

"RAG can improve answer accuracy."

"RAG can help reduce hallucinations."

"RAG can reduce the need for model retraining."


4. Split compound factual sentences where necessary.

5. Preserve uncertainty and qualification exactly.

Words such as:

can
may
helps
typically
often
reduces
supports

must not be strengthened.


6. Do not convert:

"reduces the need for retraining"

into:

"does not require retraining."


7. Do not invent facts.

8. Do not use outside knowledge.

9. Keep subjects explicit.

BAD:

"It improves accuracy."

GOOD:

"RAG can improve answer accuracy."


10. Do not extract:

- headings
- formatting
- table labels
- citations by themselves
- opinions
- instructions
- duplicate statements


11. Keep each claim short and atomic.

12. Return STRICT JSON only.


EXPECTED JSON FORMAT:

{{
  "claims": [
    "First atomic factual claim.",
    "Second atomic factual claim."
  ]
}}


If no factual claims exist:

{{
  "claims": []
}}
""".strip()


# ============================================================
# GENERATE JSON COMPATIBILITY WRAPPER
# ============================================================


async def _generate_claim_json(
    prompt: str,
) -> Any:
    """
    Compatibility wrapper around llm_service.generate_json.

    Supports signatures such as:

        generate_json(user_prompt=...)
        generate_json(prompt=...)
        generate_json(<positional prompt>)

    This prevents:
        unexpected keyword argument 'prompt'
    """

    signature = (
        inspect.signature(
            generate_json
        )
    )

    parameters = (
        signature.parameters
    )

    kwargs: dict[
        str,
        Any,
    ] = {}

    # ========================================================
    # PROMPT PARAMETER
    # ========================================================

    if (
        "user_prompt"
        in parameters
    ):

        kwargs[
            "user_prompt"
        ] = prompt

    elif (
        "prompt"
        in parameters
    ):

        kwargs[
            "prompt"
        ] = prompt

    else:

        # ----------------------------------------------------
        # FINAL COMPATIBILITY FALLBACK
        # ----------------------------------------------------

        result = (
            generate_json(
                prompt
            )
        )

        return await _maybe_await(
            result
        )

    # ========================================================
    # OPTIONAL SYSTEM PROMPT
    # ========================================================

    if (
        "system_prompt"
        in parameters
    ):

        kwargs[
            "system_prompt"
        ] = (
            "Extract atomic factual claims from the supplied "
            "answer. Return valid JSON only."
        )

    # ========================================================
    # TEMPERATURE
    # ========================================================

    if (
        "temperature"
        in parameters
    ):

        kwargs[
            "temperature"
        ] = 0.0

    # ========================================================
    # MAX TOKENS
    # ========================================================

    if (
        "max_completion_tokens"
        in parameters
    ):

        kwargs[
            "max_completion_tokens"
        ] = 450

    elif (
        "max_tokens"
        in parameters
    ):

        kwargs[
            "max_tokens"
        ] = 450

    # ========================================================
    # CALL SERVICE
    # ========================================================

    result = (
        generate_json(
            **kwargs
        )
    )

    return await _maybe_await(
        result
    )


# ============================================================
# CLAIM EXTRACTOR AGENT
# ============================================================


async def claim_extractor_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    # ========================================================
    # GET GENERATED ANSWER
    # ========================================================

    answer = (
        state.get(
            "generated_answer"
        )
        or
        state.get(
            "final_answer"
        )
        or
        state.get(
            "answer"
        )
        or
        ""
    )

    answer = str(
        answer
    ).strip()

    # ========================================================
    # EMPTY ANSWER
    # ========================================================

    if not answer:

        print(
            "[CLAIMS] No generated answer."
        )

        return {

            "claims":
                [],

            "status":
                "claims_extracted",
        }

    # ========================================================
    # BUILD EXTRACTION PROMPT
    # ========================================================

    prompt = (
        _build_claim_prompt(
            answer
        )
    )

    # ========================================================
    # GENERATE CLAIM JSON
    # ========================================================

    result = (
        await _generate_claim_json(
            prompt
        )
    )

    print(
        "[CLAIMS] Raw response type:",
        type(
            result
        ).__name__,
    )

    # ========================================================
    # NORMALIZE
    # ========================================================

    claims = (
        normalize_claims(
            result
        )
    )

    # ========================================================
    # FALLBACK
    # ========================================================

    if not claims:

        print(
            "[CLAIMS] No usable LLM claims. "
            "Using sentence fallback."
        )

        claims = (
            _fallback_claims_from_answer(
                answer
            )
        )

    # ========================================================
    # LOG
    # ========================================================

    print(
        "[CLAIMS] Normalized claim count:",
        len(
            claims
        ),
    )

    if claims:

        print(
            "[CLAIMS] First claim:",
            claims[
                0
            ],
        )

    # ========================================================
    # RETURN STATE UPDATE
    # ========================================================

    return {

        "claims":
            claims,

        "status":
            "claims_extracted",
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================


async def claim_extraction_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await claim_extractor_agent(
        state
    )


async def extract_claims_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await claim_extractor_agent(
        state
    )