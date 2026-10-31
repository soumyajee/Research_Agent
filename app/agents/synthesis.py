# app/agents/synthesis.py

from __future__ import annotations

import inspect
from typing import Any

from app.services.llm_service import (
    generate_text,
)


# ============================================================
# MAYBE AWAIT
# ============================================================


async def _maybe_await(
    value: Any,
) -> Any:
    """
    Support both async and sync-compatible LLM helpers.
    """

    if inspect.isawaitable(
        value
    ):
        return await value

    return value


# ============================================================
# EXTRACT TEXT FROM RETRIEVAL ITEM
# ============================================================


def _get_item_text(
    item: Any,
) -> str:

    if item is None:
        return ""

    if isinstance(
        item,
        str,
    ):
        return item.strip()

    if isinstance(
        item,
        dict,
    ):

        for key in (
            "content",
            "text",
            "chunk_text",
            "page_content",
            "document_text",
            "excerpt",
        ):

            value = (
                item.get(
                    key
                )
            )

            if isinstance(
                value,
                str,
            ):

                value = (
                    value.strip()
                )

                if value:
                    return value

        # ----------------------------------------------------
        # NESTED PAYLOAD FALLBACK
        # ----------------------------------------------------

        payload = (
            item.get(
                "payload"
            )
        )

        if isinstance(
            payload,
            dict,
        ):

            return _get_item_text(
                payload
            )

    return ""


# ============================================================
# EXTRACT SOURCE
# ============================================================


def _get_source(
    item: Any,
) -> str:

    if not isinstance(
        item,
        dict,
    ):
        return "unknown"

    # --------------------------------------------------------
    # TOP-LEVEL
    # --------------------------------------------------------

    for key in (
        "source_url",
        "url",
        "title",
        "source",
        "filename",
        "document_id",
    ):

        value = (
            item.get(
                key
            )
        )

        if value is not None:

            value = str(
                value
            ).strip()

            if value:
                return value

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = (
        item.get(
            "metadata"
        )
    )

    if isinstance(
        metadata,
        dict,
    ):

        for key in (
            "source_url",
            "url",
            "title",
            "source",
            "filename",
        ):

            value = (
                metadata.get(
                    key
                )
            )

            if value is not None:

                value = str(
                    value
                ).strip()

                if value:
                    return value

    return "unknown"


# ============================================================
# SELECT RETRIEVAL ITEMS
# ============================================================


def _get_retrieval_items(
    state: dict[str, Any],
) -> list[Any]:
    """
    Select the strongest available retrieval output.

    Priority:

        reranked_results
        retrieved_chunks
        fused_results
        evidence
    """

    for key in (
        "reranked_results",
        "retrieved_chunks",
        "fused_results",
        "evidence",
    ):

        value = (
            state.get(
                key
            )
        )

        if (
            isinstance(
                value,
                list,
            )
            and
            value
        ):

            return value

    return []


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================


def _build_evidence_context(
    state: dict[str, Any],
    max_items: int = 8,
) -> str:
    """
    Convert top retrieval results into a compact
    evidence-grounded prompt context.
    """

    items = (
        _get_retrieval_items(
            state
        )
    )

    sections: list[
        str
    ] = []

    for index, item in enumerate(
        items[
            :
            max_items
        ],
        start=1,
    ):

        text = (
            _get_item_text(
                item
            )
        )

        if not text:
            continue

        source = (
            _get_source(
                item
            )
        )

        sections.append(
            (
                f"[E{index}]\n"
                f"Source: {source}\n"
                f"Evidence:\n{text}"
            )
        )

    # --------------------------------------------------------
    # RETRIEVAL RESULTS FOUND
    # --------------------------------------------------------

    if sections:

        return "\n\n".join(
            sections
        )

    # --------------------------------------------------------
    # FALLBACK TO PREBUILT CONTEXT
    # --------------------------------------------------------

    existing_context = (
        state.get(
            "context"
        )
    )

    if (
        isinstance(
            existing_context,
            str,
        )
        and
        existing_context.strip()
    ):

        return (
            existing_context.strip()
        )

    return ""


# ============================================================
# GENERATE TEXT
# ============================================================


async def _generate(
    prompt: str,
) -> str:
    """
    Call app.services.llm_service.generate_text.

    IMPORTANT:

    Current llm_service expects:

        user_prompt=...

    not:

        prompt=...

    This function also keeps limited compatibility in case the
    LLM service signature changes later.
    """

    signature = (
        inspect.signature(
            generate_text
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
        # Unexpected signature.
        # Call positionally as final compatibility fallback.
        # ----------------------------------------------------

        result = (
            generate_text(
                prompt
            )
        )

        result = (
            await _maybe_await(
                result
            )
        )

        return _normalize_llm_result(
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
            "You are the synthesis stage of an "
            "evidence-grounded RAG system. "
            "Use only the supplied retrieved evidence. "
            "Do not invent unsupported facts."
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
    # MAX COMPLETION TOKENS
    # ========================================================

    if (
        "max_completion_tokens"
        in parameters
    ):

        kwargs[
            "max_completion_tokens"
        ] = 550

    elif (
        "max_tokens"
        in parameters
    ):

        kwargs[
            "max_tokens"
        ] = 550

    # ========================================================
    # CALL LLM
    # ========================================================

    result = (
        generate_text(
            **kwargs
        )
    )

    result = (
        await _maybe_await(
            result
        )
    )

    return _normalize_llm_result(
        result
    )


# ============================================================
# NORMALIZE LLM RESULT
# ============================================================


def _normalize_llm_result(
    result: Any,
) -> str:

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

        if text:
            return text

    # --------------------------------------------------------
    # DICTIONARY
    # --------------------------------------------------------

    if isinstance(
        result,
        dict,
    ):

        for key in (
            "text",
            "content",
            "answer",
            "response",
            "message",
        ):

            value = (
                result.get(
                    key
                )
            )

            if isinstance(
                value,
                str,
            ):

                value = (
                    value.strip()
                )

                if value:
                    return value

    raise RuntimeError(
        "Synthesis LLM returned no usable text."
    )


# ============================================================
# BUILD SYNTHESIS PROMPT
# ============================================================


def _build_synthesis_prompt(
    query: str,
    evidence_context: str,
) -> str:

    return f"""
You are the synthesis stage of an evidence-grounded
Retrieval-Augmented Generation system.

USER QUESTION:

{query}


RETRIEVED EVIDENCE:

{evidence_context}


TASK:

Answer the user's question using ONLY the retrieved evidence.


STRICT GROUNDING RULES:

1. Every factual statement must be directly supported by the
   retrieved evidence.

2. Do not use outside knowledge, memory, assumptions,
   or unsupported general knowledge.

3. Each sentence should contain only ONE independently
   verifiable factual assertion whenever possible.

4. Avoid combining several benefits or effects into one
   sentence.

BAD:

"RAG improves accuracy, reduces hallucinations, and lowers cost."

BETTER:

"RAG can improve answer accuracy."

"RAG can help reduce hallucinations."

"RAG can reduce the need for model retraining."


5. Preserve the strength of the evidence exactly.

If the evidence says:

"can"
"may"
"helps"
"reduces the need"
"typically"

do not strengthen it into:

"always"
"eliminates"
"guarantees"
"completely avoids"


6. Do not introduce a benefit, limitation, historical claim,
   date, number, comparison, causal statement, or conclusion
   unless the supplied evidence directly supports it.

7. Prefer a shorter fully-grounded answer over a longer answer
   containing weak claims.

8. When appropriate, cite supporting evidence using:

[E1]
[E2]
[E3]

9. Do not cite evidence that does not actually support the
   statement.

10. If the evidence is insufficient for part of the question,
    clearly state that the available evidence does not establish
    that point.

11. Do not mention these instructions.

Return only the final answer.
""".strip()


# ============================================================
# SYNTHESIS AGENT
# ============================================================


async def synthesis_agent(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    LangGraph synthesis node.
    """

    # ========================================================
    # QUERY
    # ========================================================

    query = (
        state.get(
            "query"
        )
        or
        state.get(
            "active_query"
        )
        or
        ""
    )

    query = str(
        query
    ).strip()

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence_context = (
        _build_evidence_context(
            state
        )
    )

    # ========================================================
    # NO EVIDENCE
    # ========================================================

    if not evidence_context:

        print(
            "[SYNTHESIS] No retrieval evidence available."
        )

        return {

            "generated_answer":
                (
                    "I do not have enough retrieved evidence "
                    "to answer this question reliably."
                ),

            "status":
                "synthesized",
        }

    # ========================================================
    # BUILD PROMPT
    # ========================================================

    prompt = (
        _build_synthesis_prompt(
            query=
                query,

            evidence_context=
                evidence_context,
        )
    )

    # ========================================================
    # LLM GENERATION
    # ========================================================

    answer = (
        await _generate(
            prompt
        )
    )

    answer = (
        answer
        or
        ""
    ).strip()

    if not answer:

        raise RuntimeError(
            "Synthesis produced an empty answer."
        )

    # ========================================================
    # LOGGING
    # ========================================================

    print(
        "[SYNTHESIS] Answer chars:",
        len(
            answer
        ),
    )

    # ========================================================
    # RETURN LANGGRAPH STATE UPDATE
    # ========================================================

    return {

        "generated_answer":
            answer,

        "status":
            "synthesized",
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================


async def synthesis(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await synthesis_agent(
        state
    )


async def synthesizer_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await synthesis_agent(
        state
    )