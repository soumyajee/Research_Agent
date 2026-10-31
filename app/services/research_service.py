# app/services/research_service.py

from __future__ import annotations

import asyncio
import json
import math
import uuid

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from groq import RateLimitError


# ============================================================
# CONFIG
# ============================================================

from app.config import settings


# ============================================================
# LANGGRAPH
# ============================================================

from app.graph.workflow import (
    research_graph,
)


# ============================================================
# LLM SERVICE
# ============================================================

from app.services.llm_service import (
    LLMRateLimitError,
    convert_groq_rate_limit,
)


# ============================================================
# RETRIEVAL
#
# Used again when the LLM pipeline is rate-limited.
# ============================================================

from app.retrieval.hybrid import (
    hybrid_search_debug,
)

from app.retrieval.context_builder import (
    build_context,
)


# ============================================================
# STORAGE
# ============================================================

from app.storage.postgres import (
    create_review,
    create_session,
    save_session_state,
)

from app.storage.redis import (
    cache_session_state,
)


# ============================================================
# FALLBACK CONFIG
# ============================================================

FALLBACK_TOP_K = 5

FALLBACK_GROUNDING_SCORE = 0.95
FALLBACK_FAITHFULNESS_SCORE = 0.95
FALLBACK_HALLUCINATION_SCORE = 0.05

# A retrieval-only fallback should not appear more trustworthy
# than a successful full LLM + claim extraction + JEV/NLI run.
FALLBACK_CONFIDENCE_CAP = 0.7497


# ============================================================
# JSON SAFE
#
# Prevents errors such as:
#
# Object of type UUID is not JSON serializable
# ============================================================


def _json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):

        if math.isfinite(
            value
        ):
            return value

        return None

    if isinstance(
        value,
        uuid.UUID,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        Decimal,
    ):
        return float(
            value
        )

    if isinstance(
        value,
        Enum,
    ):
        return _json_safe(
            value.value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            _json_safe(
                item
            )
            for item in value
        ]

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):

        try:
            return _json_safe(
                model_dump()
            )
        except Exception:
            pass

    tolist = getattr(
        value,
        "tolist",
        None,
    )

    if callable(
        tolist
    ):

        try:
            return _json_safe(
                tolist()
            )
        except Exception:
            pass

    item_function = getattr(
        value,
        "item",
        None,
    )

    if callable(
        item_function
    ):

        try:
            return _json_safe(
                item_function()
            )
        except Exception:
            pass

    return str(
        value
    )


# ============================================================
# CLAMP
# ============================================================


def _clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


# ============================================================
# FLOAT CONVERSION
# ============================================================


def _safe_float(
    value: Any,
) -> float | None:

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        (
            int,
            float,
        ),
    ):

        value = float(
            value
        )

        if math.isfinite(
            value
        ):
            return value

    return None


# ============================================================
# INITIAL RESEARCH STATE
# ============================================================


def build_initial_state(
    session_id: str,
    tenant_id: str,
    query: str,
) -> dict[str, Any]:

    return {

        # ====================================================
        # SESSION
        # ====================================================

        "session_id":
            session_id,

        "tenant_id":
            tenant_id,

        "query":
            query,

        "active_query":
            query,

        # ====================================================
        # PLANNER
        # ====================================================

        "intent":
            None,

        "plan":
            None,

        "research_plan":
            None,

        "sub_queries":
            [],

        # ====================================================
        # DISCOVERY
        # ====================================================

        "discovered_sources":
            [],

        "web_results":
            [],

        # ====================================================
        # RETRIEVAL
        # ====================================================

        "retrieved_documents":
            [],

        "retrieved_chunks":
            [],

        "retrieved_context":
            [],

        "dense_results":
            [],

        "sparse_results":
            [],

        "fused_results":
            [],

        "reranked_results":
            [],

        "context":
            "",

        # ====================================================
        # ANALYSIS
        # ====================================================

        "analysis":
            None,

        "analysis_result":
            None,

        "evidence":
            [],

        # ====================================================
        # GENERATION
        # ====================================================

        "generated_answer":
            None,

        "final_answer":
            None,

        "answer":
            None,

        "llm_generated":
            False,

        # ====================================================
        # CLAIM EXTRACTION
        # ====================================================

        "claims":
            [],

        "claim_evaluations":
            [],

        "claim_extraction_completed":
            False,

        # ====================================================
        # JEV / NLI
        # ====================================================

        "nli_evaluation_completed":
            False,

        # ====================================================
        # EVALUATION
        # ====================================================

        "evidence_score":
            0.0,

        "retrieval_quality":
            0.0,

        "grounding_score":
            0.0,

        "faithfulness_score":
            0.0,

        "hallucination_score":
            0.0,

        "citation_score":
            0.0,

        "confidence_score":
            0.0,

        "evaluation_mode":
            None,

        # ====================================================
        # RESPONSE MODE
        # ====================================================

        "response_mode":
            "full_agentic",

        "degraded_mode":
            False,

        # ====================================================
        # RETRY
        # ====================================================

        "retry_count":
            0,

        "max_retries":
            getattr(
                settings,
                "max_retries",
                2,
            ),

        "next_action":
            None,

        # ====================================================
        # HUMAN REVIEW
        # ====================================================

        "requires_human_review":
            False,

        "human_feedback":
            None,

        "review_reason":
            None,

        "review_id":
            None,

        # ====================================================
        # RATE LIMIT
        # ====================================================

        "rate_limit":
            None,

        # ====================================================
        # STATUS
        # ====================================================

        "status":
            "created",

        "errors":
            [],
    }


# ============================================================
# NORMALIZE TENANT
# ============================================================


def _normalize_tenant_id(
    tenant_id: str | None,
) -> str:

    tenant_id = (
        tenant_id
        or
        ""
    ).strip()

    return (
        tenant_id
        or
        "default"
    )


# ============================================================
# NORMALIZE QUERY
# ============================================================


def _normalize_query(
    query: str | None,
) -> str:

    query = (
        query
        or
        ""
    ).strip()

    if not query:

        raise ValueError(
            "Research query cannot be empty."
        )

    return query


# ============================================================
# NORMALIZE GRAPH RESULT
# ============================================================


def _normalize_graph_result(
    result: Any,
    fallback_state: dict[str, Any],
) -> dict[str, Any]:

    if result is None:

        return dict(
            fallback_state
        )

    if isinstance(
        result,
        dict,
    ):

        return dict(
            result
        )

    model_dump = getattr(
        result,
        "model_dump",
        None,
    )

    if callable(
        model_dump
    ):

        dumped = model_dump()

        if isinstance(
            dumped,
            dict,
        ):
            return dumped

    try:

        return dict(
            result
        )

    except Exception as exc:

        raise RuntimeError(
            (
                "Research graph returned unsupported "
                f"state type: {type(result).__name__}"
            )
        ) from exc


# ============================================================
# PREPARE EXISTING STATE
# ============================================================


def _prepare_existing_state(
    existing_state: dict[str, Any],
    tenant_id: str,
    query: str,
) -> dict[str, Any]:

    state = deepcopy(
        existing_state
    )

    defaults = build_initial_state(
        session_id=str(
            state.get(
                "session_id"
            )
            or
            ""
        ),
        tenant_id=tenant_id,
        query=query,
    )

    for key, value in defaults.items():

        state.setdefault(
            key,
            deepcopy(
                value
            ),
        )

    # ========================================================
    # CURRENT REQUEST
    # ========================================================

    state[
        "tenant_id"
    ] = tenant_id

    state[
        "query"
    ] = query

    state[
        "active_query"
    ] = query

    # ========================================================
    # RESET TRANSIENT OUTPUT
    # ========================================================

    state[
        "analysis"
    ] = None

    state[
        "analysis_result"
    ] = None

    state[
        "generated_answer"
    ] = None

    state[
        "final_answer"
    ] = None

    state[
        "answer"
    ] = None

    state[
        "llm_generated"
    ] = False

    state[
        "claims"
    ] = []

    state[
        "claim_evaluations"
    ] = []

    state[
        "claim_extraction_completed"
    ] = False

    state[
        "nli_evaluation_completed"
    ] = False

    # ========================================================
    # RESET SCORES
    # ========================================================

    state[
        "evidence_score"
    ] = 0.0

    state[
        "retrieval_quality"
    ] = 0.0

    state[
        "grounding_score"
    ] = 0.0

    state[
        "faithfulness_score"
    ] = 0.0

    state[
        "hallucination_score"
    ] = 0.0

    state[
        "citation_score"
    ] = 0.0

    state[
        "confidence_score"
    ] = 0.0

    # ========================================================
    # RESET MODE
    # ========================================================

    state[
        "response_mode"
    ] = "full_agentic"

    state[
        "evaluation_mode"
    ] = None

    state[
        "degraded_mode"
    ] = False

    state[
        "rate_limit"
    ] = None

    state[
        "next_action"
    ] = None

    state[
        "requires_human_review"
    ] = False

    state[
        "review_reason"
    ] = None

    state[
        "review_id"
    ] = None

    state[
        "status"
    ] = "running"

    return state


# ============================================================
# APPEND ERROR
# ============================================================


def _append_error(
    state: dict[str, Any],
    error_type: str,
    message: str,
) -> None:

    errors = state.get(
        "errors"
    )

    if not isinstance(
        errors,
        list,
    ):
        errors = []

    errors.append(
        {
            "type":
                error_type,

            "message":
                str(
                    message
                ),
        }
    )

    state[
        "errors"
    ] = errors


# ============================================================
# PERSIST STATE
#
# PostgreSQL = source of truth
# Redis = cache
# ============================================================


async def _persist_state(
    session_id: str,
    state: dict[str, Any],
) -> None:

    safe_state = (
        _json_safe(
            state
        )
    )

    # ========================================================
    # POSTGRES
    # ========================================================

    await save_session_state(
        session_id=
            str(
                session_id
            ),

        state=
            safe_state,
    )

    # ========================================================
    # REDIS
    # ========================================================

    try:

        await cache_session_state(
            session_id=
                str(
                    session_id
                ),

            state=
                safe_state,

            ttl=
                3600,
        )

    except Exception as exc:

        print(
            "[RESEARCH] Redis state-cache warning:",
            repr(
                exc
            ),
        )


# ============================================================
# FINALIZE NORMAL STATE
# ============================================================


def _finalize_state(
    state: dict[str, Any],
) -> dict[str, Any]:

    state.setdefault(
        "errors",
        [],
    )

    state.setdefault(
        "claims",
        [],
    )

    state.setdefault(
        "claim_evaluations",
        [],
    )

    # ========================================================
    # RETRIEVAL FALLBACK IS ALREADY FINAL
    # ========================================================

    if state.get(
        "degraded_mode",
        False,
    ):

        return state

    # ========================================================
    # HUMAN REVIEW
    # ========================================================

    if state.get(
        "requires_human_review",
        False,
    ):

        state[
            "status"
        ] = "human_review"

        return state

    # ========================================================
    # NORMAL ANSWER
    # ========================================================

    answer = (
        state.get(
            "final_answer"
        )
        or
        state.get(
            "generated_answer"
        )
        or
        state.get(
            "answer"
        )
    )

    if answer:

        state[
            "final_answer"
        ] = answer

        state[
            "answer"
        ] = answer

        state[
            "llm_generated"
        ] = True

        state[
            "response_mode"
        ] = "full_agentic"

        state[
            "degraded_mode"
        ] = False

        state[
            "evaluation_mode"
        ] = (
            state.get(
                "evaluation_mode"
            )
            or
            "jev_nli"
        )

        state[
            "status"
        ] = "completed"

        return state

    # ========================================================
    # TERMINAL STATES
    # ========================================================

    if state.get(
        "status"
    ) in {
        "failed",
        "cancelled",
        "rejected",
    }:

        return state

    state[
        "status"
    ] = "incomplete"

    return state


# ============================================================
# CREATE NEW SESSION
# ============================================================


async def _create_new_research_session(
    tenant_id: str,
    query: str,
) -> tuple[
    str,
    dict[str, Any],
]:

    session_id = str(
        uuid.uuid4()
    )

    state = build_initial_state(
        session_id=
            session_id,

        tenant_id=
            tenant_id,

        query=
            query,
    )

    await create_session(
        session_id=
            session_id,

        tenant_id=
            tenant_id,

        query=
            query,
    )

    state[
        "status"
    ] = "running"

    await _persist_state(
        session_id=
            session_id,

        state=
            state,
    )

    return (
        session_id,
        state,
    )


# ============================================================
# PREPARE RESUMED SESSION
# ============================================================


async def _prepare_resumed_session(
    tenant_id: str,
    query: str,
    existing_state: dict[str, Any],
) -> tuple[
    str,
    dict[str, Any],
]:

    existing_session_id = (
        existing_state.get(
            "session_id"
        )
    )

    state = _prepare_existing_state(
        existing_state=
            existing_state,

        tenant_id=
            tenant_id,

        query=
            query,
    )

    if existing_session_id:

        session_id = str(
            existing_session_id
        )

    else:

        session_id = str(
            uuid.uuid4()
        )

        await create_session(
            session_id=
                session_id,

            tenant_id=
                tenant_id,

            query=
                query,
        )

    state[
        "session_id"
    ] = session_id

    await _persist_state(
        session_id=
            session_id,

        state=
            state,
    )

    return (
        session_id,
        state,
    )


# ============================================================
# RETRIEVAL FALLBACK HELPERS
# ============================================================


def _extract_text(
    item: dict[str, Any],
) -> str:

    value = (
        item.get("text")
        or
        item.get("content")
        or
        item.get("chunk_text")
        or
        item.get("page_content")
        or
        item.get("document")
        or
        ""
    )

    return str(
        value
    ).strip()


def _extract_source(
    item: dict[str, Any],
) -> str:

    source = (
        item.get("source")
        or
        item.get("filename")
        or
        item.get("file_name")
        or
        item.get("title")
        or
        item.get("document_name")
        or
        item.get("url")
        or
        ""
    )

    metadata = item.get(
        "metadata"
    )

    if (
        not source
        and
        isinstance(
            metadata,
            dict,
        )
    ):

        source = (
            metadata.get("source")
            or
            metadata.get("filename")
            or
            metadata.get("file_name")
            or
            metadata.get("title")
            or
            metadata.get("url")
            or
            ""
        )

    source = str(
        source
    ).strip()

    return (
        source
        or
        "Unknown source"
    )


def _extract_document_id(
    item: dict[str, Any],
) -> str | None:

    value = (
        item.get("document_id")
        or
        item.get("doc_id")
    )

    metadata = item.get(
        "metadata"
    )

    if (
        value is None
        and
        isinstance(
            metadata,
            dict,
        )
    ):

        value = (
            metadata.get("document_id")
            or
            metadata.get("doc_id")
        )

    if value is None:
        return None

    return str(
        value
    )


def _extract_chunk_id(
    item: dict[str, Any],
) -> str | None:

    value = (
        item.get("chunk_id")
        or
        item.get("point_id")
    )

    metadata = item.get(
        "metadata"
    )

    if (
        value is None
        and
        isinstance(
            metadata,
            dict,
        )
    ):

        value = (
            metadata.get("chunk_id")
            or
            metadata.get("point_id")
        )

    if value is None:
        return None

    return str(
        value
    )


def _extract_retrieval_score(
    item: dict[str, Any],
) -> float | None:

    candidate_keys = (
        "rerank_score",
        "retrieval_score",
        "similarity_score",
        "similarity",
        "score",
        "rrf_score",
    )

    for key in candidate_keys:

        score = _safe_float(
            item.get(
                key
            )
        )

        if score is None:
            continue

        if 0.0 <= score <= 1.0:
            return score

    return None


# ============================================================
# NORMALIZE RETRIEVED EVIDENCE
# ============================================================


def _normalize_retrieved_evidence(
    results: list[Any] | None,
    limit: int = FALLBACK_TOP_K,
) -> list[dict[str, Any]]:

    evidence: list[
        dict[str, Any]
    ] = []

    if not isinstance(
        results,
        list,
    ):
        return evidence

    for item in results:

        if len(
            evidence
        ) >= limit:
            break

        if isinstance(
            item,
            dict,
        ):

            raw = dict(
                item
            )

        else:

            model_dump = getattr(
                item,
                "model_dump",
                None,
            )

            if callable(
                model_dump
            ):

                try:
                    raw = model_dump()
                except Exception:
                    raw = {}

            else:

                try:
                    raw = dict(
                        item
                    )
                except Exception:
                    continue

        if not isinstance(
            raw,
            dict,
        ):
            continue

        text = _extract_text(
            raw
        )

        if not text:
            continue

        evidence_id = (
            f"E{len(evidence) + 1}"
        )

        evidence.append(
            {
                "evidence_id":
                    evidence_id,

                "document_id":
                    _extract_document_id(
                        raw
                    ),

                "chunk_id":
                    _extract_chunk_id(
                        raw
                    ),

                "source":
                    _extract_source(
                        raw
                    ),

                "text":
                    text,

                "retrieval_score":
                    _extract_retrieval_score(
                        raw
                    ),
            }
        )

    return evidence


# ============================================================
# RETRIEVAL QUALITY
# ============================================================


def _calculate_retrieval_quality(
    evidence: list[dict[str, Any]],
) -> float:

    if not evidence:
        return 0.0

    coverage = _clamp(
        len(
            evidence
        )
        /
        float(
            FALLBACK_TOP_K
        )
    )

    substantial_count = 0

    for item in evidence:

        if len(
            str(
                item.get(
                    "text",
                    "",
                )
            ).strip()
        ) >= 80:

            substantial_count += 1

    text_quality = (
        substantial_count
        /
        len(
            evidence
        )
    )

    retrieval_scores: list[
        float
    ] = []

    for item in evidence:

        value = _safe_float(
            item.get(
                "retrieval_score"
            )
        )

        if (
            value is not None
            and
            0.0 <= value <= 1.0
        ):

            retrieval_scores.append(
                value
            )

    if retrieval_scores:

        mean_retrieval_score = (
            sum(
                retrieval_scores
            )
            /
            len(
                retrieval_scores
            )
        )

        score = (
            coverage * 0.30
            +
            text_quality * 0.20
            +
            mean_retrieval_score * 0.50
        )

    else:

        score = (
            coverage * 0.60
            +
            text_quality * 0.40
        )

    return round(
        _clamp(
            score
        ),
        4,
    )


# ============================================================
# EVIDENCE SCORE
# ============================================================


def _calculate_evidence_score(
    evidence: list[dict[str, Any]],
) -> float:

    if not evidence:
        return 0.0

    total = len(
        evidence
    )

    nonempty = 0
    sourced = 0
    substantial = 0

    for item in evidence:

        text = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        source = str(
            item.get(
                "source",
                "",
            )
        ).strip()

        if text:
            nonempty += 1

        if (
            source
            and
            source != "Unknown source"
        ):
            sourced += 1

        if len(
            text
        ) >= 80:
            substantial += 1

    completeness = (
        nonempty
        /
        total
    )

    source_coverage = (
        sourced
        /
        total
    )

    content_quality = (
        substantial
        /
        total
    )

    score = (
        completeness * 0.40
        +
        source_coverage * 0.30
        +
        content_quality * 0.30
    )

    return round(
        _clamp(
            score
        ),
        4,
    )


# ============================================================
# CITATION SCORE
# ============================================================


def _calculate_citation_score(
    evidence: list[dict[str, Any]],
) -> float:

    if not evidence:
        return 0.0

    valid_sources = 0

    for item in evidence:

        source = str(
            item.get(
                "source",
                "",
            )
        ).strip()

        if (
            source
            and
            source != "Unknown source"
        ):
            valid_sources += 1

    return round(
        _clamp(
            valid_sources
            /
            len(
                evidence
            )
        ),
        4,
    )


# ============================================================
# RETRIEVAL FALLBACK ANSWER
# ============================================================


def _format_fallback_answer(
    evidence: list[dict[str, Any]],
) -> str:

    if not evidence:

        return (
            "The full agentic LLM pipeline is temporarily "
            "rate-limited. The system switched to "
            "**retrieval-only fallback mode**, but no relevant "
            "evidence was found in the indexed documents."
        )

    blocks = [
        (
            "The full agentic LLM pipeline is temporarily "
            "rate-limited. The system has therefore switched "
            "to **retrieval-only fallback mode** so that a "
            "grounded response can still be returned."
        )
    ]

    for item in evidence:

        evidence_id = (
            item.get(
                "evidence_id"
            )
            or
            "E?"
        )

        text = str(
            item.get(
                "text",
                "",
            )
        ).strip()

        source = str(
            item.get(
                "source",
                "Unknown source",
            )
        ).strip()

        blocks.append(
            (
                f"**[{evidence_id}]** "
                f"{text}\n"
                f"*Source: {source}*"
            )
        )

    return "\n\n".join(
        blocks
    )


# ============================================================
# FALLBACK CONFIDENCE
# ============================================================


def _calculate_fallback_confidence(
    grounding_score: float,
    faithfulness_score: float,
    citation_score: float,
    evidence_score: float,
    retrieval_quality: float,
) -> float:

    confidence = (
        grounding_score * 0.30
        +
        faithfulness_score * 0.25
        +
        citation_score * 0.20
        +
        evidence_score * 0.15
        +
        retrieval_quality * 0.10
    )

    confidence = min(
        confidence,
        FALLBACK_CONFIDENCE_CAP,
    )

    return round(
        _clamp(
            confidence
        ),
        4,
    )


# ============================================================
# BUILD FALLBACK STATE
# ============================================================


def _build_retrieval_fallback_state(
    state: dict[str, Any],
    retrieved_results: list[Any],
    rate_limit: dict[str, Any],
) -> dict[str, Any]:

    evidence = (
        _normalize_retrieved_evidence(
            results=
                retrieved_results,

            limit=
                FALLBACK_TOP_K,
        )
    )

    retrieval_quality = (
        _calculate_retrieval_quality(
            evidence
        )
    )

    evidence_score = (
        _calculate_evidence_score(
            evidence
        )
    )

    citation_score = (
        _calculate_citation_score(
            evidence
        )
    )

    # ========================================================
    # EXTRACTIVE FALLBACK SCORES
    #
    # IMPORTANT:
    # These are NOT JEV/NLI scores.
    #
    # They indicate the deterministic retrieval-only response
    # contains direct source chunks rather than newly generated
    # claims.
    # ========================================================

    if evidence:

        grounding_score = (
            FALLBACK_GROUNDING_SCORE
        )

        faithfulness_score = (
            FALLBACK_FAITHFULNESS_SCORE
        )

        hallucination_score = (
            FALLBACK_HALLUCINATION_SCORE
        )

    else:

        grounding_score = 0.0
        faithfulness_score = 0.0
        hallucination_score = 1.0
        citation_score = 0.0

    confidence_score = (
        _calculate_fallback_confidence(
            grounding_score=
                grounding_score,

            faithfulness_score=
                faithfulness_score,

            citation_score=
                citation_score,

            evidence_score=
                evidence_score,

            retrieval_quality=
                retrieval_quality,
        )
    )

    answer = (
        _format_fallback_answer(
            evidence
        )
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    state[
        "retrieved_context"
    ] = evidence

    state[
        "retrieved_documents"
    ] = evidence

    state[
        "retrieved_chunks"
    ] = evidence

    state[
        "evidence"
    ] = evidence

    # ========================================================
    # ANSWER
    # ========================================================

    state[
        "generated_answer"
    ] = None

    state[
        "final_answer"
    ] = answer

    state[
        "answer"
    ] = answer

    state[
        "llm_generated"
    ] = False

    # ========================================================
    # CLAIM / JEV
    # ========================================================

    state[
        "claims"
    ] = []

    state[
        "claim_evaluations"
    ] = []

    state[
        "claim_extraction_completed"
    ] = False

    state[
        "nli_evaluation_completed"
    ] = False

    # ========================================================
    # SCORES
    # ========================================================

    state[
        "retrieval_quality"
    ] = retrieval_quality

    state[
        "evidence_score"
    ] = evidence_score

    state[
        "grounding_score"
    ] = grounding_score

    state[
        "faithfulness_score"
    ] = faithfulness_score

    state[
        "hallucination_score"
    ] = hallucination_score

    state[
        "citation_score"
    ] = citation_score

    state[
        "confidence_score"
    ] = confidence_score

    # ========================================================
    # MODE
    # ========================================================

    state[
        "response_mode"
    ] = "retrieval_only_fallback"

    state[
        "evaluation_mode"
    ] = (
        "deterministic_extractive_fallback"
    )

    state[
        "degraded_mode"
    ] = True

    state[
        "requires_human_review"
    ] = False

    state[
        "review_reason"
    ] = None

    state[
        "review_id"
    ] = None

    state[
        "next_action"
    ] = "retry_full_llm_later"

    state[
        "rate_limit"
    ] = rate_limit

    state[
        "status"
    ] = (
        "degraded_completed"
        if evidence
        else
        "degraded_incomplete"
    )

    return state


# ============================================================
# RUN RETRIEVAL FALLBACK
# ============================================================


async def _run_retrieval_fallback(
    session_id: str,
    tenant_id: str,
    query: str,
    state: dict[str, Any],
    rate_limit: dict[str, Any],
) -> dict[str, Any]:

    print()
    print(
        "========================================"
    )
    print(
        "[RESEARCH] RETRIEVAL-ONLY FALLBACK"
    )
    print(
        "========================================"
    )

    print(
        "[RESEARCH] Query:",
        query,
    )

    retrieval_result: dict[
        str,
        Any,
    ] = {}

    retrieval_error: Exception | None = None

    # ========================================================
    # RETRIEVE DIRECTLY
    # ========================================================

    try:

        retrieval_result = (
            await hybrid_search_debug(
                tenant_id=
                    tenant_id,

                query=
                    query,

                limit=
                    FALLBACK_TOP_K,

                use_cache=
                    False,
            )
        )

    except Exception as exc:

        retrieval_error = exc

        print(
            "[RESEARCH] Fallback retrieval failed:",
            repr(
                exc
            ),
        )

    # ========================================================
    # GET FINAL RESULTS
    # ========================================================

    final_results = (
        retrieval_result.get(
            "final_results"
        )
        or
        retrieval_result.get(
            "reranked_results"
        )
        or
        retrieval_result.get(
            "results"
        )
        or
        state.get(
            "reranked_results"
        )
        or
        state.get(
            "retrieved_chunks"
        )
        or
        []
    )

    # ========================================================
    # SAVE RETRIEVAL PIPELINE DETAILS
    # ========================================================

    if retrieval_result:

        state[
            "dense_results"
        ] = (
            retrieval_result.get(
                "dense_results"
            )
            or
            state.get(
                "dense_results"
            )
            or
            []
        )

        state[
            "sparse_results"
        ] = (
            retrieval_result.get(
                "sparse_results"
            )
            or
            state.get(
                "sparse_results"
            )
            or
            []
        )

        state[
            "fused_results"
        ] = (
            retrieval_result.get(
                "fused_results"
            )
            or
            state.get(
                "fused_results"
            )
            or
            []
        )

        state[
            "reranked_results"
        ] = (
            retrieval_result.get(
                "reranked_results"
            )
            or
            final_results
        )

    # ========================================================
    # BUILD CONTEXT STRING
    # ========================================================

    try:

        state[
            "context"
        ] = (
            build_context(
                final_results
            )
            if final_results
            else
            ""
        )

    except Exception as exc:

        print(
            "[RESEARCH] Context build warning:",
            repr(
                exc
            ),
        )

        state[
            "context"
        ] = ""

    # ========================================================
    # RETRIEVAL ERROR
    # ========================================================

    if retrieval_error is not None:

        _append_error(
            state=
                state,

            error_type=
                "fallback_retrieval_error",

            message=
                str(
                    retrieval_error
                ),
        )

    # ========================================================
    # BUILD FALLBACK OUTPUT
    # ========================================================

    state = (
        _build_retrieval_fallback_state(
            state=
                state,

            retrieved_results=
                final_results,

            rate_limit=
                rate_limit,
        )
    )

    # ========================================================
    # SAVE
    # ========================================================

    await _persist_state(
        session_id=
            session_id,

        state=
            state,
    )

    print(
        "[RESEARCH] Fallback evidence:",
        len(
            state.get(
                "retrieved_context",
                [],
            )
        ),
    )

    print(
        "[RESEARCH] Grounding:",
        state.get(
            "grounding_score"
        ),
    )

    print(
        "[RESEARCH] Faithfulness:",
        state.get(
            "faithfulness_score"
        ),
    )

    print(
        "[RESEARCH] Hallucination:",
        state.get(
            "hallucination_score"
        ),
    )

    print(
        "[RESEARCH] Citation:",
        state.get(
            "citation_score"
        ),
    )

    print(
        "[RESEARCH] Confidence:",
        state.get(
            "confidence_score"
        ),
    )

    return state


# ============================================================
# CUSTOM LLM RATE LIMIT PAYLOAD
# ============================================================


def _build_llm_rate_limit_payload(
    exc: LLMRateLimitError,
) -> dict[str, Any]:

    return {

        "provider":
            "groq",

        "quota_type":
            getattr(
                exc,
                "quota_type",
                None,
            ),

        "retry_after_seconds":
            getattr(
                exc,
                "retry_after_seconds",
                None,
            ),

        "message":
            getattr(
                exc,
                "message",
                str(
                    exc
                ),
            ),
    }


# ============================================================
# RAW GROQ RATE LIMIT PAYLOAD
# ============================================================


def _build_raw_groq_rate_limit_payload(
    exc: RateLimitError,
) -> dict[str, Any]:

    try:

        converted = (
            convert_groq_rate_limit(
                exc
            )
        )

        return {

            "provider":
                "groq",

            "quota_type":
                getattr(
                    converted,
                    "quota_type",
                    None,
                ),

            "retry_after_seconds":
                getattr(
                    converted,
                    "retry_after_seconds",
                    None,
                ),

            "message":
                getattr(
                    converted,
                    "message",
                    str(
                        exc
                    ),
                ),
        }

    except Exception:

        return {

            "provider":
                "groq",

            "quota_type":
                None,

            "retry_after_seconds":
                None,

            "message":
                str(
                    exc
                ),
        }


# ============================================================
# RUN RESEARCH
# ============================================================


async def run_research(
    tenant_id: str,
    query: str,
    existing_state: dict | None = None,
) -> tuple[
    str,
    dict[str, Any],
    str | None,
]:

    tenant_id = (
        _normalize_tenant_id(
            tenant_id
        )
    )

    query = (
        _normalize_query(
            query
        )
    )

    review_id: str | None = None

    # ========================================================
    # CREATE / RESUME SESSION
    # ========================================================

    if existing_state is None:

        (
            session_id,
            state,

        ) = await _create_new_research_session(
            tenant_id=
                tenant_id,

            query=
                query,
        )

        print()
        print(
            "========================================"
        )
        print(
            "[RESEARCH] New research session"
        )
        print(
            "========================================"
        )

    else:

        (
            session_id,
            state,

        ) = await _prepare_resumed_session(
            tenant_id=
                tenant_id,

            query=
                query,

            existing_state=
                existing_state,
        )

        print()
        print(
            "========================================"
        )
        print(
            "[RESEARCH] Resuming research session"
        )
        print(
            "========================================"
        )

    print(
        "[RESEARCH] Session:",
        session_id,
    )

    print(
        "[RESEARCH] Tenant:",
        tenant_id,
    )

    print(
        "[RESEARCH] Query:",
        query,
    )

    # ========================================================
    # NORMAL FULL LANGGRAPH
    # ========================================================

    try:

        print(
            "[RESEARCH] Starting LangGraph..."
        )

        graph_result = (
            await research_graph.ainvoke(
                state
            )
        )

        print(
            "[RESEARCH] LangGraph finished."
        )

        # ====================================================
        # NORMALIZE RESULT
        # ====================================================

        result_state = (
            _normalize_graph_result(
                result=
                    graph_result,

                fallback_state=
                    state,
            )
        )

        # ====================================================
        # GUARANTEE CORE FIELDS
        # ====================================================

        result_state[
            "session_id"
        ] = session_id

        result_state[
            "tenant_id"
        ] = (
            result_state.get(
                "tenant_id"
            )
            or
            tenant_id
        )

        result_state[
            "query"
        ] = (
            result_state.get(
                "query"
            )
            or
            query
        )

        result_state[
            "active_query"
        ] = (
            result_state.get(
                "active_query"
            )
            or
            result_state[
                "query"
            ]
        )

        # ====================================================
        # FINALIZE
        # ====================================================

        result_state = (
            _finalize_state(
                result_state
            )
        )

        # ====================================================
        # PERSIST
        # ====================================================

        await _persist_state(
            session_id=
                session_id,

            state=
                result_state,
        )

        # ====================================================
        # HUMAN REVIEW
        # ====================================================

        if result_state.get(
            "requires_human_review",
            False,
        ):

            print(
                "[RESEARCH] Human review required."
            )

            review_id = (
                await create_review(
                    session_id=
                        session_id
                )
            )

            result_state[
                "review_id"
            ] = str(
                review_id
            )

            await _persist_state(
                session_id=
                    session_id,

                state=
                    result_state,
            )

        print(
            "[RESEARCH] Status:",
            result_state.get(
                "status"
            ),
        )

        print(
            "[RESEARCH] Response mode:",
            result_state.get(
                "response_mode"
            ),
        )

        print(
            "[RESEARCH] Confidence:",
            result_state.get(
                "confidence_score"
            ),
        )

        return (
            session_id,
            _json_safe(
                result_state
            ),
            (
                str(
                    review_id
                )
                if review_id is not None
                else
                None
            ),
        )

    # ========================================================
    # CUSTOM LLM RATE LIMIT
    #
    # DO NOT RAISE.
    #
    # Switch to retrieval-only fallback.
    # ========================================================

    except LLMRateLimitError as exc:

        print()
        print(
            "[RESEARCH] LLM rate limit detected."
        )

        print(
            "[RESEARCH] Switching to retrieval-only fallback."
        )

        rate_limit = (
            _build_llm_rate_limit_payload(
                exc
            )
        )

        _append_error(
            state=
                state,

            error_type=
                "llm_rate_limit",

            message=
                str(
                    rate_limit.get(
                        "message"
                    )
                    or
                    exc
                ),
        )

        fallback_state = (
            await _run_retrieval_fallback(
                session_id=
                    session_id,

                tenant_id=
                    tenant_id,

                query=
                    query,

                state=
                    state,

                rate_limit=
                    rate_limit,
            )
        )

        return (
            session_id,
            _json_safe(
                fallback_state
            ),
            None,
        )

    # ========================================================
    # RAW GROQ RATE LIMIT
    # ========================================================

    except RateLimitError as exc:

        print()
        print(
            "[RESEARCH] Raw Groq 429 detected."
        )

        print(
            "[RESEARCH] Switching to retrieval-only fallback."
        )

        rate_limit = (
            _build_raw_groq_rate_limit_payload(
                exc
            )
        )

        _append_error(
            state=
                state,

            error_type=
                "groq_rate_limit",

            message=
                str(
                    rate_limit.get(
                        "message"
                    )
                    or
                    exc
                ),
        )

        fallback_state = (
            await _run_retrieval_fallback(
                session_id=
                    session_id,

                tenant_id=
                    tenant_id,

                query=
                    query,

                state=
                    state,

                rate_limit=
                    rate_limit,
            )
        )

        return (
            session_id,
            _json_safe(
                fallback_state
            ),
            None,
        )

    # ========================================================
    # REQUEST CANCELLED
    # ========================================================

    except asyncio.CancelledError:

        print(
            "[RESEARCH] Request cancelled."
        )

        state[
            "status"
        ] = "cancelled"

        state[
            "requires_human_review"
        ] = False

        _append_error(
            state=
                state,

            error_type=
                "cancelled",

            message=
                "Research request was cancelled.",
        )

        try:

            await _persist_state(
                session_id=
                    session_id,

                state=
                    state,
            )

        except Exception as persist_exc:

            print(
                "[RESEARCH] Failed to persist cancelled state:",
                repr(
                    persist_exc
                ),
            )

        raise

    # ========================================================
    # GENERAL PIPELINE FAILURE
    # ========================================================

    except Exception as exc:

        print()
        print(
            "[RESEARCH] Pipeline failed:"
        )

        print(
            repr(
                exc
            )
        )

        state[
            "status"
        ] = "failed"

        state[
            "requires_human_review"
        ] = False

        _append_error(
            state=
                state,

            error_type=
                type(
                    exc
                ).__name__,

            message=
                str(
                    exc
                ),
        )

        try:

            await _persist_state(
                session_id=
                    session_id,

                state=
                    state,
            )

        except Exception as persist_exc:

            print(
                "[RESEARCH] Failed to persist error state:",
                repr(
                    persist_exc
                ),
            )

        raise RuntimeError(
            (
                "Research workflow failed: "
                f"{exc}"
            )
        ) from exc