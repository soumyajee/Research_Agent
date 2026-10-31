# app/agents/grounding.py

from __future__ import annotations

from typing import Any

from app.evaluation.evidence import (
    evaluate_claims,
    get_claim_text,
    normalize_evidence,
)


# ============================================================
# SAFE SCORE
# ============================================================


def _safe_score(
    value: Any,
) -> float:

    try:

        score = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0.0

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


# ============================================================
# SELECT EVIDENCE FROM STATE
# ============================================================


def _get_state_evidence(
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Select the strongest available retrieval/evidence collection.

    Priority:

        evidence
        reranked_results
        retrieved_chunks
        retrieved_documents
        context
    """

    candidate_keys = (
        "evidence",
        "reranked_results",
        "retrieved_chunks",
        "retrieved_documents",
        "fused_results",
    )

    for key in candidate_keys:

        candidate = state.get(
            key
        )

        normalized = normalize_evidence(
            candidate
        )

        if normalized:

            return normalized

    # ========================================================
    # FALLBACK TO CONTEXT STRING
    # ========================================================

    context = (
        state.get(
            "context"
        )
        or
        ""
    )

    if isinstance(
        context,
        str,
    ):

        context = context.strip()

        if context:

            return [
                {
                    "id":
                        "context_1",

                    "text":
                        context,

                    "source":
                        "retrieved_context",

                    "raw":
                        context,
                }
            ]

    return []


# ============================================================
# CLAIM NORMALIZATION SAFETY
# ============================================================


def _clean_claims(
    claims: Any,
) -> list[Any]:
    """
    Ensure malformed claim entries do not crash grounding.
    """

    if claims is None:

        return []

    if isinstance(
        claims,
        (
            str,
            dict,
        ),
    ):

        claims = [
            claims
        ]

    if not isinstance(
        claims,
        list,
    ):

        return []

    cleaned: list[Any] = []

    for claim in claims:

        claim_text = get_claim_text(
            claim
        )

        if not claim_text:
            continue

        if isinstance(
            claim,
            dict,
        ):

            cleaned_claim = dict(
                claim
            )

            cleaned_claim[
                "claim"
            ] = claim_text

            cleaned_claim[
                "text"
            ] = claim_text

        else:

            cleaned_claim = {
                "claim":
                    claim_text,

                "text":
                    claim_text,
            }

        cleaned.append(
            cleaned_claim
        )

    return cleaned


# ============================================================
# AGGREGATE SCORES
# ============================================================


def _calculate_scores(
    evaluations: list[dict[str, Any]],
) -> dict[str, float]:

    if not evaluations:

        return {
            "evidence_score":
                0.0,

            "grounding_score":
                0.0,

            "faithfulness_score":
                0.0,

            "hallucination_score":
                1.0,

            "citation_score":
                0.0,
        }

    total = len(
        evaluations
    )

    # ========================================================
    # ENTAILMENT / GROUNDING
    # ========================================================

    entailment_values = [
        _safe_score(
            evaluation.get(
                "entailment_score"
            )
        )
        for evaluation
        in evaluations
    ]

    grounding_score = (
        sum(
            entailment_values
        )
        /
        total
    )

    # ========================================================
    # SUPPORTED CLAIM RATIO
    # ========================================================

    supported_count = sum(
        1
        for evaluation
        in evaluations
        if evaluation.get(
            "grounded"
        )
    )

    faithfulness_score = (
        supported_count
        /
        total
    )

    # ========================================================
    # CONTRADICTION / HALLUCINATION
    # ========================================================

    contradiction_values = [
        _safe_score(
            evaluation.get(
                "contradiction_score"
            )
        )
        for evaluation
        in evaluations
    ]

    average_contradiction = (
        sum(
            contradiction_values
        )
        /
        total
    )

    unsupported_ratio = (
        1.0
        -
        faithfulness_score
    )

    # Use both lack of support and direct contradiction.
    hallucination_score = max(
        unsupported_ratio,
        average_contradiction,
    )

    hallucination_score = min(
        1.0,
        hallucination_score,
    )

    # ========================================================
    # EVIDENCE SCORE
    # ========================================================

    evidence_scores = [
        _safe_score(
            evaluation.get(
                "nli_score"
            )
        )
        for evaluation
        in evaluations
    ]

    evidence_score = (
        sum(
            evidence_scores
        )
        /
        total
    )

    # ========================================================
    # CITATION / SOURCE COVERAGE
    # ========================================================

    source_count = sum(
        1
        for evaluation
        in evaluations
        if evaluation.get(
            "source"
        )
    )

    citation_score = (
        source_count
        /
        total
    )

    return {
        "evidence_score":
            round(
                evidence_score,
                4,
            ),

        "grounding_score":
            round(
                grounding_score,
                4,
            ),

        "faithfulness_score":
            round(
                faithfulness_score,
                4,
            ),

        "hallucination_score":
            round(
                hallucination_score,
                4,
            ),

        "citation_score":
            round(
                citation_score,
                4,
            ),
    }


# ============================================================
# GROUNDING AGENT
# ============================================================


async def grounding_agent(
    state: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare extracted claims against retrieved evidence through
    the local NLI evaluator.
    """

    # ========================================================
    # CLAIMS
    # ========================================================

    claims = _clean_claims(
        state.get(
            "claims",
            [],
        )
    )

    print()
    print(
        "========================================"
    )

    print(
        "[GROUNDING] Starting evaluation"
    )

    print(
        "[GROUNDING] Claims:",
        len(claims),
    )

    # ========================================================
    # NO CLAIMS
    # ========================================================

    if not claims:

        print(
            "[GROUNDING] No claims to evaluate."
        )

        print(
            "========================================"
        )

        return {
            "claims":
                [],

            "claim_evaluations":
                [],

            "evidence_score":
                0.0,

            "grounding_score":
                0.0,

            "faithfulness_score":
                0.0,

            "hallucination_score":
                0.0,

            "citation_score":
                0.0,

            "status":
                "grounding_complete",
        }

    # ========================================================
    # EVIDENCE
    # ========================================================

    evidence = _get_state_evidence(
        state
    )

    print(
        "[GROUNDING] Evidence items:",
        len(evidence),
    )

    # ========================================================
    # NLI
    # ========================================================

    evaluations = await evaluate_claims(
        claims=
            claims,

        evidence_items=
            evidence,

        entailment_threshold=
            0.60,
    )

    # ========================================================
    # SCORES
    # ========================================================

    scores = _calculate_scores(
        evaluations
    )

    print(
        "[GROUNDING] Evaluated claims:",
        len(evaluations),
    )

    print(
        "[GROUNDING] Grounding score:",
        scores[
            "grounding_score"
        ],
    )

    print(
        "[GROUNDING] Faithfulness:",
        scores[
            "faithfulness_score"
        ],
    )

    print(
        "[GROUNDING] Hallucination:",
        scores[
            "hallucination_score"
        ],
    )

    print(
        "========================================"
    )

    # ========================================================
    # STATE UPDATE
    # ========================================================

    return {
        "claims":
            claims,

        "claim_evaluations":
            evaluations,

        "evidence":
            evidence,

        "evidence_score":
            scores[
                "evidence_score"
            ],

        "grounding_score":
            scores[
                "grounding_score"
            ],

        "faithfulness_score":
            scores[
                "faithfulness_score"
            ],

        "hallucination_score":
            scores[
                "hallucination_score"
            ],

        "citation_score":
            scores[
                "citation_score"
            ],

        "status":
            "grounding_complete",
    }


# ============================================================
# COMPATIBILITY ALIASES
# ============================================================


async def ground_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await grounding_agent(
        state
    )


async def grounding_evaluation_agent(
    state: dict[str, Any],
) -> dict[str, Any]:

    return await grounding_agent(
        state
    )