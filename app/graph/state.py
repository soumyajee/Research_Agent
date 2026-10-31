from typing import Any, TypedDict


class ResearchState(TypedDict, total=False):
    session_id: str
    tenant_id: str
    query: str

    intent: str
    plan: list[dict[str, Any]]
    search_queries: list[str]
    active_query: str

    needs_web: bool
    needs_internal: bool
    needs_analysis: bool

    discovered_sources: list[dict]
    retrieved_chunks: list[dict]
    analysis_results: list[dict]

    generated_answer: str
    claims: list[dict]
    claim_evaluations: list[dict]

    evidence_score: float
    grounding_score: float
    faithfulness_score: float
    hallucination_score: float
    citation_score: float
    retrieval_quality: float
    confidence_score: float

    retry_count: int
    next_action: str
    human_feedback: str | None
    requires_human_review: bool

    final_answer: str
    status: str
    errors: list[str]
