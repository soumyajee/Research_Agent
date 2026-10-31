from app.config import settings


async def confidence_agent(state):
    grounding = state.get("grounding_score", 0.0)
    faithfulness = state.get("faithfulness_score", 0.0)
    hallucination = state.get("hallucination_score", 1.0)
    citation = state.get("citation_score", 0.0)
    retrieval = state.get("retrieval_quality", 0.0)

    confidence = (
        grounding * 0.30
        + faithfulness * 0.30
        + (1.0 - hallucination) * 0.20
        + citation * 0.10
        + retrieval * 0.10
    )

    confidence = max(0.0, min(confidence, 1.0))
    retry_count = state.get("retry_count", 0)

    if confidence >= settings.confidence_high:
        action = "finalize"
    elif (
        confidence >= settings.confidence_retry
        and retry_count < settings.max_retries
    ):
        action = "retry"
        retry_count += 1
    else:
        action = "human_review"

    return {
        "confidence_score": confidence,
        "next_action": action,
        "retry_count": retry_count,
        "status": "confidence_evaluated",
    }
