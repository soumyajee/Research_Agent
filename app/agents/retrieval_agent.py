from app.retrieval.hybrid import hybrid_search
from app.retrieval.query_rewriter import rewrite_query


async def retrieval_agent(state):
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    if retry_count > 0:
        query = await rewrite_query(
            original_query=state["query"],
            claim_evaluations=state.get("claim_evaluations", []),
        )

    results = await hybrid_search(
        tenant_id=state["tenant_id"],
        query=query,
        use_cache=retry_count == 0,
    )

    if not results:
        retrieval_quality = 0.0
    else:
        vector_scores = [
            max(0.0, min(1.0, float(result.get("vector_score", 0.0))))
            for result in results[:5]
            if result.get("vector_score") is not None
        ]
        retrieval_quality = (
            sum(vector_scores) / len(vector_scores)
            if vector_scores
            else min(len(results) / 5.0, 1.0)
        )

    return {
        "active_query": query,
        "retrieved_chunks": results,
        "retrieval_quality": retrieval_quality,
        "status": "retrieved",
    }
