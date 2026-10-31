from typing import Any

from app.retrieval.context_builder import (
    build_context,
)

from app.services.llm_service import (
    generate_json,
)


# ============================================================
# GET RETRIEVED DOCUMENTS
# ============================================================

def _get_documents(
    state: dict[str, Any],
) -> list[dict]:

    candidates = (
        state.get(
            "reranked_results"
        )
        or
        state.get(
            "retrieved_documents"
        )
        or
        state.get(
            "retrieved_chunks"
        )
        or
        []
    )


    if not isinstance(
        candidates,
        list,
    ):

        return []


    return [
        item
        for item in candidates
        if isinstance(
            item,
            dict,
        )
    ]


# ============================================================
# ANALYSIS AGENT
# ============================================================

async def analysis_agent(
    state: dict[str, Any],
) -> dict:

    query = (
        state.get("active_query")
        or
        state.get("query")
        or
        ""
    ).strip()


    if not query:

        raise ValueError(
            "Analysis agent received empty query."
        )


    documents = _get_documents(
        state
    )


    context = (
        state.get("context")
        or
        ""
    ).strip()


    if (
        not context
        and
        documents
    ):

        context = build_context(
            documents
        )


    # No LLM call when retrieval found nothing.
    if not context:

        return {
            "analysis":
                "No retrieved evidence was available.",

            "analysis_result": {
                "summary":
                    "No retrieved evidence was available.",

                "key_findings":
                    [],

                "evidence_strength":
                    "insufficient",
            },

            "evidence":
                [],

            "context":
                "",
        }


    system_prompt = """
You are the evidence analysis agent in a retrieval-augmented research system.

Analyze only the supplied evidence.

Return valid JSON only:

{
  "summary": "short factual analysis",
  "key_findings": [
    "finding 1",
    "finding 2"
  ],
  "evidence_strength": "strong|moderate|weak|insufficient"
}

Rules:
- Do not invent information.
- Do not use outside knowledge.
- Mention uncertainty when the evidence is incomplete.
- Keep the response concise.
"""


    user_prompt = f"""
Research question:

{query}

Retrieved evidence:

{context}
"""


    result = await generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=350,
        temperature=0.0,
    )


    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Analysis agent returned invalid JSON."
        )


    findings = result.get(
        "key_findings",
        [],
    )


    if not isinstance(
        findings,
        list,
    ):

        findings = []


    summary = str(
        result.get(
            "summary",
            "",
        )
    ).strip()


    return {
        "analysis":
            summary,

        "analysis_result":
            result,

        "evidence":
            findings,

        "context":
            context,

        "status":
            "analyzed",
    }


# Compatibility alias.
analyze_agent = analysis_agent