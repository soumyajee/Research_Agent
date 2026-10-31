from typing import Any

from app.services.llm_service import (
    generate_json,
)


# ============================================================
# PLANNER AGENT
# ============================================================

async def planner_agent(
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
            "Planner received an empty query."
        )


    system_prompt = """
You are the planning agent inside an Agentic RAG research system.

Your job is only to understand the request and prepare a concise research plan.

Return valid JSON only.

Required JSON format:

{
  "intent": "short intent label",
  "research_plan": [
    "step 1",
    "step 2"
  ],
  "sub_queries": [
    "search/retrieval query 1",
    "search/retrieval query 2"
  ]
}

Rules:
- Do not answer the user's question.
- Keep the plan compact.
- Create at most 3 sub_queries.
- Avoid unnecessary research steps.
"""


    user_prompt = f"""
User research question:

{query}
"""


    result = await generate_json(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_completion_tokens=250,
        temperature=0.0,
    )


    if not isinstance(
        result,
        dict,
    ):

        raise ValueError(
            "Planner returned invalid JSON structure."
        )


    intent = (
        str(
            result.get(
                "intent",
                "research",
            )
        ).strip()
        or
        "research"
    )


    plan = result.get(
        "research_plan",
        [],
    )


    if not isinstance(
        plan,
        list,
    ):

        plan = []


    sub_queries = result.get(
        "sub_queries",
        [],
    )


    if not isinstance(
        sub_queries,
        list,
    ):

        sub_queries = []


    sub_queries = [
        str(item).strip()
        for item in sub_queries
        if str(item).strip()
    ][
        :3
    ]


    if not sub_queries:

        sub_queries = [
            query
        ]


    return {
        "intent":
            intent,

        "plan":
            plan,

        "research_plan":
            plan,

        "sub_queries":
            sub_queries,

        "active_query":
            sub_queries[0],

        "status":
            "planned",
    }


# Compatibility alias if workflow imports plan_agent.
plan_agent = planner_agent