import asyncio

from app.config import settings
from app.ingestion.pipeline import ingest_text
from app.tools.web_search import browser_research


async def discovery_agent(state):
    if not state.get("needs_web", True):
        return {"discovered_sources": [], "status": "discovery_skipped"}

    search_queries = state.get("search_queries") or [state["query"]]
    search_queries = search_queries[: settings.max_web_searches]

    results = await asyncio.gather(
        *[browser_research(query) for query in search_queries],
        return_exceptions=True,
    )

    discovered_sources = []
    ingestion_jobs = []

    for index, result in enumerate(results):
        if isinstance(result, Exception):
            continue

        query = search_queries[index]
        discovered_sources.append(
            {
                "type": "browser_search",
                "query": query,
                "content": result,
            }
        )

        ingestion_jobs.append(
            ingest_text(
                tenant_id=state["tenant_id"],
                title=f"Web Research: {query}",
                text=result,
                source_type="web_search",
                source_url=None,
                extra_metadata={"search_query": query},
            )
        )

    if ingestion_jobs:
        await asyncio.gather(*ingestion_jobs, return_exceptions=True)

    return {
        "discovered_sources": discovered_sources,
        "status": "sources_discovered",
    }
