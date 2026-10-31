from app.core.groq_client import groq_client


async def browser_research(query: str):
    prompt = f"""
Research the following query using reliable web sources.

Query:
{query}

Prefer official websites, government data, academic publications,
company reports, and high-quality publications.

Return a concise evidence-focused research result with the citations
provided by the browser-search tool.
"""
    return await groq_client.browser_search(prompt)
