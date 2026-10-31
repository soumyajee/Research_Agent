import httpx


async def get_json_api(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
):
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
