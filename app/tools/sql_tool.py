from app.storage.postgres import execute_read_only_sql


async def execute_sql(query: str):
    cleaned = query.strip().lower()

    if not cleaned.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    forbidden = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "truncate ",
        "create ",
    ]

    if any(operation in cleaned for operation in forbidden):
        raise ValueError("Unsafe SQL operation detected")

    return await execute_read_only_sql(query)
