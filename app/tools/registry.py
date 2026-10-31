from app.tools.api_tool import get_json_api
from app.tools.calculator import safe_calculate
from app.tools.sql_tool import execute_sql
from app.tools.web_search import browser_research


TOOLS = {
    "calculator": safe_calculate,
    "sql": execute_sql,
    "api": get_json_api,
    "web_search": browser_research,
}


def get_tool(tool_name: str):
    return TOOLS.get(tool_name)
