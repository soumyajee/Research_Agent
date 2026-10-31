from typing import Literal

from pydantic import BaseModel


class ResearchRequest(BaseModel):
    query: str
    tenant_id: str = "default"


class HumanReviewRequest(BaseModel):
    action: Literal["approve", "reject", "research_more"]
    feedback: str | None = None
