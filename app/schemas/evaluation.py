from pydantic import BaseModel, Field


class Claim(BaseModel):
    text: str
    importance: float = Field(default=1.0, ge=0.1, le=2.0)


class ClaimList(BaseModel):
    claims: list[Claim] = Field(default_factory=list)
