from pydantic import BaseModel, Field


class ResearchTask(BaseModel):
    objective: str
    source_types: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    intent: str
    search_queries: list[str] = Field(default_factory=list)
    tasks: list[ResearchTask] = Field(default_factory=list)
    needs_web: bool = True
    needs_internal: bool = True
    needs_analysis: bool = False


class Calculation(BaseModel):
    description: str
    expression: str


class AnalysisPlan(BaseModel):
    calculations: list[Calculation] = Field(default_factory=list)
