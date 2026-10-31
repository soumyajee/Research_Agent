from langgraph.graph import END, START, StateGraph

from app.agents.analysis import analysis_agent
from app.agents.claim_extractor import claim_extractor_agent
from app.agents.confidence import confidence_agent
from app.agents.discovery import discovery_agent
from app.agents.finalizer import finalizer_agent
from app.agents.grounding import grounding_agent
from app.agents.human_review import human_review_agent
from app.agents.planner import planner_agent
from app.agents.retrieval_agent import retrieval_agent
from app.agents.synthesis import synthesis_agent
from app.graph.router import confidence_router
from app.graph.state import ResearchState


builder = StateGraph(ResearchState)

builder.add_node("planner", planner_agent)
builder.add_node("discovery", discovery_agent)
builder.add_node("retrieval", retrieval_agent)
builder.add_node("analysis", analysis_agent)
builder.add_node("synthesis", synthesis_agent)
builder.add_node("claim_extraction", claim_extractor_agent)
builder.add_node("grounding", grounding_agent)
builder.add_node("confidence", confidence_agent)
builder.add_node("finalize", finalizer_agent)
builder.add_node("human_review", human_review_agent)

builder.add_edge(START, "planner")
builder.add_edge("planner", "discovery")
builder.add_edge("discovery", "retrieval")
builder.add_edge("retrieval", "analysis")
builder.add_edge("analysis", "synthesis")
builder.add_edge("synthesis", "claim_extraction")
builder.add_edge("claim_extraction", "grounding")
builder.add_edge("grounding", "confidence")

builder.add_conditional_edges(
    "confidence",
    confidence_router,
    {
        "finalize": "finalize",
        "retry": "retrieval",
        "human_review": "human_review",
    },
)

builder.add_edge("finalize", END)
builder.add_edge("human_review", END)

research_graph = builder.compile()
