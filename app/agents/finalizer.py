async def finalizer_agent(state):
    return {
        "final_answer": state.get("generated_answer", ""),
        "requires_human_review": False,
        "status": "completed",
    }
