def confidence_router(state):
    return state.get("next_action", "human_review")
