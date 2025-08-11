type FoodPositions = tuple[int, ...]
type AgentLocation = tuple[int, int]
type Actions = list[int, ...]
type AgentName = str
type EpisodeData = {
    "steps": [int],
    "rewards": [{AgentName: float}]
}
type StateActions = {
    "return_policy": {
        AgentName: {
            AgentLocation: Actions
        }
    },
    "search_policy": {
        AgentName: {
            AgentLocation: {
                FoodPositions: Actions
            }
        }
    }
}
type Observation = {
    "agent_position": tuple[int, int],
    "carrying_food": bool,
    "carried_food": tuple[int, ...],
    "food_positions": tuple[int, ...],
    "agent_detected": bool
}