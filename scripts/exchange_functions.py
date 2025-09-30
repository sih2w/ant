import copy
import math
from scripts.constants import *
from scripts.types import *


def are_close_enough(
        agent_1_location: Location,
        agent_2_location: Location
) -> bool:
    dx = agent_1_location[0] - agent_2_location[0]
    dy = agent_1_location[1] - agent_2_location[1]
    distance = math.floor(math.hypot(dx, dy))
    return distance <= AGENT_VISION_RADIUS


def share_search_policy(
        state_actions: StateActions,
        from_agent_name: AgentName
) -> int:
    source = state_actions["searching"][from_agent_name]
    destination = state_actions["searching"]["shared"]
    exchange_count = 0

    for agent_location, food_locations_to_policy in source.items():
        for food_location, policy in food_locations_to_policy.items():
            if not food_location in destination[agent_location]:
                destination[agent_location][food_location] = copy.deepcopy(policy)
                exchange_count += 1
    return exchange_count


def share_return_policy(
        state_actions: StateActions,
        from_agent_name: AgentName
) -> int:
    source = state_actions["returning"][from_agent_name]
    destination = state_actions["returning"]["shared"]
    exchange_count = 0

    for agent_location, policy in source.items():
        if not agent_location in destination:
            destination[agent_location] = copy.deepcopy(policy)
            exchange_count += 1
    return exchange_count


def fill_policy_gaps(
        state_actions: StateActions,
        agent_1_name: AgentName,
        agent_2_name: AgentName,
        agent_1_state: State,
        agent_2_state: State,
        episode: Episode,
) -> None:
    if agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_2_name)
    elif not agent_1_state["carrying_food"] and not agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)
    elif not agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name)


def exchange(
        state_actions: StateActions,
        states: Dict[AgentName, State],
        episode: Episode
) -> None:
    length = len(states)
    for index_1 in range(length):
        for index_2 in range(index_1 + 1, length):
            agent_1_name = f"agent_{index_1}"
            agent_2_name = f"agent_{index_2}"
            agent_1_state: State = states[agent_1_name]
            agent_2_state: State = states[agent_2_name]

            close_enough = are_close_enough(
                agent_1_state["agent_location"],
                agent_2_state["agent_location"]
            )

            if close_enough:
                fill_policy_gaps(
                    state_actions,
                    agent_1_name,
                    agent_2_name,
                    agent_1_state,
                    agent_2_state,
                    episode
                )