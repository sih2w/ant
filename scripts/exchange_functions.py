import copy
import math
from typing import Any
from scripts.constants import *
from scripts.types import *


def set_used_if_given(policy: Policy) -> bool:
    if policy["given"] and not policy["used"]:
        policy["used"] = True
        return True
    return False


def are_close_enough(
        agent_1_location: Location,
        agent_2_location: Location
) -> bool:
    dx = agent_1_location[0] - agent_2_location[0]
    dy = agent_1_location[1] - agent_2_location[1]
    distance = math.floor(math.hypot(dx, dy))
    return distance <= AGENT_VISION_RADIUS


def all_equals(iterable: List or Tuple, value: Any) -> bool:
    for item in iterable:
        if item != value:
            return False
    return True


def try_give_policy(source_policy: Policy, target_policy: Policy) -> bool:
    if all_equals(source_policy["actions"], 0.00):
        # Don't consider giving a policy that has not been developed yet.
        return False

    if all_equals(target_policy["actions"], 0.00):
        # Overwrite the non-developed target policy with the developed source policy.
        target_policy["actions"] = copy.copy(source_policy["actions"])
        target_policy["given"] = True
        return True

    return False


def share_search_policy(
        state_actions: StateActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName,
        current_food_locations: FoodLocations,
) -> int:
    source = state_actions["searching"][from_agent_name][current_food_locations]
    destination = state_actions["searching"][to_agent_name][current_food_locations]
    exchange_count = 0

    for row, columns in enumerate(source):
        for column, source_policy in enumerate(columns):
            target_policy = destination[row][column]
            success = try_give_policy(source_policy, target_policy)
            if success:
                exchange_count += 1

    return exchange_count


def share_return_policy(
        state_actions: StateActions,
        from_agent_name: AgentName,
        to_agent_name: AgentName
) -> int:
    source = state_actions["returning"][from_agent_name]
    destination = state_actions["returning"][to_agent_name]
    exchange_count = 0

    for row, columns in enumerate(source):
        for column, source_policy in enumerate(columns):
            target_policy = destination[row][column]
            success = try_give_policy(source_policy, target_policy)
            if success:
                exchange_count += 1

    return exchange_count


def fill_policy_gaps(
        state_actions: StateActions,
        agent_1_name: AgentName,
        agent_2_name: AgentName,
        agent_1_state: State,
        agent_2_state: State,
        episode: Episode
) -> None:
    current_food_locations = agent_1_state["food_locations"]  # Food locations are the same for all agents.
    if agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name, agent_2_name)
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_2_name, agent_1_name)
    elif not agent_1_state["carrying_food"] and not agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_1_name, agent_2_name, current_food_locations)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name, agent_1_name, current_food_locations)
    elif not agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name, agent_2_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name, agent_1_name, current_food_locations)
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode["return_exchange_count"] += share_return_policy(state_actions, agent_1_name, agent_2_name)
        episode["search_exchange_count"] += share_search_policy(state_actions, agent_2_name, agent_1_name, current_food_locations)


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