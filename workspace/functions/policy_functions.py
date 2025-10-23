import copy
import math
from collections import defaultdict
from typing import Callable
import numpy as np
from workspace.shared_types import *
from workspace.classes.environment import AGENT_ACTIONS


def policy_factory() -> Policy:
    return {
        "actions": np.zeros(len(AGENT_ACTIONS)).tolist(),
        "shared": False,
        "used": False
    }


def create_grid(
        callback: Callable[[], T],
        grid_width: int,
        grid_height: int,
) -> List[List[T]]:
    grid = []
    for row in range(grid_height):
        new_row = []
        for column in range(grid_width):
            new_row.append(callback())
        grid.append(new_row)
    return grid


def state_actions_factory(grid_width: int, grid_height: int) -> StateActions:
    return {
        "returning": defaultdict(lambda: create_grid(lambda: policy_factory(), grid_width, grid_height)),
        "searching": defaultdict(lambda: defaultdict(lambda: create_grid(lambda: policy_factory(), grid_width, grid_height))),
    }


def get_policy(
        state_actions: StateActions,
        agent_index: int,
        state: State
) -> Policy:
    if state["carrying_food"]:
        source = state_actions["returning"][agent_index]
        agent_location = state["agent_location"]
        policy = source[agent_location[1]][agent_location[0]]
    else:
        source = state_actions["searching"][agent_index][state["food_locations"]]
        agent_location = state["agent_location"]
        policy = source[agent_location[1]][agent_location[0]]
    return policy


def choose_epsilon_action(
        state_actions: StateActions,
        rng: np.random.Generator,
        agent_index: int,
        state: State,
        epsilon: float
) -> int:
    if rng.random() > epsilon:
        actions = get_policy(state_actions, agent_index, state)["actions"]
        return int(np.argmax(actions))
    else:
        return int(rng.integers(0, len(AGENT_ACTIONS)))


def update_policy(
        state_actions: StateActions,
        agent_index: int,
        old_state: State,
        new_state: State,
        selected_action_index: int,
        reward: float,
        discount_factor_gamma: float,
        learning_rate_alpha: float,
) -> None:
    old_policy = get_policy(state_actions, agent_index, old_state)
    new_policy = get_policy(state_actions, agent_index, new_state)

    predict = old_policy["actions"][selected_action_index]
    target = reward + discount_factor_gamma * np.max(new_policy["actions"])

    old_policy["actions"][selected_action_index] += learning_rate_alpha * (target - predict)
    return None

def update_policy_use(
        episode: Episode,
        state: State,
        agent_index: int,
        state_actions: StateActions,
) -> None:
    policy = get_policy(state_actions, agent_index, state)
    if policy["shared"] and not policy["used"]:
        policy["used"] = True
        if state["carrying_food"]:
            episode["return_exchange_use_count"] += 1
        else:
            episode["search_exchange_use_count"] += 1
    return None


def get_decided_actions(
        state_actions: StateActions,
        states: List[State]
) -> List[int]:
    decided_actions = []
    for agent_index, state in enumerate(states):
        actions = get_policy(state_actions, agent_index, state)["actions"]
        decided_actions.append(int(np.argmax(actions)))
    return decided_actions


def get_training_actions(
        state_actions: StateActions,
        states: List[State],
        epsilon: float,
        rng: np.random.Generator
) -> List[int]:
    training_actions = []
    for agent_index, state in enumerate(states):
        training_actions.append(
            choose_epsilon_action(
                state_actions=state_actions,
                rng=rng,
                agent_index=agent_index,
                state=state,
                epsilon=epsilon
            )
        )
    return training_actions


def close_enough(
        agent_1_location: Location,
        agent_2_location: Location,
        agent_vision_radius: float
) -> bool:
    dx = agent_1_location[0] - agent_2_location[0]
    dy = agent_1_location[1] - agent_2_location[1]
    distance = math.floor(math.hypot(dx, dy))
    return distance <= agent_vision_radius


def try_give_policy(source: Policy, target: Policy) -> bool:
    if not target["used"]:
        target["actions"] = copy.copy(source["actions"])
        target["shared"] = True
        return True
    return False


def average_policies(source: Policy, target: Policy) -> None:
    source["shared"] = True
    target["shared"] = True
    for index, value in enumerate(source["actions"]):
        source["actions"][index] = (value + target["actions"][index]) / 2
        target["actions"][index] = source["actions"][index]
    return None


def get_search_gridded_policy(
        state_actions: StateActions,
        agent_index: int,
        food_locations: FoodLocations,
) -> GriddedPolicy:
    gridded_policy = state_actions["searching"][agent_index][food_locations]
    return gridded_policy


def get_return_gridded_policy(
        state_actions: StateActions,
        agent_index: int
) -> GriddedPolicy:
    gridded_policy = state_actions["returning"][agent_index]
    return gridded_policy


def exchange_policy(
        source: GriddedPolicy,
        target: GriddedPolicy,
        average_both: bool,
        grid_width: int,
        grid_height: int,
) -> int:
    exchange_count = 0
    for row in range(grid_height):
        for column in range(grid_width):
            source_policy = source[row][column]
            target_policy = target[row][column]
            if average_both:
                average_policies(source_policy, target_policy)
                exchange_count += 1
            else:
                success = try_give_policy(source_policy, target_policy)
                if success:
                    exchange_count += 1
    return exchange_count


def fill_policy_gaps(
        state_actions: StateActions,
        agent_1_index: int,
        agent_2_index: int,
        agent_1_state: State,
        agent_2_state: State,
        episode: Episode,
        grid_width: int,
        grid_height: int,
) -> None:
    if agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        episode["return_exchange_count"] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=True,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    elif not agent_1_state["carrying_food"] and not agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode["search_exchange_count"] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            average_both=True,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    elif not agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode["return_exchange_count"] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )

        episode["search_exchange_count"] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode["return_exchange_count"] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_2_index),
            target=get_return_gridded_policy(state_actions, agent_1_index),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )

        episode["search_exchange_count"] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )


def exchange(
        state_actions: StateActions,
        states: List[State],
        episode: Episode,
        agent_vision_radius: float,
        grid_width: int,
        grid_height: int,
) -> None:
    length = len(states)
    for index_1 in range(length):
        for index_2 in range(index_1 + 1, length):
            agent_1_state: State = states[index_1]
            agent_2_state: State = states[index_2]

            if close_enough(
                agent_1_location=agent_1_state["agent_location"],
                agent_2_location=agent_2_state["agent_location"],
                agent_vision_radius=agent_vision_radius,
            ):
                fill_policy_gaps(
                    state_actions=state_actions,
                    agent_1_index=index_1,
                    agent_2_index=index_2,
                    agent_1_state=agent_1_state,
                    agent_2_state=agent_2_state,
                    episode=episode,
                    grid_width=grid_width,
                    grid_height=grid_height,
                )
    return None