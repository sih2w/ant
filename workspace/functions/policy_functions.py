import copy
import math
import numpy as np
from workspace.shared.types import *
from workspace.shared.enums import *
from workspace.shared.run_settings import *


def policy_factory() -> Policy:
    return [[0.00] * ACTION_COUNT, False, False, False]


def gridded_policy_factory() -> GriddedPolicy:
    return [[policy_factory() for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]


def state_actions_factory() -> StateActions:
    return {
        "returning": [gridded_policy_factory() for _ in range(AGENT_COUNT)],
        "searching": [{} for _ in range(AGENT_COUNT)],
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
        source = state_actions["searching"][agent_index].setdefault(state["food_locations"], gridded_policy_factory())
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
        actions = get_policy(state_actions, agent_index, state)[PolicyAttr.ACTIONS.value]
        return int(np.argmax(actions))
    else:
        return int(rng.integers(0, 4))


def update_policy(
        state_actions: StateActions,
        agent_index: int,
        old_state: State,
        new_state: State,
        selected_action_index: int,
        reward: float
) -> None:
    old_policy = get_policy(state_actions, agent_index, old_state)
    new_policy = get_policy(state_actions, agent_index, new_state)

    predict = old_policy[PolicyAttr.ACTIONS.value][selected_action_index]
    target = reward + DISCOUNT_FACTOR_GAMMA * max(new_policy[PolicyAttr.ACTIONS.value])

    old_policy[PolicyAttr.ACTIONS.value][selected_action_index] += LEARNING_RATE_ALPHA * (target - predict)

    return None


def update_policy_use(
        episode: Episode,
        states: List[State],
        state_actions: StateActions
) -> None:
    for agent_index, state in enumerate(states):
        policy = get_policy(state_actions, agent_index, state)
        if not policy[PolicyAttr.USED.value] and (policy[PolicyAttr.AVERAGED.value] or policy[PolicyAttr.GIVEN.value]):
            if state["carrying_food"]:
                episode[EpisodeAttr.USED_RETURN_POLICIES.value] += 1
            else:
                episode[EpisodeAttr.USED_SEARCH_POLICIES.value] += 1
        policy[PolicyAttr.USED.value] = True

    return None


def get_decided_actions(
        state_actions: StateActions,
        states: List[State]
) -> List[int]:
    decided_actions = []
    for agent_index, state in enumerate(states):
        actions = get_policy(state_actions, agent_index, state)[PolicyAttr.ACTIONS.value]
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
        agent_2_location: Location
) -> bool:
    dx = agent_1_location[0] - agent_2_location[0]
    dy = agent_1_location[1] - agent_2_location[1]
    distance = math.floor(math.hypot(dx, dy))

    return distance <= AGENT_VISION_RADIUS


def try_give_policy(source: Policy, target: Policy) -> bool:
    if not target[PolicyAttr.USED.value]:
        target[PolicyAttr.ACTIONS.value] = copy.copy(source[PolicyAttr.ACTIONS.value])
        target[PolicyAttr.GIVEN.value] = True
        return True

    return False


def average_policies(source: Policy, target: Policy) -> None:
    source[PolicyAttr.AVERAGED.value] = True
    target[PolicyAttr.AVERAGED.value] = True
    for index, value in enumerate(source[PolicyAttr.ACTIONS.value]):
        source[PolicyAttr.ACTIONS.value][index] = (value + target[PolicyAttr.ACTIONS.value][index]) / 2
        target[PolicyAttr.ACTIONS.value][index] = source[PolicyAttr.ACTIONS.value][index]

    return None


def get_search_gridded_policy(
        state_actions: StateActions,
        agent_index: int,
        food_locations: FoodLocations
) -> GriddedPolicy:
    location_gridded_policies = state_actions["searching"][agent_index]
    gridded_policy = location_gridded_policies.setdefault(food_locations, gridded_policy_factory())
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
        average_both: bool
) -> int:
    exchange_count = 0
    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
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
        episode: Episode
) -> None:
    if agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 returning to nest. Agent 2 returning to nest.
        episode[EpisodeAttr.AVERAGED_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=True
        )
    elif not agent_1_state["carrying_food"] and not agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            average_both=True
        )
    elif not agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode[EpisodeAttr.GIVEN_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=False
        )

        episode[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            average_both=False
        )
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode[EpisodeAttr.GIVEN_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_2_index),
            target=get_return_gridded_policy(state_actions, agent_1_index),
            average_both=False
        )

        episode[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"]),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"]),
            average_both=False
        )


def exchange(
        state_actions: StateActions,
        states: List[State],
        episode: Episode
) -> None:
    length = len(states)
    for index_1 in range(length):
        for index_2 in range(index_1 + 1, length):
            agent_1_state: State = states[index_1]
            agent_2_state: State = states[index_2]

            if close_enough(
                agent_1_location=agent_1_state["agent_location"],
                agent_2_location=agent_2_state["agent_location"]
            ):
                fill_policy_gaps(
                    state_actions=state_actions,
                    agent_1_index=index_1,
                    agent_2_index=index_2,
                    agent_1_state=agent_1_state,
                    agent_2_state=agent_2_state,
                    episode=episode
                )

    return None