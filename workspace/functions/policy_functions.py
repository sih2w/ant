import math
import numpy as np
from workspace.shared.types import *
from workspace.shared.enums import *
from workspace.shared.run_settings import *
from workspace.classes.environment import Environment


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
        exchange_callback: ExchangeCallback
) -> int:
    exchange_count = 0
    for row in range(GRID_HEIGHT):
        for column in range(GRID_WIDTH):
            source_policy = source[row][column]
            target_policy = target[row][column]
            success = exchange_callback(source_policy, target_policy)
            if success:
                exchange_count += 1

    return exchange_count


def exchange(
        environment: Environment,
        state_actions: StateActions,
        episode: Episode
) -> None:
    environment_state = environment.get_environment_state()
    food_locations = tuple(environment_state["food_locations"])
    exchanges = 0

    for index1, agent1 in enumerate(environment.get_agents()):
        for index2, agent2 in enumerate(environment.get_agents()):
            if index1 == index2:
                continue

            if environment.carrying_food(index1) and environment.carrying_food(index2):
                exchanges += exchange_policy(
                    source=get_return_gridded_policy(state_actions, index1),
                    target=get_return_gridded_policy(state_actions, index2),
                    exchange_callback=agent1["exchange_callback"]
                )
            elif not environment.carrying_food(index1) and not environment.carrying_food(index2):
                exchanges += exchange_policy(
                    source=get_search_gridded_policy(state_actions, index1, food_locations),
                    target=get_search_gridded_policy(state_actions, index2, food_locations),
                    exchange_callback=agent1["exchange_callback"]
                )

    episode[EpisodeAttr.EXCHANGES.value] += exchanges

    return None