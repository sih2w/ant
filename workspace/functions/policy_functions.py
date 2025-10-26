import copy
import math
import numpy as np
from workspace.types import *
from workspace.enums import PolicyAttribute, EpisodeAttribute


def policy_factory() -> Policy:
    return [[0.00] * 4, False, False, False]


def gridded_policy_factory(grid_width: int, grid_height: int) -> GriddedPolicy:
    return [[policy_factory() for _ in range(grid_width)] for _ in range(grid_height)]


def state_actions_factory(grid_width: int, grid_height: int, agent_count: int) -> StateActions:
    return {
        "returning": [gridded_policy_factory(grid_width, grid_height)] * agent_count,
        "searching": [{}] * agent_count,
    }


def get_policy(
        state_actions: StateActions,
        agent_index: int,
        state: State,
        grid_width: int,
        grid_height: int
) -> Policy:
    if state["carrying_food"]:
        source = state_actions["returning"][agent_index]
        agent_location = state["agent_location"]
        policy = source[agent_location[1]][agent_location[0]]
    else:
        source = state_actions["searching"][agent_index].setdefault(state["food_locations"], gridded_policy_factory(grid_width, grid_height))
        agent_location = state["agent_location"]
        policy = source[agent_location[1]][agent_location[0]]

    return policy


def choose_epsilon_action(
        state_actions: StateActions,
        rng: np.random.Generator,
        agent_index: int,
        state: State,
        epsilon: float,
        grid_width: int,
        grid_height: int
) -> int:
    if rng.random() > epsilon:
        actions = get_policy(
            state_actions=state_actions,
            agent_index=agent_index,
            state=state,
            grid_width=grid_width,
            grid_height=grid_height
        )[PolicyAttribute.ACTIONS.value]
        return int(np.argmax(actions))
    else:
        return int(rng.integers(0, 4))


def update_policy(
        state_actions: StateActions,
        agent_index: int,
        old_state: State,
        new_state: State,
        selected_action_index: int,
        reward: float,
        discount_factor_gamma: float,
        learning_rate_alpha: float,
        grid_width: int,
        grid_height: int,
) -> None:
    old_policy = get_policy(
        state_actions=state_actions,
        agent_index=agent_index,
        state=old_state,
        grid_width=grid_width,
        grid_height=grid_height
    )
    new_policy = get_policy(
        state_actions=state_actions,
        agent_index=agent_index,
        state=new_state,
        grid_width=grid_width,
        grid_height=grid_height
    )

    predict = old_policy[PolicyAttribute.ACTIONS.value][selected_action_index]
    target = reward + discount_factor_gamma * max(new_policy[PolicyAttribute.ACTIONS.value])
    old_policy[PolicyAttribute.ACTIONS.value][selected_action_index] += learning_rate_alpha * (target - predict)

    return None


def update_policy_use(
        episode: Episode,
        states: List[State],
        state_actions: StateActions,
        grid_width: int,
        grid_height: int,
) -> None:
    for agent_index, state in enumerate(states):
        policy = get_policy(
            state_actions=state_actions,
            agent_index=agent_index,
            state=state,
            grid_width=grid_width,
            grid_height=grid_height
        )
        if not policy[PolicyAttribute.USED.value] and (policy[PolicyAttribute.AVERAGED.value] or policy[PolicyAttribute.GIVEN.value]):
            if state["carrying_food"]:
                episode[EpisodeAttribute.USED_RETURN_POLICIES.value] += 1
            else:
                episode[EpisodeAttribute.USED_SEARCH_POLICIES.value] += 1
        policy[PolicyAttribute.GIVEN.value] = True

    return None


def get_decided_actions(
        state_actions: StateActions,
        states: List[State],
        grid_width: int,
        grid_height: int,
) -> List[int]:
    decided_actions = []
    for agent_index, state in enumerate(states):
        actions = get_policy(
            state_actions=state_actions,
            agent_index=agent_index,
            state=state,
            grid_width=grid_width,
            grid_height=grid_height
        )[PolicyAttribute.ACTIONS.value]
        decided_actions.append(int(np.argmax(actions)))

    return decided_actions


def get_training_actions(
        state_actions: StateActions,
        states: List[State],
        epsilon: float,
        rng: np.random.Generator,
        grid_width: int,
        grid_height: int,
) -> List[int]:
    training_actions = []
    for agent_index, state in enumerate(states):
        training_actions.append(
            choose_epsilon_action(
                state_actions=state_actions,
                rng=rng,
                agent_index=agent_index,
                state=state,
                epsilon=epsilon,
                grid_width=grid_width,
                grid_height=grid_height,
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
    if not target[PolicyAttribute.USED.value]:
        target[PolicyAttribute.ACTIONS.value] = copy.copy(source[PolicyAttribute.ACTIONS.value])
        target[PolicyAttribute.GIVEN.value] = True
        return True

    return False


def average_policies(source: Policy, target: Policy) -> None:
    source[PolicyAttribute.AVERAGED.value] = True
    target[PolicyAttribute.AVERAGED.value] = True
    for index, value in enumerate(source[PolicyAttribute.ACTIONS.value]):
        source[PolicyAttribute.ACTIONS.value][index] = (value + target[PolicyAttribute.ACTIONS.value][index]) / 2
        target[PolicyAttribute.ACTIONS.value][index] = source[PolicyAttribute.ACTIONS.value][index]

    return None


def get_search_gridded_policy(
        state_actions: StateActions,
        agent_index: int,
        food_locations: FoodLocations,
        grid_width: int,
        grid_height: int,
) -> GriddedPolicy:
    gridded_policy = state_actions["searching"][agent_index].setdefault(food_locations, gridded_policy_factory(grid_width, grid_height))
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
        episode[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=True,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    elif not agent_1_state["carrying_food"] and not agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 searching for food.
        episode[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"], grid_width, grid_height),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"], grid_width, grid_height),
            average_both=True,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    elif not agent_1_state["carrying_food"] and agent_2_state["carrying_food"]:
        # Agent 1 searching for food. Agent 2 returning to nest.
        episode[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_1_index),
            target=get_return_gridded_policy(state_actions, agent_2_index),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )

        episode[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"], grid_width, grid_height),
            target=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"], grid_width, grid_height),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )
    else:
        # Agent 1 returning to nest. Agent 2 searching for food.
        episode[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] += exchange_policy(
            source=get_return_gridded_policy(state_actions, agent_2_index),
            target=get_return_gridded_policy(state_actions, agent_1_index),
            average_both=False,
            grid_width=grid_width,
            grid_height=grid_height,
        )

        episode[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] += exchange_policy(
            source=get_search_gridded_policy(state_actions, agent_1_index, agent_1_state["food_locations"], grid_width, grid_height),
            target=get_search_gridded_policy(state_actions, agent_2_index, agent_2_state["food_locations"], grid_width, grid_height),
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