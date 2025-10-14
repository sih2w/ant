import numpy as np

from scripts.exchange_functions import set_used_if_given
from scripts.types import *
from scripts.constants import *


def get_returning_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    state: State
) -> Policy:
    source = state_actions["returning"][agent_name]
    agent_location = state["agent_location"]
    policy = source[agent_location[1]][agent_location[0]]
    return policy


def get_searching_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    state: State
) -> Policy:
    source = state_actions["searching"][agent_name]
    agent_location = state["agent_location"]
    dictionary = source[agent_location[1]][agent_location[0]]
    policy = dictionary[state["food_locations"]]
    return policy


def get_policy(
        state_actions: StateActions,
        agent_name: AgentName,
        state: State
) -> Policy:
    if state["carrying_food"]:
        return get_returning_policy(state_actions, agent_name, state)
    else:
        return get_searching_policy(state_actions, agent_name, state)


def choose_epsilon_action(
        state_actions: StateActions,
        rng: np.random.Generator,
        agent_name: AgentName,
        state: State,
        epsilon: float
) -> int:
    if rng.random() > epsilon:
        actions = get_policy(state_actions, agent_name, state)["actions"]
        return int(np.argmax(actions))
    else:
        return rng.integers(0, ACTION_COUNT)


def get_decided_actions(
        state_actions: StateActions,
        states: Dict[AgentName, State]
) -> Dict[AgentName, int]:
    decided_actions = {}
    for agent_name, state in states.items():
        actions = get_policy(state_actions, agent_name, state)["actions"]
        decided_actions[agent_name] = int(np.argmax(actions))
    return decided_actions


def get_training_actions(
        state_actions: StateActions,
        states: Dict[AgentName, State],
        epsilon: float,
        rng: np.random.Generator,
        episode: Episode
) -> (Dict[AgentName, int]):
    training_actions = {}
    for agent_name, state in states.items():
        action = choose_epsilon_action(state_actions, rng, agent_name, state, epsilon)
        training_actions[agent_name] = action

        policy = get_policy(state_actions, agent_name, state)
        was_given_used = set_used_if_given(policy)

        if was_given_used:
            if state["carrying_food"]:
                episode["return_exchange_use_count"] += 1
            else:
                episode["search_exchange_use_count"] += 1

    return training_actions