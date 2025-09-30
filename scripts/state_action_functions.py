import copy
import numpy as np
from scripts.types import *
from scripts.constants import *


def _is_shared(key, shared_map, agent_map) -> bool:
    return key in shared_map and key in agent_map and np.array_equal(shared_map[key], agent_map[key])


def _maybe_clone_shared(shared_map, agent_map, key):
    if key in shared_map and key not in agent_map:
        agent_map[key] = copy.deepcopy(shared_map[key])
        return agent_map[key]
    return None


def get_returning_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    state: State
) -> Tuple[Policy, bool]:
    shared_map = state_actions["returning"]["shared"]
    agent_map = state_actions["returning"][agent_name]

    shared = _is_shared(state["agent_location"], shared_map, agent_map)
    policy = _maybe_clone_shared(shared_map, agent_map, state["agent_location"])

    if policy is None:
        policy = agent_map[state["agent_location"]]

    return policy, shared


def get_searching_policy(
    state_actions: StateActions,
    agent_name: AgentName,
    state: State
) -> Tuple[Policy, bool]:
    shared_map = state_actions["searching"]["shared"][state["agent_location"]]
    agent_map = state_actions["searching"][agent_name][state["agent_location"]]

    shared = _is_shared(state["food_locations"], shared_map, agent_map)
    policy = _maybe_clone_shared(shared_map, agent_map, state["food_locations"])

    if policy is None:
        policy = agent_map[state["food_locations"]]

    return policy, shared


def get_actions(
        state_actions: StateActions,
        agent_name: AgentName,
        state: State
) -> (Actions, bool):
    if state["carrying_food"]:
        policy, shared = get_returning_policy(state_actions, agent_name, state)
    else:
        policy, shared = get_searching_policy(state_actions, agent_name, state)

    return policy["actions"], shared


def choose_epsilon_action(
        state_actions: StateActions,
        rng: np.random.Generator,
        agent_name: AgentName,
        state: State,
        epsilon: float
) -> Tuple[int, bool]:
    if rng.random() > epsilon:
        actions, shared = get_actions(state_actions, agent_name, state)
        return int(np.argmax(actions)), shared
    else:
        return rng.integers(0, ACTION_COUNT), False


def get_decided_actions(
        state_actions: StateActions,
        states: Dict[AgentName, State]
) -> Dict[AgentName, int]:
    decided_actions = {}
    for agent_name, state in states.items():
        actions, _ = get_actions(state_actions, agent_name, state)
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
        action, shared = choose_epsilon_action(state_actions, rng, agent_name, state, epsilon)
        training_actions[agent_name] = action

        if shared:
            if state["carrying_food"]:
                episode["return_exchange_use_count"] += 1
            else:
                episode["search_exchange_use_count"] += 1

    return training_actions