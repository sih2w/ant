import numpy as np
from scripts.state_action_functions import get_policy
from scripts.types import *
from scripts.constants import *


def q_learning(
        state_actions: StateActions,
        agent_name: AgentName,
        old_state: State,
        new_state: State,
        selected_action_index: int,
        previous_action_index: int,
        reward: float,
        epsilon: float,
) -> None:
    old_policy = get_policy(state_actions, agent_name, old_state)
    new_policy = get_policy(state_actions, agent_name, new_state)

    predict = old_policy["actions"][selected_action_index]
    target = reward + DISCOUNT_FACTOR_GAMMA * np.max(new_policy["actions"])

    old_policy["actions"][selected_action_index] += LEARNING_RATE_ALPHA * (target - predict)


def sarsa_learning(
        state_actions: StateActions,
        agent_name: AgentName,
        old_state: State,
        new_state: State,
        selected_action_index: int,
        previous_action_index: int,
        reward: float,
        epsilon: float,
) -> None:
    old_policy = get_policy(state_actions, agent_name, old_state)
    new_policy = get_policy(state_actions, agent_name, new_state)

    predict = old_policy["actions"][previous_action_index]
    target = reward + DISCOUNT_FACTOR_GAMMA * new_policy["actions"][selected_action_index]

    old_policy["actions"][previous_action_index] += LEARNING_RATE_ALPHA * (target - predict)
