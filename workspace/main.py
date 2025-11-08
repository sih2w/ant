from os import environ
from workspace.shared.enums import PolicyAttr

environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


from workspace.classes.environment import Environment
from workspace.functions.plot_functions import plot_episodes
from workspace.functions.draw_functions import test
from workspace.functions.train_functions import train
from workspace.functions.data_functions import load_data, save_data
from workspace.shared.run_settings import *
from workspace.shared.types import *


def has_prior_food_been_deposited(environment_state: EnvironmentState, food_index: int) -> bool:
    for other_food_index, deposited in enumerate(environment_state["deposited_food"]):
        if other_food_index < food_index and not deposited:
            return False
    return True


def has_before_food_been_deposited(environment_state: EnvironmentState, food_index: int) -> bool:
    for other_food_index, deposited in enumerate(environment_state["deposited_food"]):
        if other_food_index > food_index and not deposited:
            return False
    return True


def average_policies(source: Policy, target: Policy) -> bool:
    for index, value in enumerate(source[PolicyAttr.ACTIONS.value]):
        source[PolicyAttr.ACTIONS.value][index] = (value + target[PolicyAttr.ACTIONS.value][index]) / 2
        target[PolicyAttr.ACTIONS.value][index] = source[PolicyAttr.ACTIONS.value][index]
    return True


if __name__ == "__main__":
    environment = Environment()

    if AGENT_COUNT == 1:
        environment.register_food_pickup_callbacks([has_prior_food_been_deposited])
    elif AGENT_COUNT == 2:
        environment.register_food_pickup_callbacks([
            has_prior_food_been_deposited,
            has_before_food_been_deposited
        ])
    else:
        environment.register_food_pickup_callbacks([has_prior_food_been_deposited] * AGENT_COUNT)

    environment.register_action_override_callbacks([])
    environment.register_exchange_callbacks([average_policies] * AGENT_COUNT)

    try:
        state_actions, episodes = load_data()
    except FileNotFoundError:
        state_actions, episodes = train(environment)

        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episodes)

    plot_episodes(episodes)

    if SHOW_AFTER_TRAINING:
        test(state_actions, environment)