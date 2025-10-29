from os import environ


environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


from workspace.classes.environment import Environment
from workspace.functions.plot_functions import plot_episodes
from workspace.functions.draw_functions import test
from workspace.functions.train_functions import train
from workspace.functions.data_functions import load_data, save_data
from workspace.shared.run_settings import *


if __name__ == "__main__":
    environment = Environment()

    try:
        state_actions, episodes = load_data()
    except FileNotFoundError:
        state_actions, episodes = train(environment)

        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episodes)

    plot_episodes(episodes)

    if SHOW_AFTER_TRAINING:
        test(state_actions, environment)