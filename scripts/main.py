from typing import Any
import dill
import os
import matplotlib.pyplot as plt
from scripts.q_learning import train, visualize
from scripts.shared import *


def load_data() -> Tuple[StateActions, List[Episode]]:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "rb") as file:
        data = dill.load(file)
        return data["state_actions"], data["episode_data"]


def save_data(
        state_actions: StateActions,
        episode_data: List[Episode]
) -> None:
    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "wb") as file:
        dill.dump({
            "state_actions": state_actions,
            "episode_data": episode_data
        }, file)


def sparse_episode_data(episode_data: List[Episode]) -> Any:
    new_episode_data = []
    for index in range(0, len(episode_data), GRAPH_STEP):
        new_episode_data.append(episode_data[index])
    return new_episode_data


def plot_episode_data(episode_data: List[Episode]) -> None:
    episode_data = sparse_episode_data(episode_data)
    episodes = [episode * GRAPH_STEP for episode in range(len(episode_data))]

    episode_steps = []
    episode_rewards = []
    episode_search_exchanges = []
    episode_search_exchange_uses = []
    episode_return_exchanges = []
    episode_return_exchange_uses = []

    for episode in episode_data:
        episode_steps.append(episode["steps"])
        episode_rewards.append(episode["rewards"])
        episode_search_exchanges.append(episode["search_exchange_count"])
        episode_search_exchange_uses.append(episode["search_exchange_use_count"])
        episode_return_exchanges.append(episode["return_exchange_count"])
        episode_return_exchange_uses.append(episode["return_exchange_use_count"])

    plt.plot(episodes, episode_search_exchange_uses, color="red", label="Used Search Exchanges")
    plt.plot(episodes, episode_search_exchanges, color="blue", label="Search Exchanges")
    plt.plot(episodes, episode_return_exchanges, color="green", label="Return Exchanges")
    plt.plot(episodes, episode_return_exchange_uses, color="purple", label="Used Return Exchanges")
    plt.title("Exchanges")
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.legend()
    plt.show()

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.show()

    episode_agent_rewards = defaultdict(list)
    for agent_rewards in episode_rewards:
        for name, reward in agent_rewards.items():
            episode_agent_rewards[name].append(reward)

    for name, rewards in episode_agent_rewards.items():
        plt.plot(episodes, rewards, label=name)

    plt.title("Rewards")
    plt.xlabel(f"Episode (Step = {GRAPH_STEP})")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    env = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        square_pixel_width=SQUARE_PIXEL_WIDTH,
        carry_capacity=CARRY_CAPACITY,
    )

    try:
        state_actions, episode_data = load_data()
    except FileNotFoundError:
        state_actions, episode_data = train(env)
        if SAVE_AFTER_TRAINING:
            save_data(state_actions, episode_data)
    print(episode_data)
    plot_episode_data(episode_data)

    if SHOW_AFTER_TRAINING:
        visualize(state_actions, env)