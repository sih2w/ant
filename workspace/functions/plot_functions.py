from typing import Dict
from matplotlib import pyplot as plt
from workspace.shared_types import *


def average_episode_data(
        episode_data: List[Episode],
        episode_average_step: int
) -> List[Episode]:
    new_episode_data = []
    for current in range(0, len(episode_data), episode_average_step):
        episode_sum: Episode = {
            "steps": 0,
            "given_search_policies": 0,
            "given_return_policies": 0,
            "averaged_search_policies": 0,
            "averaged_return_policies": 0,
            "used_search_policies": 0,
            "used_return_policies": 0,
            "rewards": [0 for _ in range(len(episode_data[current]["rewards"]))],
        }

        count = 0
        start = max(0, current - episode_average_step)
        end = min(current, len(episode_data))

        for previous in range(start, end):
            episode: Episode = episode_data[previous]
            episode_sum["steps"] += episode["steps"]
            episode_sum["given_search_policies"] += episode["given_search_policies"]
            episode_sum["given_return_policies"] += episode["given_return_policies"]
            episode_sum["averaged_search_policies"] += episode["averaged_search_policies"]
            episode_sum["averaged_return_policies"] += episode["averaged_return_policies"]
            episode_sum["used_search_policies"] += episode["used_search_policies"]
            episode_sum["used_return_policies"] += episode["used_return_policies"]
            count += 1

            for index, reward in enumerate(episode["rewards"]):
                episode_sum["rewards"][index] += reward

        if count > 0:
            new_episode_data.append({
                "steps": round(episode_sum["steps"] / count),
                "given_search_policies": round(episode_sum["given_search_policies"] / count),
                "given_return_policies": round(episode_sum["given_return_policies"] / count),
                "averaged_search_policies": round(episode_sum["averaged_search_policies"] / count),
                "averaged_return_policies": round(episode_sum["averaged_return_policies"] / count),
                "used_search_policies": round(episode_sum["used_search_policies"] / count),
                "used_return_policies": round(episode_sum["used_return_policies"] / count),
                "rewards": [round(total_reward / count) for total_reward in episode_sum["rewards"]]
            })

    return new_episode_data


def plot_exchange_sums(exchanges: Dict[str, List[int]]) -> None:
    keys, values = [], []
    for key, value in exchanges.items():
        keys.append(key)
        values.append(sum(value))
    plt.bar(keys, values)
    plt.title("Exchanges")
    plt.show()

    return None


def plot_exchanges_per_episode(
        episodes: List[int],
        exchanges: Dict[str, List[int]]
) -> None:
    for key, value in exchanges.items():
        plt.plot(episodes, value, label=key)
    plt.title("Exchanges Per Episode")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

    return None


def plot_exchanges(
        episode_data: List[Episode],
        episodes: List[int],
) -> None:
    exchanges: Dict[str, List[int]] = {
        "GSP": [episode["given_search_policies"] for episode in episode_data],
        "GRP": [episode["given_return_policies"] for episode in episode_data],
        # "AvgSP": [episode["averaged_search_policies"] for episode in episode_data],
        # "AvgRP": [episode["averaged_return_policies"] for episode in episode_data],
        "USP": [episode["used_search_policies"] for episode in episode_data],
        "URP": [episode["used_return_policies"] for episode in episode_data]
    }

    plot_exchanges_per_episode(episodes, exchanges)
    plot_exchange_sums(exchanges)

    return None


def plot_steps_per_episode(
        episode_data: List[Episode],
        episodes: List[int],
) -> None:
    steps = [episode["steps"] for episode in episode_data]
    plt.plot(episodes, steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episodes")
    plt.show()

    return None


def plot_rewards_per_episode(
        episode_data: List[Episode],
        episodes: List[int],
) -> None:
    episode_rewards = [episode["rewards"] for episode in episode_data]
    for index in range(len(episode_rewards[0])):
        rewards = []
        for reward in episode_rewards:
            rewards.append(reward[index])
        plt.plot(episodes, rewards)

    plt.title("Rewards")
    plt.xlabel(f"Episodes")
    plt.show()

    return None

def plot_episode_data(
        episode_data: List[Episode],
        episode_average_step: int,
) -> None:
    episode_data = average_episode_data(episode_data, episode_average_step)
    episodes = [episode * episode_average_step for episode in range(len(episode_data))]

    plot_exchanges(episode_data, episodes)
    plot_steps_per_episode(episode_data, episodes)
    plot_rewards_per_episode(episode_data, episodes)

    return None
