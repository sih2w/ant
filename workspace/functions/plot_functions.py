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


def plot_exchange_sums(
        given_search_policies: List[int],
        given_return_policies: List[int],
        averaged_search_policies: List[int],
        averaged_return_policies: List[int],
) -> None:
    categories = [
        "Given Search",
        "Given Return",
        "Avg Search",
        "Avg Return",
    ]
    values = [
        sum(given_search_policies),
        sum(given_return_policies),
        sum(averaged_search_policies),
        sum(averaged_return_policies),
    ]

    plt.bar(categories, values)
    plt.title("Exchanges")
    plt.show()
    return None


def plot_exchanges_per_episode(
        episodes: List[int],
        given_search_policies: List[int],
        given_return_policies: List[int],
        averaged_search_policies: List[int],
        averaged_return_policies: List[int],
        used_search_policies: List[int],
        used_return_policies: List[int],
) -> None:
    plt.plot(episodes, given_search_policies, color="red", label="Search Policies Given")
    plt.plot(episodes, given_return_policies, color="blue", label="Return Policies Given")
    plt.plot(episodes, averaged_search_policies, color="green", label="Averaged Search Policies")
    plt.plot(episodes, averaged_return_policies, color="purple", label="Averaged Return Policies")
    plt.plot(episodes, used_search_policies, color="yellow", label="Search Policies Used")
    plt.plot(episodes, used_return_policies, color="orange", label="Return Policies Used")
    plt.title("Exchanges Per Episode")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()
    return None


def plot_exchanges(
        episode_data: List[Episode],
        episodes: List[int],
) -> None:
    given_search_policies = [episode["given_search_policies"] for episode in episode_data]
    given_return_policies = [episode["given_return_policies"] for episode in episode_data]
    averaged_search_policies = [episode["averaged_search_policies"] for episode in episode_data]
    averaged_return_policies = [episode["averaged_return_policies"] for episode in episode_data]
    used_search_policies = [episode["used_search_policies"] for episode in episode_data]
    used_return_policies = [episode["used_return_policies"] for episode in episode_data]

    plot_exchanges_per_episode(
        episodes=episodes,
        given_search_policies=given_search_policies,
        given_return_policies=given_return_policies,
        averaged_search_policies=averaged_search_policies,
        averaged_return_policies=averaged_return_policies,
        used_search_policies=used_search_policies,
        used_return_policies=used_return_policies,
    )

    plot_exchange_sums(
        given_search_policies=given_search_policies,
        given_return_policies=given_return_policies,
        averaged_search_policies=averaged_search_policies,
        averaged_return_policies=averaged_return_policies,
    )

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
