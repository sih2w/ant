from matplotlib import pyplot as plt
from workspace.shared_types import *


def average_episode_data(
        episode_data: List[Episode],
        episode_average_step: int,
        agent_count: int
) -> List[Episode]:
    new_episode_data = []
    for current in range(0, len(episode_data), episode_average_step):
        episode_sum: Episode = {
            "steps": 0,
            "search_exchange_count": 0,
            "search_exchange_use_count": 0,
            "return_exchange_count": 0,
            "return_exchange_use_count": 0,
            "rewards": [0 for _ in range(agent_count)],
        }

        count = 0
        start = max(0, current - episode_average_step)
        end = min(current, len(episode_data))

        for previous in range(start, end):
            episode: Episode = episode_data[previous]
            episode_sum["steps"] += episode["steps"]
            episode_sum["search_exchange_count"] += episode["search_exchange_count"]
            episode_sum["search_exchange_use_count"] += episode["search_exchange_use_count"]
            episode_sum["return_exchange_count"] += episode["return_exchange_count"]
            episode_sum["return_exchange_use_count"] += episode["return_exchange_use_count"]
            count += 1

            for index, reward in enumerate(episode["rewards"]):
                episode_sum["rewards"][index] += reward

        if count > 0:
            episode_average: Episode = {
                "steps": round(episode_sum["steps"] / count),
                "search_exchange_count": round(episode_sum["search_exchange_count"] / count),
                "search_exchange_use_count": round(episode_sum["search_exchange_use_count"] / count),
                "return_exchange_count": round(episode_sum["return_exchange_count"] / count),
                "return_exchange_use_count": round(episode_sum["return_exchange_use_count"] / count),
                "rewards": [round(total_reward / count) for total_reward in episode_sum["rewards"]]
            }
            new_episode_data.append(episode_average)

    return new_episode_data


def plot_episode_data(
        episode_data: List[Episode],
        episode_average_step: int,
        agent_count: int,
) -> None:
    episode_data = average_episode_data(episode_data, episode_average_step, agent_count)
    episodes = [episode * episode_average_step for episode in range(len(episode_data))]

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
    plt.title("Exchanges per Episode")
    plt.xlabel(f"Episode (Average = {episode_average_step})")
    plt.legend()
    plt.show()

    plt.plot(episodes, episode_steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episode (Average = {episode_average_step})")
    plt.show()

    for index in range(agent_count):
        rewards = []
        for reward in episode_rewards:
            rewards.append(reward[index])
        plt.plot(episodes, rewards)

    plt.title("Rewards")
    plt.xlabel(f"Episode (Average = {episode_average_step})")
    plt.show()

    categories = [
        "Searches Given",
        "Searches Used",
        "Returns Given",
        "Returns Used"
    ]
    values = [
        sum(episode_search_exchanges),
        sum(episode_search_exchange_uses),
        sum(episode_return_exchanges),
        sum(episode_return_exchange_uses)
    ]

    plt.bar(categories, values)
    plt.title("Exchanges")
    plt.show()