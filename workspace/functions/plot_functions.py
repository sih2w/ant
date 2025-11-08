from matplotlib import pyplot as plt
from workspace.shared.types import *
from workspace.shared.enums import *
from workspace.functions.episode_functions import episode_factory
from workspace.shared.run_settings import *


def average_episode_data(episodes: List[Episode]) -> List[Episode]:
    new_episodes = []
    for current in range(0, len(episodes), EPISODE_AVERAGE_STEP):
        count = 0
        start = max(0, current - EPISODE_AVERAGE_STEP)
        end = min(current, len(episodes))
        episode_sum: Episode = episode_factory()

        for previous in range(start, end):
            episode: Episode = episodes[previous]
            episode_sum[EpisodeAttr.STEPS.value] += episode[EpisodeAttr.STEPS.value]
            episode_sum[EpisodeAttr.EXCHANGES.value] += episode[EpisodeAttr.EXCHANGES.value]
            episode_sum[EpisodeAttr.EPSILON.value] += episode[EpisodeAttr.EPSILON.value]
            for index, reward in enumerate(episode[EpisodeAttr.REWARDS.value]):
                episode_sum[EpisodeAttr.REWARDS.value][index] += reward
            count += 1

        if count > 0:
            episode_average = episode_factory()
            episode_average[EpisodeAttr.STEPS.value] = round(episode_sum[EpisodeAttr.STEPS.value] / count)
            episode_average[EpisodeAttr.EXCHANGES.value] = round(episode_sum[EpisodeAttr.EXCHANGES.value] / count)
            episode_average[EpisodeAttr.EPSILON.value] = episode_sum[EpisodeAttr.EPSILON.value] / count
            for index, reward in enumerate(episode_sum[EpisodeAttr.REWARDS.value]):
                episode_average[EpisodeAttr.REWARDS.value][index] = round(reward / count)
            new_episodes.append(episode_average)

    return new_episodes


def plot_exchanges_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int],
) -> None:
    exchanges = [episode[EpisodeAttr.EXCHANGES.value] for episode in episodes]
    plt.plot(episode_numbers, exchanges, color="green", label="Steps")
    plt.axvline(x=WORKER_EPISODE_COUNT, color="tan", label="Merge")
    plt.title("Exchanges Per Episode")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

    return None


def plot_steps_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int]
) -> None:
    steps = [episode[EpisodeAttr.STEPS.value] for episode in episodes]
    plt.plot(episode_numbers, steps, color="green", label="Steps")
    plt.axvline(x=WORKER_EPISODE_COUNT, color="tan", label="Merge")
    plt.title("Steps")
    plt.xlabel(f"Episodes")
    plt.show()

    return None


def plot_epsilons(
        episodes: List[Episode],
        episode_numbers: List[int]
) -> None:
    epsilons = [episode[EpisodeAttr.EPSILON.value] for episode in episodes]
    plt.plot(episode_numbers, epsilons, color="green", label="Epsilons")
    plt.axvline(x=WORKER_EPISODE_COUNT, color="tan", label="Merge")
    plt.title("Epsilons")
    plt.xlabel(f"Episodes")
    plt.show()

    return None


def plot_rewards_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int]
) -> None:
    episode_rewards = [episode[EpisodeAttr.REWARDS.value] for episode in episodes]
    for index in range(AGENT_COUNT):
        rewards = []
        for reward in episode_rewards:
            rewards.append(reward[index])
        plt.plot(episode_numbers, rewards)

    plt.axvline(x=WORKER_EPISODE_COUNT, color="tan", label="Merge")
    plt.title("Rewards")
    plt.xlabel(f"Episodes")
    plt.show()

    return None


def plot_episodes(episodes: List[Episode]) -> None:
    episodes = average_episode_data(episodes)
    episode_numbers = [episode * EPISODE_AVERAGE_STEP for episode in range(len(episodes))]

    plot_exchanges_per_episode(episodes, episode_numbers)
    plot_steps_per_episode(episodes, episode_numbers)
    plot_rewards_per_episode(episodes, episode_numbers)
    plot_epsilons(episodes, episode_numbers)

    return None
