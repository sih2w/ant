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
            episode_sum[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] += episode[EpisodeAttr.GIVEN_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttr.GIVEN_RETURN_POLICIES.value] += episode[EpisodeAttr.GIVEN_RETURN_POLICIES.value]
            episode_sum[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value] += episode[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttr.AVERAGED_RETURN_POLICIES.value] += episode[EpisodeAttr.AVERAGED_RETURN_POLICIES.value]
            episode_sum[EpisodeAttr.USED_SEARCH_POLICIES.value] += episode[EpisodeAttr.USED_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttr.USED_RETURN_POLICIES.value] += episode[EpisodeAttr.USED_RETURN_POLICIES.value]
            for index, reward in enumerate(episode[EpisodeAttr.REWARDS.value]):
                episode_sum[EpisodeAttr.REWARDS.value][index] += reward
            count += 1

        if count > 0:
            episode_average = episode_factory()
            episode_average[EpisodeAttr.STEPS.value] = round(episode_sum[EpisodeAttr.STEPS.value] / count)
            episode_average[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttr.GIVEN_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttr.GIVEN_RETURN_POLICIES.value] / count)
            episode_average[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttr.AVERAGED_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttr.AVERAGED_RETURN_POLICIES.value] / count)
            episode_average[EpisodeAttr.USED_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttr.USED_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttr.USED_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttr.USED_RETURN_POLICIES.value] / count)
            for index, reward in enumerate(episode_sum[EpisodeAttr.REWARDS.value]):
                episode_average[EpisodeAttr.REWARDS.value][index] = round(reward / count)
            new_episodes.append(episode_average)

    return new_episodes


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
        episode_numbers: List[int],
        exchanges: Dict[str, List[int]]
) -> None:
    for key, value in exchanges.items():
        plt.plot(episode_numbers, value, label=key)
    plt.title("Exchanges Per Episode")
    plt.xlabel("Episodes")
    plt.legend()
    plt.show()

    return None


def plot_exchanges(
        episodes: List[Episode],
        episode_numbers: List[int],
) -> None:
    exchanges: Dict[str, List[int]] = {
        "GSP": [episode[EpisodeAttr.GIVEN_SEARCH_POLICIES.value] for episode in episodes],
        "GRP": [episode[EpisodeAttr.GIVEN_RETURN_POLICIES.value] for episode in episodes],
        "AvgSP": [episode[EpisodeAttr.AVERAGED_SEARCH_POLICIES.value] for episode in episodes],
        "AvgRP": [episode[EpisodeAttr.AVERAGED_RETURN_POLICIES.value] for episode in episodes],
        "USP": [episode[EpisodeAttr.USED_SEARCH_POLICIES.value] for episode in episodes],
        "URP": [episode[EpisodeAttr.USED_RETURN_POLICIES.value] for episode in episodes]
    }

    plot_exchanges_per_episode(episode_numbers, exchanges)
    plot_exchange_sums(exchanges)

    return None


def plot_steps_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int],
) -> None:
    steps = [episode[EpisodeAttr.STEPS.value] for episode in episodes]
    plt.plot(episode_numbers, steps, color="green")
    plt.title("Steps")
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

    plt.title("Rewards")
    plt.xlabel(f"Episodes")
    plt.show()

    return None

def plot_episodes(episodes: List[Episode]) -> None:
    episodes = average_episode_data(episodes)
    episode_numbers = [episode * EPISODE_AVERAGE_STEP for episode in range(len(episodes))]

    plot_exchanges(episodes, episode_numbers)
    plot_steps_per_episode(episodes, episode_numbers)
    plot_rewards_per_episode(episodes, episode_numbers)

    return None
