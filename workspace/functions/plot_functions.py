from matplotlib import pyplot as plt
from workspace.types import *
from workspace.functions.episode_functions import episode_factory
from workspace.enums import EpisodeAttribute

def average_episode_data(
        episodes: List[Episode],
        episode_average_step: int,
        agent_count: int
) -> List[Episode]:
    new_episodes = []
    for current in range(0, len(episodes), episode_average_step):
        count = 0
        start = max(0, current - episode_average_step)
        end = min(current, len(episodes))
        episode_sum: Episode = episode_factory(agent_count)

        for previous in range(start, end):
            episode: Episode = episodes[previous]
            episode_sum[EpisodeAttribute.STEPS.value] += episode[EpisodeAttribute.STEPS.value]
            episode_sum[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] += episode[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] += episode[EpisodeAttribute.GIVEN_RETURN_POLICIES.value]
            episode_sum[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value] += episode[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value] += episode[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value]
            episode_sum[EpisodeAttribute.USED_SEARCH_POLICIES.value] += episode[EpisodeAttribute.USED_SEARCH_POLICIES.value]
            episode_sum[EpisodeAttribute.USED_RETURN_POLICIES.value] += episode[EpisodeAttribute.USED_RETURN_POLICIES.value]
            for index, reward in enumerate(episode[EpisodeAttribute.REWARDS.value]):
                episode_sum[EpisodeAttribute.REWARDS.value][index] += reward
            count += 1

        if count > 0:
            episode_average = episode_factory(agent_count)
            episode_average[EpisodeAttribute.STEPS.value] = round(episode_sum[EpisodeAttribute.STEPS.value] / count)
            episode_average[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] / count)
            episode_average[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value] / count)
            episode_average[EpisodeAttribute.USED_SEARCH_POLICIES.value] = round(episode_sum[EpisodeAttribute.USED_SEARCH_POLICIES.value] / count)
            episode_average[EpisodeAttribute.USED_RETURN_POLICIES.value] = round(episode_sum[EpisodeAttribute.USED_RETURN_POLICIES.value] / count)
            for index, reward in enumerate(episode_sum[EpisodeAttribute.REWARDS.value]):
                episode_average[EpisodeAttribute.REWARDS.value][index] = round(reward / count)
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
        "GSP": [episode[EpisodeAttribute.GIVEN_SEARCH_POLICIES.value] for episode in episodes],
        "GRP": [episode[EpisodeAttribute.GIVEN_RETURN_POLICIES.value] for episode in episodes],
        "AvgSP": [episode[EpisodeAttribute.AVERAGED_SEARCH_POLICIES.value] for episode in episodes],
        "AvgRP": [episode[EpisodeAttribute.AVERAGED_RETURN_POLICIES.value] for episode in episodes],
        "USP": [episode[EpisodeAttribute.USED_SEARCH_POLICIES.value] for episode in episodes],
        "URP": [episode[EpisodeAttribute.USED_RETURN_POLICIES.value] for episode in episodes]
    }

    plot_exchanges_per_episode(episode_numbers, exchanges)
    plot_exchange_sums(exchanges)

    return None


def plot_steps_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int],
) -> None:
    steps = [episode[EpisodeAttribute.STEPS.value] for episode in episodes]
    plt.plot(episode_numbers, steps, color="green")
    plt.title("Steps")
    plt.xlabel(f"Episodes")
    plt.show()

    return None


def plot_rewards_per_episode(
        episodes: List[Episode],
        episode_numbers: List[int],
        agent_count: int,
) -> None:
    episode_rewards = [episode[EpisodeAttribute.REWARDS.value] for episode in episodes]
    for index in range(agent_count):
        rewards = []
        for reward in episode_rewards:
            rewards.append(reward[index])
        plt.plot(episode_numbers, rewards)

    plt.title("Rewards")
    plt.xlabel(f"Episodes")
    plt.show()

    return None

def plot_episodes(
        episodes: List[Episode],
        episode_average_step: int,
        agent_count: int,
) -> None:
    episodes = average_episode_data(episodes, episode_average_step, agent_count)
    episode_numbers = [episode * episode_average_step for episode in range(len(episodes))]

    plot_exchanges(episodes, episode_numbers)
    plot_steps_per_episode(episodes, episode_numbers)
    plot_rewards_per_episode(episodes, episode_numbers, agent_count)

    return None
