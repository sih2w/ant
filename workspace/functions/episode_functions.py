from typing import List
from workspace.types import Episode


def episode_factory(agent_count: int) -> Episode:
    return [0, [0] * agent_count, 0, 0, 0, 0, 0, 0]


def has_episode_ended(
        terminations: List[bool],
        truncations: List[bool]
) -> bool:
    for termination in terminations:
        if termination:
            return True
    else:
        for truncation in truncations:
            if truncation:
                return True

    return False