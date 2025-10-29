from workspace.shared.types import *
from workspace.shared.run_settings import *


def episode_factory() -> Episode:
    return [0, [0] * AGENT_COUNT, 0, 0, 0, 0, 0, 0]


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