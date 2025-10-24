from typing import List


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