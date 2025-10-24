import dill
from workspace.shared_types import *


def load_data(directory: str, file_name: str) -> Tuple[StateActions, List[Episode]]:
    with open(f"{directory}/{file_name}.dill", "rb") as file:
        data = dill.load(file)
        return data["state_actions"], data["episode_data"]


def save_data(
        directory: str,
        file_name: str,
        state_actions: StateActions,
        episode_data: List[Episode]
) -> None:
    with open(f"{directory}/{file_name}.dill", "wb") as file:
        dill.dump({
            "state_actions": state_actions,
            "episode_data": episode_data
        }, file)

    return None