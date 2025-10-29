import dill
import os
from workspace.shared.types import *
from workspace.shared.run_settings import *


SAVE_DIRECTORY = "runs"
FILE_NAME = (
    f"WE={WORKER_EPISODE_COUNT} "
    f"ME={MERGED_EPISODE_COUNT} "
    f"S={SEED} "
    f"GW={GRID_WIDTH} "
    f"GH={GRID_HEIGHT} "
    f"AC={AGENT_COUNT} "
    f"FC={FOOD_COUNT} "
    f"NC={NEST_COUNT} "
    f"OC={OBSTACLE_COUNT} "
    f"AVR={AGENT_VISION_RADIUS} "
    f"EI={EXCHANGE_INFO} "
    f"CC={CARRY_CAPACITY} "
    f"WC={WORKER_COUNT} "
)


def load_data() -> Tuple[StateActions, List[Episode]]:
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "rb") as file:
        data = dill.load(file)
        return data["state_actions"], data["episodes"]


def save_data(
        state_actions: StateActions,
        episodes: List[Episode]
) -> None:
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)

    with open(f"{SAVE_DIRECTORY}/{FILE_NAME}.dill", "wb") as file:
        dill.dump({
            "state_actions": state_actions,
            "episodes": episodes
        }, file)

    return None