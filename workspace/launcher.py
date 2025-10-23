import os
from workspace.classes.environment import ScavengingAntEnv
from workspace.functions.plot_functions import plot_episode_data
from workspace.functions.draw_functions import test
from workspace.functions.train_functions import train
from workspace.functions.data_functions import load_data, save_data

EPISODE_COUNT = 10000
SEED = 0
EXCHANGE_INFO = False
GRID_WIDTH = 30
GRID_HEIGHT = 20
AGENT_COUNT = 3
FOOD_COUNT = 5
OBSTACLE_COUNT = 20
NEST_COUNT = 1
AGENT_VISION_RADIUS = 0
CARRY_CAPACITY = 1


EPSILON_START = 1.00
EPSILON_DECAY_RATE = EPSILON_START / (EPISODE_COUNT / 2)
EPSILON_MIN = 0.01
LEARNING_RATE_ALPHA = 0.50
DISCOUNT_FACTOR_GAMMA = 0.90


SQUARE_PIXEL_WIDTH = 40
EPISODE_AVERAGE_STEP = 100
DRAW_ARROWS = True
SHOW_AFTER_TRAINING = True
SAVE_AFTER_TRAINING = True


SAVE_DIRECTORY = "runs"
FILE_NAME = (
    f"E={EPISODE_COUNT} "
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
)


if __name__ == "__main__":
    os.makedirs(name=SAVE_DIRECTORY, exist_ok=True)
    environment = ScavengingAntEnv(
        seed=SEED,
        grid_width=GRID_WIDTH,
        grid_height=GRID_HEIGHT,
        agent_count=AGENT_COUNT,
        food_count=FOOD_COUNT,
        obstacle_count=OBSTACLE_COUNT,
        nest_count=NEST_COUNT,
        square_pixel_width=SQUARE_PIXEL_WIDTH,
        carry_capacity=CARRY_CAPACITY,
    )

    try:
        state_actions, episode_data = load_data(SAVE_DIRECTORY, FILE_NAME)
    except FileNotFoundError:
        state_actions, episode_data = train(
            environment,
            episode_count=EPISODE_COUNT,
            exchange_info=EXCHANGE_INFO,
            epsilon_decay_rate=EPSILON_DECAY_RATE,
            epsilon_min=EPSILON_MIN,
            discount_factor_gamma=DISCOUNT_FACTOR_GAMMA,
            learning_rate_alpha=LEARNING_RATE_ALPHA,
            agent_vision_radius=AGENT_VISION_RADIUS,
        )

        if SAVE_AFTER_TRAINING:
            save_data(SAVE_DIRECTORY, FILE_NAME, state_actions, episode_data)

    plot_episode_data(
        episode_data=episode_data,
        episode_average_step=EPISODE_AVERAGE_STEP,
        agent_count=AGENT_COUNT,
    )

    if SHOW_AFTER_TRAINING:
        test(state_actions, environment)