from gymnasium.wrappers import FlattenObservation
import scavenging_ant.envs.scavenging_ant as scavenging_ant

if __name__ == "__main__":
    env = scavenging_ant.ScavengingAntEnv(
        render_mode="human",
        render_fps=1,
        persistent_obstacles=True,
        persistent_food=True,
        persistent_nests=True,
        seed=100,
        grid_height=4,
        grid_width=5,
        food_count=10,
        percent_obstacles=0
    )
    env = FlattenObservation(env)
    env.reset()
    terminated = False

    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print("---")
        print(observation)