import scavenging_ant.envs.scavenging_ant as scavenging_ant

if __name__ == "__main__":
    env = scavenging_ant.ScavengingAntEnv(
        render_mode="human",
        render_fps=10,
        grid_height=4,
        grid_width=5,
        food_count=10,
        seed=0,
        agent_count=2
    )

    env.reset(seed=0)
    terminated = False

    while not terminated:
        actions = {}
        for name in env.agents:
            actions[name] = env.action_space(name).sample()

        observations, rewards, terminations, truncations, infos = env.step(actions)
        observations = env.flatten_observations(observations)

        for _, termination in terminations.items():
            if termination:
                terminated = True
                break

        print(observations)