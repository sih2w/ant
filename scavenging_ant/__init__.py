from gymnasium.envs.registration import register

register(
    id="scavenging_ant/ScavengingAnt-v0",
    entry_point="scavenging_ant.envs:ScavengingAntEnv",
)
