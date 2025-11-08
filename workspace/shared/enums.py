from enum import Enum


class PolicyAttr(Enum):
    ACTIONS = 0


class EpisodeAttr(Enum):
    STEPS = 0
    REWARDS = 1
    EXCHANGES = 2
    EPSILON = 3
