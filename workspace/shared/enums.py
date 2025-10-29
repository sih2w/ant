from enum import Enum


class PolicyAttr(Enum):
    ACTIONS = 0
    AVERAGED = 1
    GIVEN = 2
    USED = 3


class EpisodeAttr(Enum):
    STEPS = 0
    REWARDS = 1
    GIVEN_SEARCH_POLICIES = 2
    GIVEN_RETURN_POLICIES = 3
    AVERAGED_SEARCH_POLICIES = 4
    AVERAGED_RETURN_POLICIES = 5
    USED_SEARCH_POLICIES = 6
    USED_RETURN_POLICIES = 7
    EPSILON = 8
