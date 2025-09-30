from typing import List, Tuple, TypeAlias, Dict, TypedDict, DefaultDict
import pygame


Used: TypeAlias = bool or None
AgentName: TypeAlias = str
Location: TypeAlias = Tuple[int, int]
FoodLocations: TypeAlias = Tuple[Location, ...]
Actions: TypeAlias = List[float]


class Episode(TypedDict):
    steps: int
    rewards: Dict[AgentName, int]
    search_exchange_count: int
    search_exchange_use_count: int
    return_exchange_count: int
    return_exchange_use_count: int


class Policy(TypedDict):
    actions: Actions


class StateActions(TypedDict):
    returning: DefaultDict[AgentName, DefaultDict[Location, Policy]]
    searching: DefaultDict[AgentName, DefaultDict[Location, DefaultDict[FoodLocations, Policy]]]


class State(TypedDict):
    agent_location: Location
    carrying_food: bool
    food_locations: FoodLocations


class Food(TypedDict):
    location: Location
    carried: bool
    deposited: bool
    spawn_location: Location


class Nest(TypedDict):
    location: Location
    spawn_location: Location


class Agent(TypedDict):
    location: Location
    carried_food: List[Food]
    last_action: int
    spawn_location: Location
    carry_capacity: int
    color: pygame.Color


class Obstacle(TypedDict):
    location: Location
    spawn_location: Location
