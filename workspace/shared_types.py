from typing import List, Tuple, TypeAlias, TypedDict, DefaultDict, TypeVar
import pygame


T = TypeVar("T")
Used: TypeAlias = bool or None
Location: TypeAlias = Tuple[int, int]
FoodLocations: TypeAlias = Tuple[Location, ...]
Actions: TypeAlias = List[float]


class Episode(TypedDict):
    steps: int
    rewards: List[int]
    given_search_policies: int
    given_return_policies: int
    averaged_search_policies: int
    averaged_return_policies: int
    used_search_policies: int
    used_return_policies: int


class Policy(TypedDict):
    actions: Actions
    given: bool
    averaged: bool
    used: bool


GriddedPolicy: TypeAlias = List[List[Policy]]


class StateActions(TypedDict):
    returning: DefaultDict[int, List[List[Policy]]]
    searching: DefaultDict[int, DefaultDict[FoodLocations, List[List[Policy]]]]


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
