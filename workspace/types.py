from typing import List, Tuple, TypeAlias, TypedDict, DefaultDict, TypeVar, Dict
import pygame

T = TypeVar("T")
Used: TypeAlias = bool or None
Location: TypeAlias = Tuple[int, int]
FoodLocations: TypeAlias = Tuple[Location, ...]
Actions: TypeAlias = List[float]
Episode: TypeAlias = List
Policy: TypeAlias = List
GriddedPolicy: TypeAlias = List[List[Policy]]


class StateActions(TypedDict):
    returning: List[List[List[Policy]]]
    searching: List[Dict[FoodLocations, List[List[Policy]]]]


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
