from typing import List, Tuple, TypeAlias, Dict, TypedDict, DefaultDict

Used: TypeAlias = bool or None
AgentName: TypeAlias = str
Location: TypeAlias = Tuple[int, int]
FoodLocations: TypeAlias = Tuple[Location, ...]
Actions: TypeAlias = List[float]


class Episode(TypedDict):
    steps: int
    rewards: Dict[AgentName, int]


class StateActions(TypedDict):
    return_policy: DefaultDict[AgentName, DefaultDict[Location, Actions]]
    search_policy: DefaultDict[AgentName, DefaultDict[Location, DefaultDict[FoodLocations, Actions]]]


class ExchangedActions(TypedDict):
    return_policy: Dict[AgentName, Dict[Location, Used]]
    search_policy: Dict[AgentName, Dict[Location, Dict[FoodLocations, Used]]]


class Observation(TypedDict):
    agent_location: Location
    carrying_food: bool
    food_locations: FoodLocations


class Food(TypedDict):
    location: Location
    carried: bool
    hidden: bool


class Nest(TypedDict):
    location: Location


class Agent(TypedDict):
    location: Location
    carried_food: Food | None
    last_action: int


class Obstacle(TypedDict):
    location: Location
