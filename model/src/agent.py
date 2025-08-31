from collections import defaultdict


class Agent:
    """
    Represents an agent with a unique ID and a record of visited locations. This class is used in mobility models to track the movement of agents. Each agent has:
    - `id`: A unique identifier for the agent.
    - `visited_locations`: A defaultdict that counts how many times the agent has visited each location.
    """
    def __init__(self, agent_id: int) -> None:
        """
        Initializes an Agent instance. The agent starts with an empty record of visited locations. The `visited_locations` is a defaultdict with integer default values, allowing easy counting of visits to each location.
        Args:
            agent_id (int): A unique identifier for the agent.
        """
        self._id = agent_id
        self._visited_locations = defaultdict(int)

    @property
    def id(self) -> int:
        """
        The unique identifier of the agent.

        Returns:
            int: The agent's unique ID.
        """
        return self._id

    @property
    def visited_locations(self) -> defaultdict:
        """
        A dictionary-like object that counts how many times the agent has visited each location.

        Getter:
            **Returns:**
                defaultdict: A mapping from location IDs to visit counts.

        Setter:
            **Args:**
                new_visited (dict | defaultdict): A new mapping from location IDs to visit counts.

            **Raises:**
                ValueError: If `new_visited` is not a dict or defaultdict.
        """
        return self._visited_locations

    @visited_locations.setter
    def visited_locations(self, new_visited: dict | defaultdict) -> None:
        if not isinstance(new_visited, (dict, defaultdict)):
            raise ValueError("visited_locations must be a dict or defaultdict")
        self._visited_locations = defaultdict(int, new_visited)

