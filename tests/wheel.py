import random
from typing import List


class Wheel:
    """Represents a roulette wheel for the game."""

    def __init__(self, pockets: int = 37):
        """Initializes the wheel with a set number of pockets."""
        self.numbers: List[int] = (
            list(range(pockets)) + [0] if pockets == 37 else list(range(pockets))
        )
        self.current_number: int = None

    def spin(self) -> int:
        """Spins the wheel and returns a random result."""
        self.current_number = random.choice(self.numbers)
        return self.current_number
