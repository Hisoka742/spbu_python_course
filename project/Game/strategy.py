# strategy.py
from dataclasses import dataclass
from typing import Optional
from abc import abstractmethod

@dataclass
class Bet:
    """Represents a bet made by a bot."""
    type: str
    amount: int
    number: Optional[int] = None

class BotStrategyMeta(type):
    """Metaclass that enforces the implementation of the 'bet' method in each strategy."""
    def __new__(cls, name, bases, dct):
        if "bet" not in dct:
            raise TypeError(f"Every strategy class must implement 'bet' method.")
        return super().__new__(cls, name, bases, dct)

class Strategy(metaclass=BotStrategyMeta):
    """Abstract base class for betting strategies."""

    @abstractmethod
    def bet(self, balance: int, random_func) -> Bet:
        """Determines the bot's bet based on its balance using the random function provided."""
        pass
