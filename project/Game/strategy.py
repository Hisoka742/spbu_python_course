from dataclasses import dataclass
from typing import Optional, Callable
import random


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

    def bet(self, balance: int, random_func: Optional[Callable] = None) -> Bet:
        raise NotImplementedError("Each strategy must implement the 'bet' method.")

    def needs_random_func(self) -> bool:
        """Indicates if the strategy requires a random function."""
        return False


class ConservativeStrategy(Strategy):
    def bet(self, balance: int, random_func: Optional[Callable] = None) -> Bet:
        bet_amount = min(10, balance)
        return Bet(type="red", amount=bet_amount)


class AggressiveStrategy(Strategy):
    def bet(self, balance: int, random_func: Optional[Callable] = None) -> Bet:
        bet_amount = min(50, balance)
        return Bet(type="number", amount=bet_amount, number=random.randint(0, 36))


class RandomStrategy(Strategy):
    def bet(self, balance: int, random_func: Optional[Callable] = None) -> Bet:
        if random_func is None:
            random_func = random  # Default to the `random` module if no function provided
        bet_amount = random_func.randint(1, min(20, balance))
        bet_type = random_func.choice(["red", "black", "number"])
        number = random_func.randint(0, 36) if bet_type == "number" else None
        return Bet(type=bet_type, amount=bet_amount, number=number)

    def needs_random_func(self) -> bool:
        return True
