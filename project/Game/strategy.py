from dataclasses import dataclass
from typing import Optional
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

    def bet(self, balance: int, random_func) -> Bet:
        """Determines the bot's bet based on its balance using the random function provided."""
        raise NotImplementedError("Each strategy must implement the 'bet' method.")


class ConservativeStrategy(Strategy):
    """Strategy that bets conservatively on red."""

    def bet(self, balance: int, random_func) -> Bet:
        bet_amount = min(10, balance)
        return Bet(type="red", amount=bet_amount)


class AggressiveStrategy(Strategy):
    """Strategy that bets aggressively on a specific number."""

    def bet(self, balance: int, random_func) -> Bet:
        bet_amount = min(50, balance)
        return Bet(type="number", amount=bet_amount, number=random_func(range(0, 37)))


class RandomStrategy(Strategy):
    """Strategy that bets randomly on color or a number."""

    def bet(self, balance: int, random_func) -> Bet:
        bet_amount = random_func(range(1, min(21, balance + 1)))
        bet_type = random_func(["red", "black", "number"])
        number = random_func(range(0, 37)) if bet_type == "number" else None
        return Bet(type=bet_type, amount=bet_amount, number=number)
