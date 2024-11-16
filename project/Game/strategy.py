from dataclasses import dataclass
from typing import Optional, Callable
import random
from project.Game.bet_type import BetType


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

    def bet(
        self, balance: int, random_func: Optional[Callable[[int, int], int]] = None
    ) -> Bet:
        raise NotImplementedError("Each strategy must implement the 'bet' method.")

    def needs_random_func(self) -> bool:
        """Indicates if the strategy requires a random function."""
        return False


class ConservativeStrategy(Strategy):
    def bet(
        self, balance: int, random_func: Optional[Callable[[int, int], int]] = None
    ) -> Bet:
        bet_amount = min(10, balance)
        return Bet(type=BetType.RED.value, amount=bet_amount)


class AggressiveStrategy(Strategy):
    def bet(
        self, balance: int, random_func: Optional[Callable[[int, int], int]] = None
    ) -> Bet:
        bet_amount = min(50, balance)
        return Bet(
            type=BetType.NUMBER.value, amount=bet_amount, number=random.randint(0, 36)
        )


class RandomStrategy(Strategy):
    def bet(
        self,
        balance: int,
        random_func: Optional[Callable[[int, int], int]] = random.randint,
    ) -> Bet:
        if random_func is None:
            random_func = random.randint
        bet_amount = random_func(1, min(20, balance))
        bet_type = random.choice(
            [BetType.RED.value, BetType.BLACK.value, BetType.NUMBER.value]
        )
        number = random_func(0, 36) if bet_type == BetType.NUMBER.value else None
        return Bet(type=bet_type, amount=bet_amount, number=number)

    def needs_random_func(self) -> bool:
        return True


class ComplexStrategy(Strategy):
    """A more sophisticated strategy that uses history to inform betting."""

    def __init__(self):
        self.history = []

    def bet(
        self, balance: int, random_func: Optional[Callable[[int, int], int]] = None
    ) -> Bet:
        # Choose opposite color if previous bet lost
        last_result = self.history[-1] if self.history else None
        bet_type = BetType.RED if last_result != BetType.RED else BetType.BLACK
        bet_amount = min(10, balance)
        return Bet(type=bet_type.value, amount=bet_amount)

    def update_history(self, result: BetType) -> None:
        """Updates betting history based on results."""
        self.history.append(result)
