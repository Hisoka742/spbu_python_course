from abc import ABC, abstractmethod
from dataclasses import dataclass
import random


@dataclass
class Bet:
    """Represents a bet made by a bot."""

    type: str
    amount: int
    number: int = None  # Optional field


class Strategy(ABC):
    """Abstract base class for betting strategies."""

    @abstractmethod
    def bet(self, balance: int) -> Bet:
        """Determines the bot's bet based on its balance."""
        pass


class ConservativeStrategy(Strategy):
    """Strategy that bets conservatively on red."""

    def bet(self, balance: int) -> Bet:
        bet_amount = min(10, balance)
        return Bet(type="red", amount=bet_amount)


class AggressiveStrategy(Strategy):
    """Strategy that bets aggressively on a specific number."""

    def bet(self, balance: int) -> Bet:
        bet_amount = min(50, balance)
        return Bet(type="number", amount=bet_amount, number=random.randint(0, 36))


class RandomStrategy(Strategy):
    """Strategy that bets randomly on color or a number."""

    def bet(self, balance: int) -> Bet:
        bet_amount = random.randint(1, min(20, balance))
        bet_type = random.choice(["red", "black", "number"])
        number = random.randint(0, 36) if bet_type == "number" else None
        return Bet(type=bet_type, amount=bet_amount, number=number)
