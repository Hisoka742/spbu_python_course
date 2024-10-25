from typing import Optional
from project.Game.strategy import Strategy, Bet
import inspect  # Add this import


class Bot:
    """Represents a bot player in the game with a specific strategy."""

    DEFAULT_BALANCE: int = 100  # Default starting balance for bots

    def __init__(self, name: str, strategy: Strategy):
        self.name = name
        self.strategy = strategy
        self.balance = Bot.DEFAULT_BALANCE

    def make_bet(self, random_func) -> Bet:
        """Makes a bet according to the bot's strategy using a random function if required."""
        bet_method = self.strategy.bet
        if len(inspect.signature(bet_method).parameters) > 1:
            return bet_method(self.balance, random_func)
        else:
            return bet_method(self.balance)

    def win(self, amount: int) -> None:
        """Increases the bot's balance after a win."""
        self.balance += amount

    def lose(self, amount: int) -> None:
        """Decreases the bot's balance after a loss."""
        self.balance -= amount

    def is_broke(self) -> bool:
        """Checks if the bot has run out of money."""
        return self.balance <= 0
