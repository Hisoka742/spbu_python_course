# bot.py
import random
from project.Game.strategy import Strategy, Bet
from typing import Callable


class Bot:
    """Represents a bot player in the game with a specific strategy."""

    DEFAULT_BALANCE: int = 100  # Default starting balance for bots

    def __init__(self, name: str, strategy: Strategy):
        self.name = name
        self.strategy = strategy
        self.balance = Bot.DEFAULT_BALANCE

    def make_bet(self, random_func: Callable[[int, int], int] = random.randint) -> Bet:
        """
        Makes a bet according to the bot's strategy.

        Args:
            random_func (Callable[[int, int], int]): A random function to generate
                random numbers if needed by the strategy.

        Returns:
            Bet: The generated bet based on the strategy.
        """
        if self.strategy.needs_random_func():
            return self.strategy.bet(self.balance, random_func)
        else:
            return self.strategy.bet(self.balance)

    def win(self, amount: int) -> None:
        """
        Increases the bot's balance after a win.

        Args:
            amount (int): The amount to add to the bot's balance.
        """
        self.balance += amount

    def lose(self, amount: int) -> None:
        """
        Decreases the bot's balance after a loss.

        Args:
            amount (int): The amount to subtract from the bot's balance.
        """
        self.balance -= amount

    def is_broke(self) -> bool:
        """
        Checks if the bot has run out of money.

        Returns:
            bool: True if the bot's balance is 0 or less, False otherwise.
        """
        return self.balance <= 0
