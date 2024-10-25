from strategy import Strategy, Bet


class Bot:
    """Represents a bot player in the game with a specific strategy."""

    DEFAULT_BALANCE: int = 100  # Default starting balance for bots

    def __init__(self, name: str, strategy: Strategy):
        self.name = name
        self.strategy = strategy
        self.balance = Bot.DEFAULT_BALANCE

    def make_bet(self) -> Bet:
        """Makes a bet according to the bot's strategy."""
        return self.strategy.bet(self.balance)

    def win(self, amount: int) -> None:
        """Increases the bot's balance after a win."""
        self.balance += amount

    def lose(self, amount: int) -> None:
        """Decreases the bot's balance after a loss."""
        self.balance -= amount

    def is_broke(self) -> bool:
        """Checks if the bot has run out of money."""
        return self.balance <= 0
