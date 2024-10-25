from typing import List
import random
from project.Game.wheel import Wheel
from project.Game.bot import Bot


class Roulette:
    """Represents the main game logic for Roulette."""

    def __init__(self, bots: List[Bot], rounds: int = 10, verbose: bool = False):
        """
        Initializes a new game of Roulette.

        Args:
            bots (List[Bot]): List of bot players participating in the game.
            rounds (int): Total number of rounds to play. Defaults to 10.
            verbose (bool): Flag for enabling console output. Defaults to False.
        """
        self.wheel = Wheel()  # Create a new roulette wheel
        self.bots = bots  # List of bot players
        self.rounds = rounds  # Total number of rounds
        self.current_round = 0  # Start at round 0
        self.verbose = verbose  # Verbose output flag

    def change_rules(self, new_pockets: int) -> None:
        self.wheel = Wheel(new_pockets)
        if self.verbose:
            print(f"Game rules updated: Wheel now has {new_pockets} pockets")

    def play_round(self) -> None:
        """Plays one round of the game."""
        self.current_round += 1  # Increment the round count
        result = self.wheel.spin(random.choice)  # Spin the wheel to get the result
        if self.verbose:
            print(f"Round {self.current_round}: Wheel landed on {result}")

        # Each bot makes a bet and results are processed
        for bot in self.bots:
            bet = bot.make_bet(random.choice)  # Pass random.choice as required
            if self.verbose:
                print(f"{bot.name} bets {bet.amount} on {bet.type}")

            # Check if the bot's bet was a number bet and if it won
            if bet.type == "number" and bet.number == result:
                bot.win(bet.amount * 35)  # Number bets pay 35x
                if self.verbose:
                    print(f"{bot.name} wins on number bet!")
            # Check if the bot's bet was a color bet (red or black)
            elif (bet.type == "red" and result % 2 == 0) or (
                bet.type == "black" and result % 2 == 1
            ):
                bot.win(bet.amount * 2)  # Color bets pay 2x
                if self.verbose:
                    print(f"{bot.name} wins on color bet!")
            else:
                bot.lose(bet.amount)  # Bot loses the bet
                if self.verbose:
                    print(f"{bot.name} loses the bet.")

    def show_state(self) -> str:
        """
        Returns the current state of each bot's balance.

        Returns:
            str: A formatted string showing each bot's balance.
        """
        return "\n".join(f"{bot.name} has {bot.balance} left." for bot in self.bots)

    def check_winner(self) -> bool:
        """
        Checks if only one bot remains with money.

        Returns:
            bool: True if only one bot has money left, False otherwise.
        """
        return len([bot for bot in self.bots if not bot.is_broke()]) == 1

    def run_game(self) -> None:
        """Runs the game until a winner is found or all rounds are completed."""
        while self.current_round < self.rounds and not self.check_winner():
            self.play_round()
        if self.verbose:
            print("Game over!")
