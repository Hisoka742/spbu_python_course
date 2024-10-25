import random
from project.Game.wheel import Wheel
from project.Game.bot import Bot
from typing import List


class Roulette:
    """Represents the main game logic for Roulette."""

    def __init__(self, bots: List[Bot], rounds: int = 10, verbose: bool = False):
        self.wheel = Wheel()
        self.bots = bots
        self.rounds = rounds
        self.current_round = 0
        self.verbose = verbose

    def change_rules(self, new_pockets: int) -> None:
        self.wheel = Wheel(new_pockets)
        if self.verbose:
            print(f"Game rules updated: Wheel now has {new_pockets} pockets")

    def play_round(self) -> None:
        self.current_round += 1
        result = self.wheel.spin()
        if self.verbose:
            print(f"Round {self.current_round}: Wheel landed on {result}")

        for bot in self.bots:
            bet = bot.make_bet(random_func=random.random)  # Pass random function here
            if self.verbose:
                print(f"{bot.name} bets {bet.amount} on {bet.type}")

        if bet.type == "number" and bet.number == result:
            bot.win(bet.amount * 35)
            if self.verbose:
                print(f"{bot.name} wins on number bet!")
        elif (bet.type == "red" and result % 2 == 0) or (
            bet.type == "black" and result % 2 == 1
        ):
            bot.win(bet.amount * 2)
            if self.verbose:
                print(f"{bot.name} wins on color bet!")
        else:
            bot.lose(bet.amount)
            if self.verbose:
                print(f"{bot.name} loses the bet.")

    def show_state(self) -> str:
        return "\n".join(f"{bot.name} has {bot.balance} left." for bot in self.bots)

    def check_winner(self) -> bool:
        return len([bot for bot in self.bots if not bot.is_broke()]) == 1

    def run_game(self) -> None:
        while self.current_round < self.rounds and not self.check_winner():
            self.play_round()
        if self.verbose:
            print("Game over!")
