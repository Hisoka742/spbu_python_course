import random
from Project.Game.game import (
    Wheel,
    Bot,
    ConservativeStrategy,
    AggressiveStrategy,
    RandomStrategy,
    Roulette,
    BotStrategyMeta,
    GameRulesMeta,
)

# Metaclass to enforce that any bot strategy class must implement the 'bet' method
class BotStrategyMeta(type):
    def __new__(cls, name, bases, dct):
        if "bet" not in dct:
            raise TypeError("Every strategy class must implement 'bet' method")
        return super().__new__(cls, name, bases, dct)


# Metaclass to enforce that the game class must implement the 'change_rules' method
class GameRulesMeta(type):
    def __new__(cls, name, bases, dct):
        if "change_rules" not in dct:
            raise TypeError("Game class must implement 'change_rules' method")
        return super().__new__(cls, name, bases, dct)


# Class representing the roulette wheel
class Wheel:
    def __init__(self):
        # Numbers on the roulette wheel (0-36, plus 00 represented as 0 again)
        self.numbers = list(range(0, 37)) + [0]
        self.current_number = None

    # Spin the wheel to get a random result
    def spin(self):
        self.current_number = random.choice(self.numbers)
        return self.current_number


# Class representing a bot player
class Bot:
    def __init__(self, name, strategy):
        self.name = name  # Bot's name
        self.strategy = strategy  # Bot's strategy (e.g., Conservative, Aggressive)
        self.balance = 100  # Starting balance for each bot

    # Make a bet according to the bot's strategy
    def make_bet(self):
        return self.strategy.bet(self.balance)

    # Increase the bot's balance after a win
    def win(self, amount):
        self.balance += amount

    # Decrease the bot's balance after a loss
    def lose(self, amount):
        self.balance -= amount

    # Check if the bot has run out of money
    def is_broke(self):
        return self.balance <= 0


# Conservative betting strategy: always bet a small amount on red
class ConservativeStrategy(metaclass=BotStrategyMeta):
    def bet(self, balance):
        bet_amount = min(10, balance)  # Bet 10 or less if balance is lower
        return {"type": "red", "amount": bet_amount}


# Aggressive betting strategy: bet a large amount on a random number
class AggressiveStrategy(metaclass=BotStrategyMeta):
    def bet(self, balance):
        bet_amount = min(50, balance)  # Bet 50 or less if balance is lower
        return {"type": "number", "amount": bet_amount, "number": random.randint(0, 36)}


# Random betting strategy: randomly choose a bet type (red, black, or number)
class RandomStrategy(metaclass=BotStrategyMeta):
    def bet(self, balance):
        bet_amount = random.randint(
            1, min(20, balance)
        )  # Bet a random amount between 1 and 20
        bet_type = random.choice(["red", "black", "number"])  # Random bet type
        bet = {"type": bet_type, "amount": bet_amount}
        if bet_type == "number":
            bet["number"] = random.randint(
                0, 36
            )  # If betting on a number, choose a random one
        return bet


# Class representing the Roulette game
class Roulette(metaclass=GameRulesMeta):
    def __init__(self, bots, rounds=10):
        self.wheel = Wheel()  # Create a new roulette wheel
        self.bots = bots  # List of bot players
        self.rounds = rounds  # Total number of rounds
        self.current_round = 0  # Start at round 0

    # Method to change game rules, e.g., number of pockets on the wheel
    def change_rules(self, new_pockets):
        self.wheel.numbers = list(range(new_pockets))  # Update the numbers on the wheel
        print(f"Game rules updated: Wheel now has {new_pockets} pockets")

    # Play one round of the game
    def play_round(self):
        self.current_round += 1  # Increment the round count
        result = self.wheel.spin()  # Spin the wheel to get the result
        print(f"Round {self.current_round}: Wheel landed on {result}")

        # Each bot makes a bet and results are processed
        for bot in self.bots:
            bet = bot.make_bet()  # Get the bot's bet
            print(f"{bot.name} bets {bet['amount']} on {bet['type']}")

            # Check if the bot's bet was a number bet and if it won
            if bet["type"] == "number" and bet.get("number") == result:
                bot.win(bet["amount"] * 35)  # Number bets pay 35x
                print(f"{bot.name} wins on number bet!")
            # Check if the bot's bet was a color bet (red or black)
            elif (bet["type"] == "red" and result % 2 == 0) or (
                bet["type"] == "black" and result % 2 == 1
            ):
                bot.win(bet["amount"] * 2)  # Color bets pay 2x
                print(f"{bot.name} wins on color bet!")
            else:
                bot.lose(bet["amount"])  # Bot loses the bet
                print(f"{bot.name} loses the bet.")

        self.show_state()  # Show the state of the game after each round

    # Show the current state of each bot's balance
    def show_state(self):
        for bot in self.bots:
            print(f"{bot.name} has {bot.balance} left.")

    # Check if there is only one bot left with money (the winner)
    def check_winner(self):
        active_bots = [
            bot for bot in self.bots if not bot.is_broke()
        ]  # Bots still in the game
        return len(active_bots) == 1  # True if only one bot left, otherwise False

    # Run the game until all rounds are played or a winner is determined
    def run_game(self):
        while self.current_round < self.rounds and not self.check_winner():
            self.play_round()  # Play each round
        print("Game over!")  # End of the game
