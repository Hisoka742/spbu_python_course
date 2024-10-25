import pytest
from project.Game.wheel import Wheel
from project.Game.bot import Bot
from project.Game.strategy import (
    ConservativeStrategy,
    AggressiveStrategy,
    RandomStrategy,
)
from project.Game.roulette import Roulette

# Test Wheel functionality
def test_wheel_spin():
    wheel = Wheel()
    number = wheel.spin()
    assert number in wheel.numbers


# Test Metaclass enforcement for Strategies
def test_strategy_metaclass_enforcement():
    # Create a class without the `bet` method, which should raise an error
    with pytest.raises(TypeError):

        class InvalidStrategy(metaclass=BotStrategyMeta):
            pass


def test_conservative_strategy():
    strategy = ConservativeStrategy()
    bet = strategy.bet(100, random.choice)  # Pass random.choice as required
    assert bet.type == "red"  # Use dot notation for attributes
    assert 1 <= bet.amount <= 10


def test_aggressive_strategy():
    strategy = AggressiveStrategy()
    bet = strategy.bet(100, random.choice)
    assert bet.type == "number"
    assert 1 <= bet.amount <= 50
    assert 0 <= bet.number <= 36


def test_random_strategy():
    strategy = RandomStrategy()
    bet = strategy.bet(100, random.choice)
    assert bet.type in ["red", "black", "number"]
    assert 1 <= bet.amount <= 20
    if bet.type == "number":
        assert 0 <= bet.number <= 36


# Test Game Rule customization
def test_change_game_rules():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
    ]
    game = Roulette(bots)
    # Change game rules to have 50 pockets
    game.change_rules(50)
    assert len(game.wheel.numbers) == 50


# Test game flow with metaclass strategies
def test_game_flow_with_metaclasses():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
        Bot(name="Random", strategy=RandomStrategy()),
    ]
    game = Roulette(bots, rounds=5)
    game.play_round()
    assert game.current_round == 1
    assert any(
        bot.balance < 100 for bot in bots
    )  # At least one bot should have lost money


def test_check_winner():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
        Bot(name="Random", strategy=RandomStrategy()),
    ]
    game = Roulette(bots, rounds=5)
    for _ in range(5):
        game.play_round()
    assert game.check_winner() in [True, False]
