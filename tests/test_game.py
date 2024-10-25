import pytest
import random  # Import random for testing
from project.Game.wheel import Wheel
from project.Game.bot import Bot
from project.Game.strategy import (
    BotStrategyMeta,
    ConservativeStrategy,
    AggressiveStrategy,
    RandomStrategy,
)
from project.Game.roulette import Roulette


def test_strategy_metaclass_enforcement():
    with pytest.raises(TypeError):

        class InvalidStrategy(metaclass=BotStrategyMeta):
            pass


def test_conservative_strategy():
    strategy = ConservativeStrategy()
    bet = strategy.bet(100)
    assert bet.type == "red"
    assert 1 <= bet.amount <= 10


def test_aggressive_strategy():
    strategy = AggressiveStrategy()
    bet = strategy.bet(100)
    assert bet.type == "number"
    assert 1 <= bet.amount <= 50
    assert 0 <= bet.number <= 36


def test_random_strategy():
    strategy = RandomStrategy()
    bet = strategy.bet(100)
    assert bet.type in ["red", "black", "number"]
    assert 1 <= bet.amount <= 20
    if bet.type == "number":
        assert 0 <= bet.number <= 36


def test_game_flow_with_metaclasses():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
        Bot(name="Random", strategy=RandomStrategy()),
    ]
    game = Roulette(bots, rounds=5)
    game.play_round()
    assert game.current_round == 1
    assert any(bot.balance < 100 for bot in bots)


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
