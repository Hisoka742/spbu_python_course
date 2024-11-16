import pytest
import random
from project.Game.wheel import Wheel
from project.Game.bot import Bot
from project.Game.strategy import (
    BotStrategyMeta,
    ConservativeStrategy,
    AggressiveStrategy,
    RandomStrategy,
    ComplexStrategy,
)
from project.Game.roulette import Roulette
from project.Game.bet_type import BetType
from project.Game.strategy import Bet


def test_strategy_metaclass_enforcement():
    with pytest.raises(TypeError):

        class InvalidStrategy(metaclass=BotStrategyMeta):
            pass


def test_conservative_strategy():
    strategy = ConservativeStrategy()
    bet = strategy.bet(100)
    assert bet.type == BetType.RED.value
    assert 1 <= bet.amount <= 10


def test_aggressive_strategy():
    strategy = AggressiveStrategy()
    bet = strategy.bet(100)
    assert bet.type == BetType.NUMBER.value
    assert 1 <= bet.amount <= 50
    assert 0 <= bet.number <= 36


def test_random_strategy():
    strategy = RandomStrategy()
    bet = strategy.bet(100)
    assert bet.type in [BetType.RED.value, BetType.BLACK.value, BetType.NUMBER.value]
    assert 1 <= bet.amount <= 20
    if bet.type == BetType.NUMBER.value:
        assert 0 <= bet.number <= 36


def test_complex_strategy():
    strategy = ComplexStrategy()
    bet = strategy.bet(100)
    assert bet.type in [BetType.RED.value, BetType.BLACK.value]
    assert 1 <= bet.amount <= 10
    strategy.update_history(BetType.RED)
    bet = strategy.bet(50)
    assert bet.type == BetType.BLACK.value  # Complex strategy switches after loss


def test_game_flow():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
        Bot(name="Random", strategy=RandomStrategy()),
    ]
    game = Roulette(bots, rounds=5, verbose=False)
    game.play_round()
    assert game.current_round == 1
    assert any(bot.balance < 100 for bot in bots)


def test_check_winner():
    bots = [
        Bot(name="Conservative", strategy=ConservativeStrategy()),
        Bot(name="Aggressive", strategy=AggressiveStrategy()),
        Bot(name="Random", strategy=RandomStrategy()),
    ]
    game = Roulette(bots, rounds=5, verbose=False)
    for _ in range(5):
        game.play_round()
    assert game.check_winner() in [True, False]


def test_change_rules_via_metaclass():
    class CustomRoulette(Roulette):
        DEFAULT_POCKETS = 42

    game = CustomRoulette(bots=[], rounds=5, verbose=False)
    assert game.wheel.numbers == list(range(42))
