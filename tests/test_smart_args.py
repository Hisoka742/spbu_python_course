import pytest
from project.curry_uncurry_cache.curry_uncurry_cache import (
    Evaluated,
    Isolated,
    smart_args,
)


def test_smart_args_isolated():
    @smart_args()
    def test_func(*, d=Isolated()):
        d["key"] = "value"
        return d

    result = test_func()
    assert result["key"] == "value"  # Test the deep-copied dict


def test_smart_args_evaluated():
    counter = [0]

    def increment_counter():
        counter[0] += 1
        return counter[0]

    @smart_args()
    def test_func(*, value=Evaluated(increment_counter)):
        return value

    assert test_func() == 1
    assert test_func() == 2
    assert test_func() == 3


def test_smart_args_positional():
    @smart_args(allow_positional=True)
    def test_func(x=Isolated()):
        x.append(42)
        return x

    result = test_func([1, 2, 3])
    assert result == [1, 2, 3, 42]
