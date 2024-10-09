import pytest
from curry_uncurry_cache import (
    curry_explicit,
    uncurry_explicit,
    cache_results,
    InvalidArityException,
    Evaluated,
    Isolated,
    smart_args,
)


def test_curry_basic():
    f = lambda x, y, z: f"<{x},{y},{z}>"
    f_curried = curry_explicit(f, 3)
    assert f_curried(1)(2)(3) == "<1,2,3>"


def test_uncurry_basic():
    f = lambda x, y, z: f"<{x},{y},{z}>"
    f_curried = curry_explicit(f, 3)
    f_uncurried = uncurry_explicit(f_curried, 3)
    assert f_uncurried(1, 2, 3) == "<1,2,3>"


def test_curry_invalid_arity():
    f = lambda x, y: x + y
    with pytest.raises(InvalidArityException, match="Arity cannot be negative"):
        curry_explicit(f, -1)


def test_uncurry_invalid_arity():
    f = lambda x, y: x + y
    f_curried = curry_explicit(f, 2)
    f_uncurried = uncurry_explicit(f_curried, 2)
    with pytest.raises(InvalidArityException, match="Expected 2 arguments, got 1"):
        f_uncurried(1)


def test_cache_decorator():
    @cache_results(max_size=2)
    def add(x, y):
        return x + y

    assert add(1, 2) == 3
    assert add(1, 2) == 3  # Should be from cache
    assert add(2, 3) == 5
    assert add(3, 4) == 7
    # Cache should have evicted the (1, 2) result
    assert add(1, 2) == 3  # Recalculated


def test_smart_args_isolated():
    @smart_args()
    def test_func(*, d=Isolated()):
        d["key"] = "value"
        return d

    d = {"key": "initial"}
    result = test_func(d=d)
    assert result["key"] == "value"
    assert d["key"] == "initial"  # Original dict is not mutated


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


if __name__ == "__main__":
    pytest.main()
