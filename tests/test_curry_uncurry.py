import pytest
from curry_uncurry_cache import curry_explicit, uncurry_explicit, InvalidArityException


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
