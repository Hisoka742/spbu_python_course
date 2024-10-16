import pytest
from project.curry_uncurry_cache.curry_uncurry_cache import cache_results


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
