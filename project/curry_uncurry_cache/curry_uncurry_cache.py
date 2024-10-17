import functools
import inspect
from collections import OrderedDict
from typing import Callable, Any, Optional, Dict, Tuple
import copy


class InvalidArityException(Exception):
    """Exception raised when invalid arity is provided."""

    pass


def curry_explicit(func: Callable, arity: int) -> Callable:
    """
    Transforms a function into a curried version, i.e., a series of unary functions.

    :param func: The function to curry.
    :param arity: The number of arguments the function accepts.
    :raises InvalidArityException: If arity is negative or more arguments are provided than arity.
    :return: Curried version of the function.
    """
    if arity < 0:
        raise InvalidArityException("Arity cannot be negative")

    if arity == 0:
        return lambda: func()

    def curried(*args):
        if len(args) > arity:
            raise InvalidArityException(f"Expected {arity} arguments, got {len(args)}")

        if len(args) == arity:
            return func(*args)

        return lambda x: curried(*args, x)

    return curried


def uncurry_explicit(func: Callable, arity: int) -> Callable:
    """
    Uncurries a curried function, allowing it to accept all arguments at once.

    :param func: The curried function to uncurry.
    :param arity: The number of arguments the function expects.
    :raises InvalidArityException: If arity is negative or incorrect number of arguments are provided.
    :return: The uncurried version of the function.
    """
    if arity < 0:
        raise InvalidArityException("Arity cannot be negative")

    def uncurried(*args):
        if len(args) != arity:
            raise InvalidArityException(f"Expected {arity} arguments, got {len(args)}")

        result = func
        for arg in args:
            result = result(arg)
        return result

    return uncurried


def cache_results(max_size: int = 0) -> Callable:
    """
    Decorator to cache function results based on arguments, with a limit on cache size.

    :param max_size: The maximum number of cached results. 0 means unlimited.
    :return: A decorated function with caching enabled.
    """

    def decorator(func: Callable) -> Callable:
        cache: OrderedDict[Tuple[Any, ...], Any] = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result

            if max_size > 0 and len(cache) > max_size:
                cache.popitem(last=False)

            return result

        return wrapper

    return decorator


class Evaluated:
    """Lazy evaluation of a function when accessed."""

    def __init__(self, func: Callable[[], Any]):
        assert callable(func), "Evaluated must be initialized with a function"
        self.func = func

    def evaluate(self) -> Any:
        """Evaluate the stored function."""
        return self.func()


class Isolated:
    """Deep copy an object to prevent external mutations."""

    def __init__(self, value: Optional[Dict] = None):
        # Deep copy the input value to ensure isolation
        self.value = copy.deepcopy(value if value is not None else {})
        print(f"Initialized Isolated with value: {self.value}")

    def copy(self) -> Any:
        """Return a deep copy of the stored value."""
        copied_value = copy.deepcopy(self.value)
        print(f"Returning a deep copy: {copied_value}")
        return copied_value

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __setitem__(self, key: Any, value: Any):
        self.value[key] = value

    def __repr__(self):
        return repr(self.value)


def smart_args(allow_positional: bool = False) -> Callable:
    """
    Decorator to handle special arguments like Evaluated and Isolated.
    """

    def decorator(func: Callable) -> Callable:
        spec = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = spec.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()

            for name, value in bound_args.arguments.items():
                if isinstance(value, Evaluated):
                    print(f"Evaluating {name} using Evaluated...")
                    bound_args.arguments[name] = value.evaluate()
                elif isinstance(value, Isolated):
                    print(f"Copying {name} using Isolated...")
                    bound_args.arguments[name] = Isolated(
                        value.copy()
                    ).copy()  # Ensure deep copy

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator
