import functools
import inspect
from collections import OrderedDict
import copy


# Exception for invalid arity values
class InvalidArityException(Exception):
    pass


# Currying function
def curry_explicit(func, arity):
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


# Uncurrying function
def uncurry_explicit(func, arity):
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


# Cache decorator
def cache_results(max_size=0):
    def decorator(func):
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)  # Update cache order for LRU
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result

            if max_size > 0 and len(cache) > max_size:
                cache.popitem(last=False)  # Remove least recently used item
            return result

        return wrapper

    return decorator


# Evaluated and Isolated classes for smart arguments
class Evaluated:
    def __init__(self, func):
        assert callable(func), "Evaluated must be initialized with a function"
        self.func = func

    def evaluate(self):
        return self.func()


class Isolated:
    def __init__(self):
        pass

    def copy(self, value):
        return copy.deepcopy(value)


# Decorator to handle smart arguments with support for positional args
def smart_args(allow_positional=False):
    def decorator(func):
        spec = inspect.getfullargspec(func)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = signature.bind_partial(*args, **kwargs)
            for name, param in signature.parameters.items():
                if name in bound_args.arguments:
                    value = bound_args.arguments[name]
                    if isinstance(value, Evaluated):
                        bound_args.arguments[name] = value.evaluate()
                    elif isinstance(value, Isolated):
                        bound_args.arguments[name] = value.copy(
                            bound_args.arguments[name]
                        )

            if allow_positional:
                # Process positional arguments for types Evaluated and Isolated
                arg_values = list(bound_args.arguments.values())
                for idx, value in enumerate(arg_values):
                    if isinstance(value, Evaluated):
                        arg_values[idx] = value.evaluate()
                    elif isinstance(value, Isolated):
                        arg_values[idx] = value.copy(arg_values[idx])

                return func(*arg_values)
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


# Example use of smart_args decorator with Isolated and Evaluated
@smart_args()
def check_isolation(*, d=Isolated()):
    d["a"] = 0
    return d


@smart_args()
def check_evaluation(*, x=10, y=Evaluated(lambda: 42)):
    print(x, y)


# Example usage:
f2 = curry_explicit(lambda x, y, z: f"<{x},{y},{z}>", 3)
g2 = uncurry_explicit(f2, 3)

if __name__ == "__main__":
    print(f2(123)(456)(562))  # Output: <123,456,562>
    print(g2(123, 456, 562))  # Output: <123,456,562>

    # Example caching:
    @cache_results(max_size=3)
    def cached_function(x, y):
        return x + y

    # Example of smart arguments
    check_isolation(d={"a": 10})
    check_evaluation()
    check_evaluation(y=100)