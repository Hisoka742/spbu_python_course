import functools  # For higher-order functions like wraps
import inspect  # For introspecting function signatures
from collections import OrderedDict  # For maintaining an ordered cache (LRU)
import copy  # For deep copying in the Isolated class


# Exception for invalid arity values
class InvalidArityException(Exception):
    pass


# Currying function: transforms a function into a series of unary functions
def curry_explicit(func, arity):
    # Ensure the arity is valid (non-negative)
    if arity < 0:
        raise InvalidArityException("Arity cannot be negative")

    # If arity is 0, just return the result of the function
    if arity == 0:
        return lambda: func()

    # Define a curried function
    def curried(*args):
        # Ensure no more than the expected number of arguments are passed
        if len(args) > arity:
            raise InvalidArityException(f"Expected {arity} arguments, got {len(args)}")

        # If all arguments are provided, call the original function
        if len(args) == arity:
            return func(*args)

        # Otherwise, return a function that accepts the next argument
        return lambda x: curried(*args, x)

    return curried


# Uncurrying function: converts a curried function back to a function that takes all arguments at once
def uncurry_explicit(func, arity):
    # Ensure the arity is valid (non-negative)
    if arity < 0:
        raise InvalidArityException("Arity cannot be negative")

    # Define an uncurried function
    def uncurried(*args):
        # Ensure the correct number of arguments is passed
        if len(args) != arity:
            raise InvalidArityException(f"Expected {arity} arguments, got {len(args)}")

        # Apply the arguments one by one to the curried function
        result = func
        for arg in args:
            result = result(arg)
        return result

    return uncurried


# Cache decorator: caches function results based on arguments
def cache_results(max_size=0):
    def decorator(func):
        # OrderedDict to store cache entries with LRU (least recently used) eviction
        cache = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the arguments
            key = (args, tuple(sorted(kwargs.items())))
            # If the result is already cached, move it to the end (mark it as recently used)
            if key in cache:
                cache.move_to_end(key)
                return cache[key]

            # Call the original function and cache the result
            result = func(*args, **kwargs)
            cache[key] = result

            # If cache exceeds the max size, evict the least recently used item
            if max_size > 0 and len(cache) > max_size:
                cache.popitem(last=False)  # Remove the oldest item

            return result

        return wrapper

    return decorator


# Evaluated class: for lazy evaluation of function results (evaluates only when accessed)
class Evaluated:
    def __init__(self, func):
        # Ensure that the provided argument is a callable function
        assert callable(func), "Evaluated must be initialized with a function"
        self.func = func

    # Evaluate the function when needed
    def evaluate(self):
        return self.func()


# Isolated class: deep copies the object to prevent modifications to the original
class Isolated:
    def __init__(self, value=None):
        # Initialize with a deep copy of the value (default is an empty dictionary)
        self.value = copy.deepcopy(value if value is not None else {})

    # Return a deep copy of the internal dictionary or passed value
    def copy(self, value=None):
        return copy.deepcopy(self.value if value is None else value)

    # Allow dictionary-style access
    def __getitem__(self, key):
        return self.value[key]

    # Allow dictionary-style assignment
    def __setitem__(self, key, value):
        self.value[key] = value


# Decorator to handle smart arguments (Isolated, Evaluated) and positional args
def smart_args(allow_positional=False):
    def decorator(func):
        # Get the full argument specification for the function
        spec = inspect.getfullargspec(func)
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Bind the passed arguments to the function's signature
            bound_args = signature.bind_partial(*args, **kwargs)
            bound_args.apply_defaults()  # Fill in default arguments

            # Process the bound arguments
            for name, value in bound_args.arguments.items():
                # If the value is an instance of Evaluated, evaluate it
                if isinstance(value, Evaluated):
                    bound_args.arguments[name] = value.evaluate()
                # If the value is an instance of Isolated, deep copy it
                elif isinstance(value, Isolated):
                    bound_args.arguments[name] = value.copy(bound_args.arguments[name])

            # Call the original function with the processed arguments
            return func(*bound_args.args, **bound_args.kwargs)

        return wrapper

    return decorator


# Example usage of smart_args with Isolated and Evaluated
@smart_args()
def check_isolation(*, d=Isolated()):
    d["a"] = 0  # Modify the dictionary
    return d


@smart_args()
def check_evaluation(*, x=10, y=Evaluated(lambda: 42)):
    print(x, y)  # Print the values (Evaluated should be evaluated)


# Example usage:
# Currying and uncurrying examples
f2 = curry_explicit(lambda x, y, z: f"<{x},{y},{z}>", 3)  # Curry a 3-argument function
g2 = uncurry_explicit(f2, 3)  # Uncurry the function back to its original form

if __name__ == "__main__":
    print(f2(123)(456)(562))  # Output: <123,456,562>
    print(g2(123, 456, 562))  # Output: <123,456,562>

    # Example of caching:
    @cache_results(max_size=3)
    def cached_function(x, y):
        return x + y

    # Example of smart arguments:
    check_isolation(d={"a": 10})  # Call with an isolated argument
    check_evaluation()  # Evaluate the `Evaluated` argument lazily
    check_evaluation(y=100)  # Override `Evaluated` with a regular value
