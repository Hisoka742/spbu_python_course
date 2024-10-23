import itertools

# RGBA Generator Function
def rgba_generator():
    return (
        (r, g, b, a)
        for r in range(256)
        for g in range(256)
        for b in range(256)
        for a in range(0, 101, 2)  # Only even alpha values
    )


def get_nth_rgba(n):
    """Returns the nth RGBA vector from the generator."""
    gen = rgba_generator()
    for i, val in enumerate(gen):
        if i == n:
            print(f"At index {n}: {val}")  # Add this for debugging
            return val


# Prime number generator with decorator
def prime_generator():
    """A generator function that yields an infinite sequence of prime numbers."""
    D = {}
    q = 2  # First number to test for primality
    while True:
        if q not in D:
            # q is a new prime number
            print(f"Prime found: {q}")  # Add this to print primes
            yield q
            D[q * q] = [q]
        else:
            for p in D[q]:
                D.setdefault(p + q, []).append(p)
            del D[q]
        q += 1


def prime_decorator(func):
    """Decorator to get the k-th prime number."""

    def wrapper(k):
        gen = func()
        return next(itertools.islice(gen, k - 1, None))  # k-th prime, 1-indexed

    return wrapper


@prime_decorator
def get_prime():
    return prime_generator()
