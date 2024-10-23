import itertools

# RGBA Generator Function
import itertools

# Corrected RGBA Generator Function
def rgba_generator():
    """Generates RGBA values with R, G, B ranging from 0 to 255, and A taking even values from 0 to 100."""
    for r in range(256):
        for g in range(256):
            for b in range(256):
                for a in range(0, 101, 2):  # Only even alpha values
                    yield (r, g, b, a)


def get_nth_rgba(n):
    """Returns the nth RGBA vector from the generator."""
    # Total number of alpha values is 51 (0, 2, 4, ..., 100)
    alpha_range = 51

    # Calculate components based on n
    a = (n % alpha_range) * 2  # Alpha in steps of 2
    n //= alpha_range
    b = n % 256  # Blue component
    n //= 256
    g = n % 256  # Green component
    n //= 256
    r = n % 256  # Red component

    return (r, g, b, a)


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
