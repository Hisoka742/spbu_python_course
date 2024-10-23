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
    """Returns the nth RGBA vector from the generator, calculated directly."""
    # Total number of combinations for each channel
    total_combinations = (
        256 * 256 * 256 * 51
    )  # R, G, B (256 each) and A (51 even numbers from 0 to 100)

    if n >= total_combinations or n < 0:
        raise ValueError("n is out of bounds for RGBA combinations")

    # Calculate each component directly from n
    a = (n % 51) * 2  # There are 51 even values for alpha, starting at 0
    n //= 51
    b = n % 256
    n //= 256
    g = n % 256
    n //= 256
    r = n % 256

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
