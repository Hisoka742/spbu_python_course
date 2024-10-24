import itertools


def rgba_generator():
    return (
        (r, g, b, a)
        for r in range(256)
        for g in range(256)
        for b in range(256)
        for a in range(0, 101, 2)  # Only even values for transparency
    )


def get_rgba_element(i):
    """Returns the i-th element from the RGBA generator"""
    gen = rgba_generator()
    for index, value in enumerate(gen):
        if index == i:
            return value
    raise IndexError("Index out of bounds")


# Prime number generator with decorator
def prime_generator():
    """A generator that yields prime numbers"""
    primes = []
    num = 2
    while True:
        is_prime = all(num % p != 0 for p in primes)
        if is_prime:
            primes.append(num)
            yield num
        num += 1


def prime_decorator(func):
    """A decorator to return the k-th prime number"""

    def wrapper(k):
        prime_gen = prime_generator()
        prime = None
        for _ in range(k):
            prime = next(prime_gen)
        return prime

    return wrapper


@prime_decorator
def get_prime(k):
    """Returns the k-th prime number"""
    return k
