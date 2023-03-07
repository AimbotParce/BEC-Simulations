from math import factorial

import jax.numpy as jnp
from numpy.polynomial.hermite import hermval

import lib.constants as constants

constants.g = 0
n = int(input("Enter a value for n: "))
w = 1


def V(x, t):
    return 1 / 2 * constants.mass * w**2 * x**2


def waveFunction(x, t):
    const = 1 / jnp.sqrt(2**n * factorial(n)) * (constants.mass * w / (constants.hbar * jnp.pi)) ** (1 / 4)
    exponential = jnp.exp(-constants.mass * w * x**2 / (2 * constants.hbar))
    hermite = hermval(x * jnp.sqrt(constants.mass * w / constants.hbar), [0] * n + [1])
    return const * exponential * hermite
