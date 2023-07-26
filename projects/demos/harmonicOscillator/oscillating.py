import jax.numpy as jnp

import lib.constants as constants

# g = 0
x0 = 3
w = 1


def V(x, t, constants):
    return 1 / 2 * constants["mass"] * constants["w"] ** 2 * x**2


def waveFunction(x, t, constants):
    const = (constants["mass"] * w / (constants["hbar"] * jnp.pi)) ** (1 / 4)
    exponential = jnp.exp(-constants["mass"] * w * (x - constants["x0"]) ** 2 / (2 * constants["hbar"]))
    return const * exponential
