import jax.numpy as jnp

import lib.constants as constants


def waveFunction(x, t):
    timeIndependent = (
        jnp.sqrt(constants.ns) / jnp.cosh(x - 3 / jnp.sqrt(2)) * jnp.exp(1j * x - 3 * constants.velocity / jnp.sqrt(2))
    )
    timeDependency = jnp.exp(1j * t)

    return timeIndependent * timeDependency


def V(x, t):
    return x**2
