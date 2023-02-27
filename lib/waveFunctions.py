"""
Theoretical wave functions and initial conditions.
"""
import jax.numpy as jnp

import lib.constants as constants


def brightSolitonMalo(x, time=0):
    timeIndependent = (
        jnp.sqrt(constants.ns) / jnp.cosh(x / jnp.sqrt(2)) * jnp.exp(1j * x * constants.velocity / jnp.sqrt(2))
    )
    timeDependency = jnp.exp(1j * time)

    return timeIndependent * timeDependency


def brightSolitonWiki(x, time=0):
    timeIndependent = 1 / jnp.cosh(x)
    timeDependency = jnp.exp(-1j * constants.mu * time)
    return timeIndependent * timeDependency


def darkSolitonWiki(x, time=0):
    timeIndependent = jnp.tanh(x)
    timeDependency = 1
    return timeIndependent * timeDependency


def randomGaussian(x, time=0):
    return jnp.exp(-((x) ** 2) / 4 - 1j * x) / (2 * jnp.pi) ** (1 / 4)
