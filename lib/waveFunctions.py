"""
Theoretical wave functions and initial conditions.
"""
import jax.numpy as jnp

import lib.constants as constants


def brightSolitonMalo(x, t=0):
    timeIndependent = (
        jnp.sqrt(constants.baseDensity) / jnp.cosh(x / jnp.sqrt(2)) * jnp.exp(1j * x * constants.velocity / jnp.sqrt(2))
    )
    timeDependency = jnp.exp(1j * t)

    return timeIndependent * timeDependency


def brightSolitonWiki(x, t=0):
    timeIndependent = 1 / jnp.cosh(x)
    timeDependency = jnp.exp(-1j * constants.chemicalPotential * t)
    return timeIndependent * timeDependency


def darkSolitonWiki(x, t=0):
    timeIndependent = jnp.tanh(x)
    timeDependency = 1
    return timeIndependent * timeDependency


def randomGaussian(x, t=0):
    return jnp.exp(-((x) ** 2) / 4 - 1j * x) / (2 * jnp.pi) ** (1 / 4)
