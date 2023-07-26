"""
Theoretical wave functions and initial conditions.
"""
import jax.numpy as jnp


def brightSolitonMalo(x, t, constants):
    timeIndependent = (
        jnp.sqrt(constants.baseDensity)
        / jnp.cosh((x - constants.x0) / jnp.sqrt(2))
        * jnp.exp(1j * (x - constants.x0) * constants.velocity / jnp.sqrt(2))
    )
    timeDependency = jnp.exp(1j * t)

    return timeIndependent * timeDependency


def brightSolitonWiki(x, t, constants):
    timeIndependent = 1 / jnp.cosh(x - constants.x0)
    timeDependency = jnp.exp(-1j * constants.chemicalPotential * t)
    return timeIndependent * timeDependency


def darkSolitonWiki(x, t, constants):
    timeIndependent = jnp.tanh(x - constants.x0)
    timeDependency = 1
    return timeIndependent * timeDependency


def randomGaussian(x, t, constants):
    return jnp.exp(-((x - constants.x0) ** 2) / 4 - 1j * constants.velocity * x) / (2 * jnp.pi) ** (1 / 4)


def brightSoliton(x, t, constants):
    v = constants["velocity"]
    g = constants["g"]
    x0 = constants["x0"]

    eta = jnp.sqrt((v**2 + 2) / (-2 * g))
    kappa = jnp.sqrt(2 / (v**2 + 2))

    spacePart = eta / jnp.cosh(((x - x0) - v * t) / kappa) * jnp.exp(1j * (x - x0) * v)
    timePart = jnp.exp(1j * (1 / 2 - v**2 / 4) * t)

    return spacePart * timePart


def stillBrightSoliton(x, t, constants):
    xi = constants["healingLength"]
    phi0 = constants["psi0"]
    x0 = constants["x0"]

    return jnp.sqrt(2) * phi0 / jnp.cosh((x - x0) / xi)
