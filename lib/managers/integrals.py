import jax
import jax.numpy as jnp

import lib.constants as constants


@jax.jit
def integrateProbability(x: jnp.ndarray, psi: jnp.ndarray):
    """
    Compute the probability of finding the particle in the system.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The space steps (shape: (xCount,))
    psi : jax.numpy.ndarray
        The wave function at each time step (shape: (tCount, xCount))
    """
    return jnp.sum(jnp.abs(psi) ** 2) * constants.dx


@jax.jit
def computeEnergy(x: jnp.ndarray, t: float, psi: jnp.ndarray, V: jnp.ndarray):
    """
    Compute the energy of the system.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The space steps (shape: (xCount,))
    t : float
        Time
    psi : jax.numpy.ndarray
        The wave function at each time step (shape: (tCount,))
    V : jax.numpy.ndarray
        The potential function at each time step (signature: V(x, t))
    """

    kineticEnergy = (
        -constants.hbar**2 / 2 / constants.mass * jnp.gradient(jnp.gradient(psi, constants.dx), constants.dx)
    )
    potentialEnergy = V * psi
    interactionEnergy = constants.g * jnp.abs(psi) ** 2 * psi

    return jnp.abs(jnp.sum(jnp.conjugate(psi) * (kineticEnergy + potentialEnergy + interactionEnergy))) * constants.dx
