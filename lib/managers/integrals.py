import jax
import jax.numpy as jnp

import lib.constants as constants


@jax.jit
def integrateProbability(x, psi):
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
def computeEnergy(x, t, psi, V):
    """
    Compute the energy of the system.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The space steps (shape: (xCount,))
    t : float
        Time
    psi : jax.numpy.ndarray
        The wave function at each time step (shape: (tCount, xCount))
    V : function
        The potential function at each time step (signature: V(x, t))
    """
    potential = V(x, t)

    kineticEnergy = jnp.sum(jnp.abs(jnp.gradient(psi)) ** 2) * constants.dx
    potentialEnergy = jnp.sum(jnp.abs(psi) ** 2 * potential) * constants.dx
    interactionEnergy = jnp.sum(jnp.abs(psi) ** 4) * constants.dx * constants.g / 2
    return kineticEnergy + potentialEnergy + interactionEnergy
