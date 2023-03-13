import jax
import jax.numpy as jnp


@jax.jit
def computeNorm(x: jnp.ndarray, psi: jnp.ndarray, dx: float):
    """
    Compute the probability of finding the particle in the system.

    Parameters
    ----------
    x : jax.numpy.ndarray
        The space steps (shape: (xCount,))
    psi : jax.numpy.ndarray
        The wave function at each time step (shape: (tCount, xCount))
    """
    return jnp.sum(jnp.abs(psi) ** 2) * dx


@jax.jit
def computeEnergy(
    x: jnp.ndarray,
    t: float,
    psi: jnp.ndarray,
    V: jnp.ndarray,
    dx: float,
    interactionConstant: float,
    mass: float,
    hbar: float,
):
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

    kineticEnergy = -(hbar**2) / 2 / mass * jnp.gradient(jnp.gradient(psi, dx), dx)
    potentialEnergy = V * psi
    interactionEnergy = interactionConstant * jnp.abs(psi) ** 2 * psi

    return jnp.abs(jnp.sum(jnp.conjugate(psi) * (kineticEnergy + potentialEnergy + interactionEnergy))) * dx
