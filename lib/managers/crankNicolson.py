import jax
import jax.numpy as jnp


@jax.jit
def computeRight(x, psi, potential, dx, dt, mass, hbar, interactionConstant):
    """
    Computes the right hand side of the Crank-Nicolson equation.

    Parameters
    ----------
    x : jnp.ndarray
        The x values of the grid.
    psi : jnp.ndarray
        The wave function at the current time step.
    potential : jnp.ndarray
        The potential at the current time step.
    dx : float
        The step size in x.
    dt : float
        The step size in t.
    mass : float
        The mass of the particle.
    hbar : float
        The reduced Planck constant.
    interactionConstant : float
        The interaction constant. (g)
    """
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex64)
    mainDiagonal = (
        4 * mass * dx**2 / (hbar**2) * (1j * hbar / dt + potential + interactionConstant * jnp.abs(psi) ** 2) + 2
    )
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = -1
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    return result


@jax.jit
def computeLeft(x, psi, potential, dx, dt, mass, hbar, interactionConstant):
    """
    Computes the left hand side of the Crank-Nicolson equation.

    Parameters
    ----------
    x : jnp.ndarray
        The x values of the grid.
    psi : jnp.ndarray
        The wave function at the current time step.
    potential : jnp.ndarray
        The potential at the current time step.
    dx : float
        The step size in x.
    dt : float
        The step size in t.
    mass : float
        The mass of the particle.
    hbar : float
        The reduced Planck constant.
    interactionConstant : float
        The interaction constant. (g)
    """
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex64)
    mainDiagonal = (
        4 * mass * dx**2 / (hbar**2) * (1j * hbar / dt - potential - interactionConstant * jnp.abs(psi) ** 2) - 2
    )
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = 1
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    return result
