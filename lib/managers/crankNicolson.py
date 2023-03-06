import jax
import jax.numpy as jnp

# Gross Pitaevskii solver


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
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
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

    # Periodic boundary conditions
    result = result.at[(0, -1)].set(others)
    result = result.at[(-1, 0)].set(others)

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
        The wave function at the future time step.
    potential : jnp.ndarray
        The potential at the future time step.
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
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = 4j * mass * dx**2 / hbar / dt - 2
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = 1
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    # Periodic boundary conditions
    result = result.at[(0, -1)].set(others)
    result = result.at[(-1, 0)].set(others)

    return result


import jax
import jax.numpy as jnp

# Just for testing, the Schrodinger equation:


@jax.jit
def computeRightScho(x, psi, potential, dx, dt, mass, hbar, interactionConstant):
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
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = hbar - 1j * dt * potential / 2 - hbar**2 * dt * 1j / (2 * mass * dx**2)
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = -(hbar**2) * dt / (4 * mass * dx**2 * 1j)
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    # Periodic boundary conditions
    result = result.at[(0, -1)].set(others)
    result = result.at[(-1, 0)].set(others)

    return result


@jax.jit
def computeLeftScho(x, psi, potential, dx, dt, mass, hbar, interactionConstant):
    """
    Computes the left hand side of the Crank-Nicolson equation.

    Parameters
    ----------
    x : jnp.ndarray
        The x values of the grid.
    psi : jnp.ndarray
        The wave function at the future time step.
    potential : jnp.ndarray
        The potential at the future time step.
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
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = hbar + 1j * dt * potential / 2 + hbar**2 * dt * 1j / (2 * mass * dx**2)
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = hbar**2 * dt / (4 * mass * dx**2 * 1j)
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    # Periodic boundary conditions
    result = result.at[(0, -1)].set(others)
    result = result.at[(-1, 0)].set(others)

    return result
