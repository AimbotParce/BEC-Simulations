import jax
import jax.numpy as jnp


@jax.jit
def computeConstantRight(x, dx, r, interactionConstant, baseDensity, potential):
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = (
        jnp.ones(len(x), dtype=jnp.complex128) * (1j / r + 1)
        + dx**2 * potential / jnp.abs(interactionConstant) / baseDensity
    )
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = -1 * jnp.ones(len(x) - 1) / 2
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    return result


@jax.jit
def computeVariableRight(dx, interactionConstant, baseDensity, psi):
    result = jnp.zeros((len(psi), len(psi)), dtype=jnp.complex128)
    mainDiagonal = dx**2 * jnp.abs(psi) ** 2 * interactionConstant / jnp.abs(interactionConstant) / baseDensity
    indices = jnp.diag_indices(len(psi))
    result = result.at[indices].set(mainDiagonal)
    return result


@jax.jit
def computeRight(x, psi, dx, r, interactionConstant, baseDensity, potential):
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = (
        jnp.ones(len(x), dtype=jnp.complex128) * (1j / r + 1)
        + dx**2 * potential / jnp.abs(interactionConstant) / baseDensity
        + dx**2 * jnp.abs(psi) ** 2 * interactionConstant / jnp.abs(interactionConstant) / baseDensity
    )
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = -1 * jnp.ones(len(x) - 1) / 2
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    return result


@jax.jit
def computeLeft(x, r):
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    mainDiagonal = jnp.ones(len(x), dtype=jnp.complex128) * (1j / r - 1)
    indices = jnp.diag_indices(len(x))
    result = result.at[indices].set(mainDiagonal)

    others = jnp.ones(len(x) - 1) / 2
    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0] + 1, indices[1])
    result = result.at[indices].set(others)

    indices = jnp.diag_indices(len(x) - 1)
    indices = (indices[0], indices[1] + 1)
    result = result.at[indices].set(others)

    return result
