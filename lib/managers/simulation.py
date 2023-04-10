import logging
from argparse import Namespace
from types import ModuleType
from typing import Callable, Union

import jax
import jax.numpy as jnp
from tqdm import tqdm

log = logging.getLogger("BECsimulations")


def simulate(
    x: jnp.ndarray,
    t: jnp.ndarray,
    waveFunctionGenerator: Callable,
    V: Callable,
    arguments: Union[Namespace, dict],
    constants: dict,
    crankNicolson: ModuleType,
    percentDict: dict = {},
    backend: str = "gpu",
) -> jnp.ndarray:
    """
    Simulate the time evolution of the Gross-Pitaevskii equation using the Crank-Nicolson method.

    Parameters
    ----------
    x : jnp.ndarray
        The space grid.
    t : jnp.ndarray
        The time grid.
    waveFunctionGenerator : Callable
        The function that generates the initial wave function.
    V : Callable
        The potential function.
    arguments : Namespace | dict
        The arguments passed to the program.
    constants : dict
        The constants used in the simulation.
    crankNicolson : ModuleType
        The module containing the Crank-Nicolson functions (computeLeft, computeRight)
    percentDict : dict
        Dictionary onto which to set the variable "percet" to the current percentage of the simulation.
        THIS IS UGLY, BUT IT'S THE ONLY WAY I FOUND TO UPDATE THE PERCENTAGE IN THE MAIN THREAD (PASS BY REFERENCE)

    """
    if not "percent" in percentDict:
        percentDict["percent"] = 0

    disableTQDM = not log.isEnabledFor(logging.INFO)
    psi = jnp.zeros((len(t), len(x)), dtype=jnp.complex128)

    # log.info("Crank-Nicolson method for the time evolution of the Gross-Pitaevskii equation")
    # log.info("The Crank-Nicolson method solves the equation Ax(t+dt) = Bx(t)")
    # log.info("A and B can be computed at each time step")

    log.info("Precomputing the potential over time...")

    potential = jnp.zeros((len(t), len(x)), dtype=jnp.float64)
    for iteration in tqdm(range(0, len(t)), desc="Potential", disable=disableTQDM):
        potential = potential.at[iteration].set(V(x, t[iteration], constants))
    log.info("Running the simulation...")

    # Preallocate A
    A = jnp.zeros((len(x), len(x)), dtype=jnp.complex128)
    log.info(
        "Memory allocated: %.2f MB",
        (psi.nbytes * 2 + x.nbytes + t.nbytes + A.nbytes * 2 + potential.nbytes) / 1024 / 1024,
    )  #              ^ Consider theoretical               ^ Take into account B

    psi = psi.at[0].set(waveFunctionGenerator(x, 0, constants))
    # psi = psi.at[0, 0].set(0)  # Set the first element to 0 to avoid NaNs
    # psi = psi.at[0, -1].set(0)  # Set the last element to 0 to avoid NaNs
    # This doesn't do anything.

    computeLeft = jax.jit(crankNicolson.computeLeft, backend=backend)
    computeRight = jax.jit(crankNicolson.computeRight, backend=backend)

    for iteration in tqdm(range(0, constants["tCount"]), desc="Simulation", disable=disableTQDM):
        percentDict["percent"] = iteration / constants["tCount"] * 100
        time = t[iteration]
        A = computeLeft(
            x,
            psi[iteration],  # psi
            potential[iteration + 1],
            constants["dx"],
            constants["dt"],
            constants["mass"],
            constants["hbar"],
            constants["g"],
        )

        B = computeRight(
            x,
            psi[iteration],
            potential[iteration],
            constants["dx"],
            constants["dt"],
            constants["mass"],
            constants["hbar"],
            constants["g"],
        )
        right = B @ psi[iteration]
        psi = psi.at[iteration + 1].set(jnp.linalg.solve(A, right))

        if not arguments.ignoreNan:
            if iteration % 100 == 0:
                # Test if a NaN has been generated
                if jnp.isnan(psi[iteration - 101 : iteration + 1]).any():
                    log.error(
                        "NaN encountered at iteration %d. If you wish to ignore this, add -inan/--ignore-nan to execution command.",
                        iteration,
                    )
                    raise ValueError("NaN encountered")

    log.info("Simulation finished.")

    return psi
