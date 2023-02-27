import logging as log
from typing import Callable

import jax.numpy as jnp
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import lib.constants as constants
from lib.managers.crankNicolson import computeLeft, computeRight


def simulate(x: jnp.ndarray, t: float, waveFunctionGenerator: Callable, V: Callable) -> jnp.ndarray:
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

    """

    log.info("Crank-Nicolson method for the time evolution of the Gross-Pitaevskii equation")
    log.info("The Crank-Nicolson method solves the equation Ax(t+dt) = Bx(t)")
    log.info("A is constant, B must be computed at each time step")

    log.info("Computing A...")
    A = computeLeft(x, constants.r)

    log.info("Running the simulation...")

    parameterTable = pd.DataFrame(
        {
            "X Step": [constants.dx],
            "X Interval": [constants.xMax - constants.xMin],
            "X Points": [constants.xCount],
            "T Step": [constants.dt],
            "T Interval": [constants.tMax - constants.tMin],
            "T Points": [constants.tCount],
            "velocity": [constants.velocity],
            "g": [constants.g],
            "ns": [constants.ns],
        }
    )

    log.info("Simulation parameters:\n%s", tabulate(parameterTable, headers="keys", tablefmt="psql"))

    psi = jnp.zeros((constants.tCount, len(x)), dtype=jnp.complex64)

    log.info("Memory allocated: %.2f MB", psi.nbytes / 1024 / 1024)

    psi = psi.at[0].set(waveFunctionGenerator(x, 0))

    for iteration in tqdm(range(0, constants.tCount), desc="Simulation"):
        time = t[iteration]
        potential = V(x, time)
        B = computeRight(x, psi[iteration], constants.dx, constants.r, constants.g, constants.ns, potential)
        right = B @ psi[iteration]
        psi = psi.at[iteration + 1].set(jnp.linalg.solve(A, right))

    log.info("Simulation finished.")

    return psi
