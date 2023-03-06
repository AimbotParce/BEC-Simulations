import logging as log
from argparse import Namespace
from typing import Callable

import jax.numpy as jnp
from tqdm import tqdm

import lib.constants as constants
from lib.managers.crankNicolson import computeLeft, computeRight


def simulate(
    x: jnp.ndarray, t: jnp.ndarray, waveFunctionGenerator: Callable, V: Callable, arguments: Namespace
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
    arguments : Namespace
        The arguments passed to the program.

    """
    psi = jnp.zeros((len(t), len(x)), dtype=jnp.complex64)

    log.info("Crank-Nicolson method for the time evolution of the Gross-Pitaevskii equation")
    log.info("The Crank-Nicolson method solves the equation Ax(t+dt) = Bx(t)")
    log.info("A is constant, B must be computed at each time step")

    log.info("Computing A...")
    A = computeLeft(x, psi, V(0, 0), constants.dx, constants.dt, constants.mass, constants.hbar, constants.g)

    log.info("Running the simulation...")

    log.info("Memory allocated: %.2f MB", (psi.nbytes + x.nbytes + t.nbytes + A.nbytes * 2) / 1024 / 1024)
    #                                                                           ^ Take into account B

    psi = psi.at[0].set(waveFunctionGenerator(x, 0))

    for iteration in tqdm(range(0, constants.tCount), desc="Simulation"):
        time = t[iteration]
        potential = V(x, time)
        B = computeRight(
            x, psi[iteration], potential, constants.dx, constants.dt, constants.mass, constants.hbar, constants.g
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
