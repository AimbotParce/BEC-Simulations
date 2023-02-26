import logging as log
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy
from tqdm import tqdm

import lib.constants as constants
from lib.managers.logging import setupLog


@jax.jit
def computeConstantRight(x, r, interactionConstant, baseDensity, potential):
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex64)
    mainDiagonal = (
        jnp.ones(len(x), dtype=jnp.complex64) * (1j / r + 1) + potential / jnp.abs(interactionConstant) / baseDensity
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
def computeVariableRight(interactionConstant, baseDensity, psi):
    result = jnp.zeros((len(psi), len(psi)), dtype=jnp.complex64)
    mainDiagonal = jnp.abs(psi) ** 2 * interactionConstant / jnp.abs(interactionConstant) / baseDensity
    indices = jnp.diag_indices(len(psi))
    result = result.at[indices].set(mainDiagonal)
    return result


@jax.jit
def computeConstantLeft(x, r):
    result = jnp.zeros((len(x), len(x)), dtype=jnp.complex64)
    mainDiagonal = jnp.ones(len(x), dtype=jnp.complex64) * (1j / r - 1)
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


@jax.jit
def integrateProbability(psi):
    return jnp.sum(jnp.abs(psi) ** 2) * constants.dx


@jax.jit
def computeEnergy(psi, V):
    kineticEnergy = jnp.sum(jnp.abs(jnp.gradient(psi)) ** 2) * constants.dx
    potentialEnergy = jnp.sum(jnp.abs(psi) ** 2 * V) * constants.dx
    interactionEnergy = jnp.sum(jnp.abs(psi) ** 4) * constants.dx * constants.g / 2
    return kineticEnergy + potentialEnergy + interactionEnergy


def brightSolitonMalo(x, time=0):
    timeIndependent = (
        jnp.sqrt(constants.ns) / jnp.cosh(x / jnp.sqrt(2)) * jnp.exp(1j * x * constants.velocity / jnp.sqrt(2))
    )
    timeDependency = jnp.exp(1j * time)

    return timeIndependent * timeDependency


def brightSolitonWiki(x, time=0):
    timeIndependent = 1 / jnp.cosh(x)
    timeDependency = jnp.exp(-1j * constants.mu * time)
    return timeIndependent * timeDependency


def randomGaussian(x, time=0):
    return jnp.exp(-((x) ** 2) / 4 - 1j * x) / (2 * jnp.pi) ** (1 / 4)


def main():
    setupLog()

    x = jnp.arange(constants.xMin, constants.xMax, constants.dx)
    V = jnp.zeros_like(x)

    log.info("Crank-Nicolson method for the time evolution of the Gross-Pitaevskii equation")
    log.info("The Crank-Nicolson method solves the equation Ax(t+dt) = Bx(t)")
    log.info("A is a constant matrix, B has a constant part and a variable part")

    log.info("Computing A...")
    A = computeConstantLeft(x, constants.r)
    log.info("Computing the constant part of B...")
    Bconst = computeConstantRight(x, constants.r, constants.g, constants.ns, V)

    log.info("Running the simulation...")

    waveFunctionGenerator = brightSolitonWiki

    psi = jnp.zeros((constants.tCount, len(x)), dtype=jnp.complex64)
    psi = psi.at[0].set(waveFunctionGenerator(x, 0))

    for t in tqdm(range(constants.tCount - 1)):
        Bvar = computeVariableRight(constants.g, constants.ns, psi[t])
        right = (Bconst + Bvar) @ psi[t]
        psi = psi.at[t + 1].set(jnp.linalg.solve(A, right))

    log.info("Simulation finished. Plotting the results...")

    # Interactive figure
    fig, ax = plt.subplots()
    ax.set_ylim(constants.plotYMin, constants.plotYMax)
    ax.set_xlim(constants.xMin, constants.xMax)
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x), V(x)")
    ax.set_title("Simulation of the Gross-Pitaevskii equation")

    # Lines
    (potential,) = ax.plot(
        x, V / jnp.max(V) * (constants.plotYMax - constants.plotYMin) - constants.plotYMin, color="red"
    )
    (probability,) = ax.plot(x, jnp.abs(psi[0]) ** 2)
    (realPart,) = ax.plot(x, jnp.real(psi[0]))
    (imaginaryPart,) = ax.plot(x, jnp.imag(psi[0]))
    (theoretical,) = ax.plot(x, jnp.abs(waveFunctionGenerator(x, 0)) ** 2, color="black")

    # Legends
    ax.legend(
        (potential, probability, realPart, imaginaryPart, theoretical),
        ("V(x)", "Probability", "Real part", "Imaginary part", "Theoretical"),
        loc="lower right",
    )

    # Texts
    timeText = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    cumulativeProbabilityText = ax.text(0.02, 0.90, "", transform=ax.transAxes)
    energyText = ax.text(0.02, 0.85, "", transform=ax.transAxes)

    plt.ion()
    plt.show()

    for t in range(0, constants.tCount, constants.plotStep):
        time = t * constants.dt + constants.tMin

        # Update texts
        timeText.set_text("t = %.2f" % time)
        cumulativeProbabilityText.set_text("Cumulative probability = %.2f" % integrateProbability(psi[t]))
        energyText.set_text("Energy = %.8f" % computeEnergy(psi[t], V))

        # Update lines
        probability.set_ydata(jnp.abs(psi[t]) ** 2)
        realPart.set_ydata(jnp.real(psi[t]))
        imaginaryPart.set_ydata(jnp.imag(psi[t]))
        theoretical.set_ydata(jnp.abs(waveFunctionGenerator(x, time)) ** 2)

        # Update plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(constants.plotPause)


if __name__ == "__main__":
    main()
