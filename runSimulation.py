import logging as log

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

import lib.constants as constants
from lib.crankNicolson import computeLeft, computeRight
from lib.managers.logging import setupLog
from lib.waveFunctions import *


@jax.jit
def integrateProbability(psi):
    return jnp.sum(jnp.abs(psi) ** 2) * constants.dx


@jax.jit
def computeEnergy(psi, V):
    kineticEnergy = jnp.sum(jnp.abs(jnp.gradient(psi)) ** 2) * constants.dx
    potentialEnergy = jnp.sum(jnp.abs(psi) ** 2 * V) * constants.dx
    interactionEnergy = jnp.sum(jnp.abs(psi) ** 4) * constants.dx * constants.g / 2
    return kineticEnergy + potentialEnergy + interactionEnergy


@jax.jit
def V(x, t):
    return x**2


@jax.jit
def computeVplot(x, time):
    v = V(x, time)
    return v / jnp.max(v) * (constants.plotYMax - constants.plotYMin) + constants.plotYMin


def main():
    setupLog()

    x = jnp.arange(constants.xMin, constants.xMax, constants.dx)
    waveFunctionGenerator = brightSolitonMalo

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

    for t in tqdm(range(constants.tCount - 1), desc="Simulation"):
        potential = V(x, t * constants.dt + constants.tMin)
        B = computeRight(x, psi[t], constants.dx, constants.r, constants.g, constants.ns, potential)
        right = B @ psi[t]
        psi = psi.at[t + 1].set(jnp.linalg.solve(A, right))

    log.info("Simulation finished. Plotting the results...")

    # Animation
    fig, ax = plt.subplots()
    ax.set_ylim(constants.plotYMin, constants.plotYMax)
    ax.set_xlim(constants.xMin, constants.xMax)
    ax.set_xlabel("x")
    ax.set_ylabel("psi(x), V(x)")
    ax.set_title("Simulation of the Gross-Pitaevskii equation")

    # Lines
    (potential,) = ax.plot(x, computeVplot(x, 0), color="red")
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

    def animate(t):
        time = t * constants.dt + constants.tMin

        # Update texts
        timeText.set_text("t = %.2f" % time)
        cumulativeProbabilityText.set_text("Cumulative probability = %.2f" % integrateProbability(psi[t]))
        energyText.set_text("Energy = %.8f" % computeEnergy(psi[t], V(x, time)))

        # Update lines
        potential.set_ydata(computeVplot(x, time))
        probability.set_ydata(jnp.abs(psi[t]) ** 2)
        realPart.set_ydata(jnp.real(psi[t]))
        imaginaryPart.set_ydata(jnp.imag(psi[t]))
        theoretical.set_ydata(jnp.abs(waveFunctionGenerator(x, time)) ** 2)

    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, constants.tCount, constants.plotStep), interval=1, repeat=True
    )
    plt.show()


if __name__ == "__main__":
    main()
