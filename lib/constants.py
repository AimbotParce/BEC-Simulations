import logging as log
import sys

import jax.numpy as jnp
import pandas as pd
from tabulate import tabulate

# Constants
g = -1000
baseDensity = 1
chemicalPotential = jnp.abs(g) * baseDensity
hbar = 1
mass = 1
velocity = 0
healingLength = hbar / jnp.sqrt(2 * mass * chemicalPotential)


# Space and time steps [m] and [s]
dx = 0.2
dt = 0.005
r = dt / (dx**2)

# Space interval [m]
xMin = -25
xMax = 25

# Time interval [s]
tMin = 0
tMax = 10

# Number of space and time steps
xCount = int((xMax - xMin) / dx)
tCount = int((tMax - tMin) / dt)


# Plot constants
plotPause = 0.001
plotFPS = 1 / plotPause
plotStep = 10
plotYMax = 2
plotYMin = -2


def printConstants():
    constantsTable = pd.DataFrame(
        {
            "g\n(g)": [g],
            "Base Density\n(baseDensity)": [baseDensity],
            "Chemical Potential\n(chemicalPotential)": [chemicalPotential],
            "hbar\n(hbar)": [hbar],
            "Mass\n(mass)": [mass],
            "Velocity\n(velocity)": [velocity],
            "Healing Length\n(healingLength)": [healingLength],
        }
    )

    log.info("Constants:\n%s", tabulate(constantsTable, headers="keys", tablefmt="psql", showindex=False))


def printSimulationParams():
    parameterTable = pd.DataFrame(
        {
            "X Step\n(dx)": [dx],
            "X Interval\n(xMax-xMin)": [xMax - xMin],
            "X Points\n(xCount)": [xCount],
            "T Step\n(dt)": [dt],
            "T Interval\n(tMax-tMin)": [tMax - tMin],
            "T Points\n(tCount)": [tCount],
        }
    )

    log.info("Simulation parameters:\n%s", tabulate(parameterTable, headers="keys", tablefmt="psql", showindex=False))


def printAnimationParams():
    parameterTable = pd.DataFrame(
        {
            "Pause\n(plotPause)": [plotPause],
            "FPS\n(plotFPS)": [plotFPS],
            "Step\n(plotStep)": [plotStep],
            "Y Max\n(plotYMax)": [plotYMax],
            "Y Min\n(plotYMin)": [plotYMin],
        }
    )

    log.info("Animation parameters:\n%s", tabulate(parameterTable, headers="keys", tablefmt="psql", showindex=False))


def overrideConstants(args):
    """
    Override the constants with the ones specified in the command line arguments
    """
    existingConstants = {
        name: getattr(sys.modules[__name__], name) for name in dir(sys.modules[__name__]) if not name.startswith("_")
    }

    if args.overrideConstants:
        for constant in args.overrideConstants:
            if "=" not in constant:
                log.error(f"Invalid constant override {constant}")
                continue

            name, value = constant.split("=")
            if not name in existingConstants:
                log.error(f"Invalid constant override {constant}")
                continue

            try:
                value = type(existingConstants[name])(value)
            except ValueError:
                log.error(f"Invalid constant override {constant}")
                continue

            setattr(sys.modules[__name__], name, value)

            log.info(f"Overriding constant {name} with value {value}")
