import logging as log
import os
from datetime import datetime

import jax
import jax.numpy as np
import numpy

import lib.constants as constants
from lib.managers.logging import setupLog


@jax.jit
def computeConstantRight(result, x, interactionConstant, baseDensity, potential):
    result.at[:, :].set(1 + potential[:] / np.abs(interactionConstant) / baseDensity)


def computeConstantRightUNJITTED(result, x, interactionConstant, baseDensity, potential):
    for i in range(len(x)):
        result[i, i] = 1 + potential[i] / numpy.abs(interactionConstant) / baseDensity


def main():
    setupLog()

    x = np.arange(constants.xMin, constants.xMax, constants.dx)

    # psi = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)

    log.info("Computing B")
    start = datetime.now()

    B = np.zeros((len(x), len(x)))
    computeConstantRight(B, x, constants.g, constants.ns, x**2)
    delta = datetime.now() - start
    log.info("Finished in %s" % delta)
    print(B)

    log.info("Repeating without jit")
    start = datetime.now()

    B = numpy.zeros((len(x), len(x)))
    computeConstantRightUNJITTED(B, x, constants.g, constants.ns, x**2)

    delta = datetime.now() - start
    log.info("Finished in %s" % delta)
    print(B)


if __name__ == "__main__":
    main()
