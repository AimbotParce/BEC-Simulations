import logging as log

import jax
import jax.numpy as jnp

import lib.constants as constants
from lib.interface.logging import setupLog
from lib.managers.animation import animate
from lib.managers.simulation import simulate
from lib.waveFunctions import *


def main():
    setupLog()

    x = jnp.arange(constants.xMin, constants.xMax, constants.dx)
    t = jnp.arange(constants.tMin, constants.tMax, constants.dt)
    waveFunctionGenerator = brightSolitonMalo

    psi = simulate(x, t, waveFunctionGenerator, V)
    animate(x, t, psi, V, waveFunctionGenerator)
