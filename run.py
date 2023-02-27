import inspect
import logging as log
import os
from importlib.machinery import SourceFileLoader

import jax
import jax.numpy as jnp

import lib.constants as constants
from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog
from lib.managers.animation import animate
from lib.managers.simulation import simulate
from lib.waveFunctions import *


def main():
    args = setupParser()
    setupLog(level=args.verbose)

    # Setup the X and T arrays
    x = jnp.arange(constants.xMin, constants.xMax, constants.dx)
    t = jnp.arange(constants.tMin, constants.tMax, constants.dt)

    # Load the wave function and potential function
    path = os.path.abspath(args.input)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    if not path.endswith(".py"):
        raise ValueError(f"File {path} must be a Python file")

    log.info(f"Loading wave function and potential function from {path}")

    module = SourceFileLoader("module", path).load_module()
    if not hasattr(module, "waveFunction"):
        raise AttributeError(f"File {path} must have a waveFunction function")
    if not inspect.isfunction(module.waveFunction):
        raise AttributeError(f"waveFunction must be a function")
    waveFunctionGenerator = module.waveFunction
    # Check if function has the right signature
    if not inspect.signature(waveFunctionGenerator).parameters.keys == ["x", "t"]:
        raise AttributeError(f"waveFunction must have the signature waveFunction(x, t)")

    if not hasattr(module, "V"):
        raise AttributeError(f"File {path} must have a potential function (V)")
    if not inspect.isfunction(module.V):
        raise AttributeError(f"V must be a function")
    V = module.V
    # Check if function has the right signature
    if not inspect.signature(V).parameters.keys == ["x", "t"]:
        raise AttributeError(f"V must have the signature V(x, t)")

    log.info("Compiling functions")
    waveFunctionGenerator = jax.jit(waveFunctionGenerator)
    V = jax.jit(V)

    psi = simulate(x, t, waveFunctionGenerator, V)
    animate(x, t, psi, V, waveFunctionGenerator)