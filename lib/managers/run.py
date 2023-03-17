import base64
import inspect
import json
import logging
import marshal
import os
from argparse import Namespace
from importlib.machinery import SourceFileLoader
from typing import Union

import jax
import jax.numpy as jnp

from .. import constants
from ..interface.arguments import setupParser
from ..interface.logging import setupLog
from ..managers.animation import animate
from ..managers.crankNicolson import dimensionless as CNdefault
from ..managers.integrals import computeEnergy, computeIntegral
from ..managers.simulation import simulate
from ..utils.metadata import toJSON

jax.config.update("jax_enable_x64", True)

log = logging.getLogger("BECsimulations")


def loadWaveFunctionAndPotential(path):
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
    signature = list(inspect.signature(waveFunctionGenerator).parameters.keys())
    if not signature == ["x", "t", "constants"]:
        raise AttributeError(
            f"waveFunction must have the signature waveFunction(x, t), but has waveFunction({', '.join(signature)})"
        )

    if not hasattr(module, "V"):
        raise AttributeError(f"File {path} must have a potential function (V)")
    if not inspect.isfunction(module.V):
        raise AttributeError(f"V must be a function")
    V = module.V
    # Check if function has the right signature
    signature = list(inspect.signature(V).parameters.keys())
    if not signature == ["x", "t", "constants"]:
        raise AttributeError(f"V must have the signature V(x, t), but has V({', '.join(signature)})")

    return waveFunctionGenerator, V


def run(
    args: Union[Namespace, dict],
    constants: dict,
    CNModule=CNdefault,
    percentDict: dict = {},
):
    # Load the wave function and potential function
    path = os.path.abspath(args.input)
    waveFunctionGenerator, V = loadWaveFunctionAndPotential(path)

    # Setup the X and T arrays
    x = jnp.arange(constants["xMin"], constants["xMax"], constants["dx"])
    t = jnp.arange(constants["tMin"], constants["tMax"], constants["dt"])

    log.info("Compiling functions")
    jittedWaveFunction = jax.jit(waveFunctionGenerator)
    jittedV = jax.jit(V)
    log.info("Done")

    psi = simulate(x, t, jittedWaveFunction, jittedV, args, constants, CNModule, percentDict)

    psiTeo = jnp.zeros_like(psi)
    for j in range(0, constants["tCount"]):
        psiTeo = psiTeo.at[j].set(jittedWaveFunction(x, t[j], constants))

    log.info(f"Saving simulation to folder {args.output}")
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    jnp.save(os.path.join(args.output, "evolution.npy"), psi)

    if args.theoretical:
        jnp.save(os.path.join(args.output, "theoretical.npy"), psiTeo)
    # Save the metadata:
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(
            {
                "constants": toJSON(constants),
                "potential": base64.b64encode(marshal.dumps(V.__code__)).decode("utf-8"),
                "wave_function": base64.b64encode(marshal.dumps(waveFunctionGenerator.__code__)).decode("utf-8"),
                "simulator": CNModule.__name__,
            },
            f,
            indent=4,
        )

    if args.animate:
        animate(x, t, psi, jittedV, args, constants, computeEnergy, computeIntegral, psiTeo)


def getSimulatorModule(CNModPath: str = None):
    if CNModPath:
        log.info(f"Using Crank-Nicolson module from {CNModPath}")
        return SourceFileLoader("module", CNModPath).load_module()
    else:
        return CNdefault
