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

from ..constants import Constants, isEligibleConstant
from ..interface.arguments import setupParser
from ..interface.logging import setupLog
from ..managers.animation import animate
from ..managers.crankNicolson import dimensionless as CNdefault
from ..managers.integrals import computeEnergy, computeIntegral
from ..managers.simulation import simulate
from ..utils.metadata import toJSON

jax.config.update("jax_enable_x64", True)

log = logging.getLogger("BECsimulations")


class SimulationLoader:
    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found")
        if not path.endswith(".py"):
            raise ValueError(f"File {path} must be a Python file")
        self.path = path

        log.info(f"Loading wave function and potential function from {path}")

        self.module = SourceFileLoader("module", path).load_module()
        self.waveFunctionGenerator = self.loadWaveFunction()
        self.V = self.loadV()

        log.info(f"Identifying constant overrides on simulation file")
        self.overriddenConstants = self.loadOverriddenConstants()

    def loadWaveFunction(self):
        if not hasattr(self.module, "waveFunction"):
            raise AttributeError(f"File {self.path} must have a waveFunction function")
        if not inspect.isfunction(self.module.waveFunction):
            raise AttributeError(f"waveFunction must be a function")

        # Check if function has the right signature
        signature = list(inspect.signature(self.module.waveFunction).parameters.keys())
        if not signature == ["x", "t", "constants"]:
            raise AttributeError(
                f"waveFunction must have the signature waveFunction(x, t), but has waveFunction({', '.join(signature)})"
            )

        return self.module.waveFunction

    def loadV(self):
        if not hasattr(self.module, "V"):
            raise AttributeError(f"File {self.path} must have a potential function (V)")
        if not inspect.isfunction(self.module.V):
            raise AttributeError(f"V must be a function")
        # Check if function has the right signature
        signature = list(inspect.signature(self.module.V).parameters.keys())
        if not signature == ["x", "t", "constants"]:
            raise AttributeError(f"V must have the signature V(x, t), but has V({', '.join(signature)})")
        return self.module.V

    def loadOverriddenConstants(self):
        """Any variable in the file that is not a function or module will be considered a constant"""

        return {key: value for key, value in self.module.__dict__.items() if isEligibleConstant(key, value)}


def run(
    args: Union[Namespace, dict],
    constants: Constants,
    CNModule=CNdefault,
    percentDict: dict = {},
):
    # If args.cpuOnly is True, then we will not use the GPU
    if args.cpuOnly or jax.lib.xla_bridge.get_backend().platform == "cpu":
        backend = "cpu"
        log.warning("Using CPU only mode.")
        jax.config.update("jax_platform_name", "cpu")
    else:
        backend = "gpu"

    # Load the wave function and potential function
    path = os.path.abspath(args.input)
    loader = SimulationLoader(path)
    waveFunctionGenerator, V = loader.waveFunctionGenerator, loader.V

    # Override constants from the file
    constants.override(loader.overriddenConstants)
    log.info("Using the following constants:")
    constants.print(logger=log.info)
    constants = constants.toJSON()

    # Setup the X and T arrays
    x = jnp.arange(constants["xMin"], constants["xMax"], constants["dx"])
    t = jnp.arange(constants["tMin"], constants["tMax"], constants["dt"])

    log.info("Compiling functions")
    jittedWaveFunction = jax.jit(waveFunctionGenerator, backend=backend)
    jittedV = jax.jit(V, backend=backend)
    log.info("Done")

    psi = simulate(x, t, jittedWaveFunction, jittedV, args, constants, CNModule, percentDict, backend=backend)

    psiTeo = jnp.zeros_like(psi)
    for j in range(0, constants["tCount"]):
        psiTeo = psiTeo.at[j].set(jittedWaveFunction(x, t[j], constants))

    log.info(f"Saving simulation to folder {args.output}")
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    jnp.save(os.path.join(args.output, "evolution.npy"), psi)

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
