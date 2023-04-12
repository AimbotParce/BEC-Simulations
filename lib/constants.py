# To check if something is a callable or module
import inspect
import json
import logging as log
import sys
from dataclasses import dataclass

import jax.numpy as jnp


class Constants:
    # Constants
    g = -1
    baseDensity = 1
    hbar = 1
    mass = 1
    velocity = 0

    # Space and time steps [m] and [s]
    x0 = 0
    dx = 0.2
    dt = 0.005

    # Space interval [m]
    xMin = -10
    xMax = 10

    # Time interval [s]
    tMin = 0
    tMax = 5

    # Plot constants
    plotPause = 0.001
    plotStep = 10
    plotYMax = 2
    plotYMin = -2

    @property
    def chemicalPotential(self):
        return float(jnp.abs(self.g) * self.baseDensity)

    @property
    def healingLength(self):
        return float(self.hbar / jnp.sqrt(2 * self.mass * self.chemicalPotential))

    @property
    def r(self):
        return self.dt / (self.dx**2)

    @property
    def xCount(self):
        return int((self.xMax - self.xMin) / self.dx)

    @property
    def tCount(self):
        return int((self.tMax - self.tMin) / self.dt)

    @property
    def plotFPS(self):
        return 1 / self.plotPause

    def toJSON(self):
        for k in dir(self):
            if k.startswith("_") or k == "toJSON":
                continue
        return {name: getattr(self, name) for name in dir(self) if isEligibleConstant(name, getattr(self, name))}

    def __str__(self):
        out = {key: value if isJSONSerializable(value) else str(value) for key, value in self.toJSON().items()}
        return json.dumps(out, indent=4)

    def override(self, JSON):
        for key in JSON:
            if hasattr(self, key):
                try:
                    t = type(getattr(self, key))
                    setattr(self, key, t(JSON[key]))
                    log.info(f"Overriding constant {key} with value {JSON[key]} as type {t}")
                except:
                    log.warning("Could not override constant %s with correct type. Saving it raw.", key)
                    setattr(self, key, JSON[key])
            else:
                log.info("New constant %s with value %s", key, JSON[key])
                setattr(self, key, JSON[key])

    def print(self, logger=print):
        if logger == print:
            logger(str(self))
        else:
            textLines = str(self).splitlines()
            for line in textLines:
                logger(line)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def isEligibleConstant(key, obj):
    return not (inspect.isfunction(obj) or inspect.ismodule(obj) or inspect.ismethod(obj)) and not key.startswith("_")


def isJSONSerializable(obj):
    try:
        json.dumps(obj)
        return True
    except:
        return False
