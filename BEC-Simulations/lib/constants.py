# To check if something is a callable or module
import inspect
import json
import logging as log

# import jax.numpy as jnp
import numpy as np


class Constants:
    ################################################# UNIVERSAL #################################################

    # hbar = 1.05457182e-34
    hbar = 1.0

    ################################################# SOLITON #################################################

    x0 = 0.0
    velocity = 0.0
    baseDensity = 1  # Base density in atoms per a0^3
    # https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Introduction_to_the_Physics_of_Atoms_Molecules_and_Photons_(Benedict)/01%3A_Chapters/1.08%3A_New_Page#:~:text=in%20a%20condensate-,the%20density%20is%20only,3,-.%20In%20fluids%20and
    N = 1400  # Particle count https://arxiv.org/pdf/1301.2073.pdf

    # mass = 1.16e-26  # kg https://arxiv.org/pdf/1301.2073.pdf
    mass = 1.0
    healingLength = 1.0

    @property
    def psi0(self):
        return np.sqrt(self.baseDensity)

    @property
    def U0(self):
        return -1 / (2 * self.baseDensity * self.healingLength**2)

    # @property
    # def healingLength(self):
    #     # https://arxiv.org/pdf/1301.2073.pdf
    #     return np.sqrt(self.hbar**2 / (2 * self.mass * np.abs(self.U0) * np.abs(self.psi0) ** 2))

    # @property
    # def chemicalPotential(self):
    #     return float(np.abs(self.U0) * self.baseDensity)

    ################################################# POTENTIAL #################################################

    # Potential constants
    a0 = 1.0  # Potential width

    @property
    def potentialW(self):
        # a0² = hbar / (m * w) -> w = hbar / (m * a0²)
        return self.hbar / self.mass / self.a0**2

    @property
    def oscillationPeriod(self):
        return 2 * np.pi / self.potentialW

    ################################################# SIMULATION #################################################

    dx = 0.05
    dt = 0.005
    xMin = -10.0
    xMax = 10.0
    tMin = 0.0
    tMax = 5.0

    @property
    def r(self):
        return self.dt / (self.dx**2)

    @property
    def xCount(self):
        return int((self.xMax - self.xMin) / self.dx)

    @property
    def tCount(self):
        return int((self.tMax - self.tMin) / self.dt)

    ################################################# ANIMATION #################################################

    plotPause = 0.001
    plotStep = 10
    plotYMax = 2
    plotYMin = -2

    @property
    def plotFPS(self):
        return 1 / self.plotPause

    ################################################# UTILS #################################################

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
                log.info("New constant %s with value %s. Setting it as float by default.", key, JSON[key])
                setattr(self, key, float(JSON[key]))

    def print(self, logger=print):
        if logger == print:
            logger(str(self))
        else:
            textLines = str(self).splitlines()[1:-1]  # Remove first and last line
            for line in textLines:
                logger(line.replace(",", ""))

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
