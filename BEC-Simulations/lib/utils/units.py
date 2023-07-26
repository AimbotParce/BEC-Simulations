# Time and space units depending on the simulator.
import os


def get_unit(simulatorPath):
    simulator = os.path.splitext(os.path.basename(simulatorPath))[0]

    class unit:
        if simulator in ["default", "schrodinger"]:
            time = "s"
            space = "m"
            energy = "J"
        elif simulator == "dimensionless":
            time = r"\tau"
            space = r"a_0"
            energy = r"m\omega^2"

        @staticmethod
        def fmt(txt, unit=None):
            return f"{txt} $[{unit}]$" if unit else txt

        @classmethod
        def x(cls, txt):
            return cls.fmt(txt, unit.space)

        @classmethod
        def t(cls, txt):
            return cls.fmt(txt, unit.time)

        @classmethod
        def e(cls, txt):
            return cls.fmt(txt, unit.energy)

        @classmethod
        def wf(cls, txt):
            return cls.fmt(txt, unit.space + "^{-1/2}")

    return unit
