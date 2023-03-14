import os

from lib import constants
from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog
from lib.managers.run import getSimulatorModule, run

args = setupParser()
setupLog(level=args.verbose)

# Override constants (Do this after loading the wave function and potential function
# because they might have overridden some constants themselves)
constants.overrideConstants(args)

constants.printConstants()
constants.printSimulationParams()
constants.printAnimationParams()

CNModule = getSimulatorModule(os.path.abspath(args.CNmodule))

run(args, constants.toDict(), CNModule)
