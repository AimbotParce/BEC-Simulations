
import os

from lib.constants import Constants
from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog
from lib.managers.run import getSimulatorModule, run

args = setupParser()
setupLog(level=args.verbose)


# Override constants (Do this after loading the wave function and potential function
# because they might have overridden some constants themselves)
constants = Constants()
constants.override(args.overrideConstants if args.overrideConstants else {})


CNModule = getSimulatorModule(os.path.abspath(args.CNmodule) if args.CNmodule else None)

run(args, constants, CNModule)
