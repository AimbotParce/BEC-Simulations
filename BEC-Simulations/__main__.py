import os
from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog

args = setupParser()
setupLog(level=args.verbose, simple=args.simpleText)

if args.cpuOnly:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from lib.constants import Constants
from lib.managers.run import getSimulatorModule, run


# Override constants (Do this after loading the wave function and potential function
# because they might have overridden some constants themselves)
constants = Constants()

CNModule = getSimulatorModule(os.path.abspath(args.CNmodule) if args.CNmodule else None)

run(args, constants, CNModule)
