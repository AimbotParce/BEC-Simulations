"""Run a benchmark on the program for different sizes. Run the __main__ file directly with the provided args, but adding in the size for x"""
import logging
import os

from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog

args = setupParser()
setupLog(level=args.verbose)
log = logging.getLogger("BECsimulations")

log.info("Starting benchmark.")

log.info("Running simulations without gpu.")
