"""Run a benchmark on the program for different sizes. Run the __main__ file directly with the provided args, but adding in the size for x"""
import logging
import os
import sys
import time

from lib.interface.arguments import setupParser
from lib.interface.logging import setupLog

args = setupParser()
setupLog(level=args.verbose)
log = logging.getLogger("BECsimulations")

log.info("Starting benchmark.")
dxArray = [0.2, 0.1, 0.05, 0.01, 0.005]

program = os.path.abspath(os.path.join(os.path.dirname(__file__), "__main__.py"))
outFile = os.path.abspath(os.path.join(args.output, "benchmark.txt"))


if not "-oc" in sys.argv and not "--override-constants" in sys.argv:
    log.info("Adding override constants to args.")
    sys.argv.append("-oc")

with open(outFile, "w") as f:
    f.write("# CPU benchmark\n")

log.info("Running simulations on CPU.")
for size in dxArray:
    args = " ".join(sys.argv[1:] + [f"dx={size}", "-cpu"])
    log.info(f"[CPU] Starting simulation with dx = {size}.")
    log.info(f"[CPU] Running command: python3 {program} {args}")
    start = time.time()
    os.system(f"python3 {program} {args}")
    end = time.time()
    log.info(f"[CPU] Finished simulation with dx = {size} in {end - start} seconds.")

    # Save time to output folder
    with open(outFile, "a") as f:
        f.write(f"{size}\t{end - start}\t[{args}]\n")

log.info("Running simulations on GPU.")

with open(outFile, "a") as f:
    f.write("\n\n# GPU benchmark\n")

for size in dxArray:
    args = " ".join(sys.argv[1:] + [f"dx={size}"])
    log.info(f"[GPU] Starting simulation with dx = {size}.")
    log.info(f"[GPU] Running command: python3 {program} {args}")
    start = time.time()
    os.system(f"python3 {program} {args}")
    end = time.time()
    log.info(f"[GPU] Finished simulation with dx = {size} in {end - start} seconds.")

    # Save time to output folder
    with open(outFile, "a") as f:
        f.write(f"{size}\t{end - start}\t[{args}]\n")
