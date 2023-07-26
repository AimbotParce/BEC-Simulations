"""
Test the effect of the healing length compared to the potential well size on the falling of the soliton on the side of the potential well.
"""

import os
import subprocess
from pathlib import Path

# from concurrent.futures import ThreadPoolExecutor

cwd = Path(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(cwd)

# Input path is this file's path, but relative to the cwd
input_path = Path(os.path.join(os.path.dirname(__file__), "squeezing_potential_simulation.py"))


r = 0.3  # r = dt/dx^2


def run_simulation(healing_length):
    output_path = Path(os.path.join(os.path.dirname(__file__), "%.2f" % healing_length))
    os.makedirs(output_path, exist_ok=True)
    dx = healing_length / 10
    xMax = healing_length * 10 + 0.9 + healing_length**2 * 10
    dt = r * dx**2
    print(
        "Running simulation for Healing Length: %.2f with xMax=%.2f, dx=%.2f, dt=%.8f" % (healing_length, xMax, dx, dt)
    )
    cmd_list = [
        "python3",
        "BEC-Simulations",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        # "-cpu",
        # "--simple-text",
        "--override-constants",
        "healingLength=%.2f" % healing_length,
        "dx=%.8f" % dx,  # Ensure the dx is small enough
        "dt=%.8f" % dt,  # Ensure the dt is small enough
        "xMax=%.8f" % xMax,
        "xMin=%.8f" % -xMax,
        # "1>>" + os.path.join(output_path, "stderr.log"),
        # "2>>" + os.path.join(output_path, "stdout.log"),
    ]

    subprocess.run(cmd_list)


healingLengths = [0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1]

for healingLength in healingLengths:
    run_simulation(healingLength)
