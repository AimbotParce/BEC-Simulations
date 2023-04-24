import os
import shutil
import subprocess

path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

amps = [0.1, 0.5, 1.0, 2.0]

# Change directory to the folder just above BEC-Simulations
os.chdir(os.path.join(path, "..", "..", "..", "..", ".."))

for amp in amps:
    # Create folder
    folder = os.path.join(path, "amp" + str(amp))
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    # Copy file
    shutil.copy(os.path.join(path, "higherFreq.py"), folder)
    # Run simulation

    # New path to higherFreq.py
    simuFile = os.path.join(folder, "higherFreq.py")
    subprocess.run(
        [
            "python3",
            "BEC-Simulations",
            "-i",
            simuFile,
            "-o",
            folder,
            "--override-constants",
            "amplitude=" + str(amp),
        ]
    )
