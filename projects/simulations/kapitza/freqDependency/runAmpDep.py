import os
import shutil
import subprocess

path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

freqs = [1.0, 2.0, 3.0, 4.0, 5.0]

# Change directory to the folder just above BEC-Simulations
os.chdir(os.path.join(path, "..", "..", "..", "..", ".."))

for freq in freqs:
    # Create folder
    folder = os.path.join(path, "freq" + str(freq))
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    # Copy file
    shutil.copy(os.path.join(path, "stabilized.py"), folder)
    shutil.copy(os.path.join(path, "plotExpectedValue.py"), folder)
    # Run simulation

    # New path to higherFreq.py
    simuFile = os.path.join(folder, "stabilized.py")
    subprocess.run(
        [
            "python3",
            "BEC-Simulations",
            "-i",
            simuFile,
            "-o",
            folder,
            "--override-constants",
            "amplitude=" + str(freq),
        ]
    )

    # Run the plot script
    os.chdir(folder)
    subprocess.run(
        [
            "python3",
            "plotExpectedValue.py",
        ]
    )
