<p align="center">
  <img src="https://user-images.githubusercontent.com/40344474/223997006-4c4ae49e-3a96-43cb-95ef-9a9727b502d5.png#gh-light-mode-only" width=30% >
  <img src="https://user-images.githubusercontent.com/40344474/223997775-5516082f-edf9-4c87-9c30-a5734f2d2321.png#gh-dark-mode-only" width=30% >
</p>

## Simulation of a Bose Einstein Condensate

A simulation program for Bose Einstein Condensates (BECs) that uses the **Crank-Nicolson method** is a powerful tool for investigating the properties of these exotic quantum systems. This program allows the user to specify any wave function and potential for the BEC, and then use the Crank-Nicolson method to simulate the dynamics of the system over time. The Crank-Nicolson method is a numerical technique for solving partial differential equations that is especially well-suited for simulating quantum systems like BECs, where the wave function is described by a complex-valued function that satisfies the **Gross-Pitaevskii equation**.

To accelerate the simulation process, this program uses **JAX** instead of simple numpy to perform calculations on the GPU. JAX is a machine learning library that provides automatic differentiation and high-performance computing features. It is particularly useful for accelerating scientific computing tasks that involve large arrays and complex mathematical operations, making it an ideal choice for simulating BECs. By using JAX, this simulation program can run simulations faster and more efficiently than if it were relying solely on numpy, allowing researchers to explore a wider range of BEC scenarios and investigate more complex systems.

This program is being used to simulate Bose-Einstein Condensate solutions such as **bright and dark solitons** on a Kapitza pendulum-like potential. This potential creates a periodic modulation in space and time, which can be used to study the behavior of BECs in non-trivial geometries.

## Installation

To install the program, one must first download the latest release from [github's latest release](https://github.com/AimbotParce/BEC-Simulations/releases/latest).
Unzip the **source code** on a file of your choice.
  
Then, one must install the dependencies:
  
```bash
pip3 install -r requirements.txt
```

Keep in mind that this program uses JAX, which requires a GPU to run. If you do not have a GPU, you can still run the program, but it will be much slower. If you do have a GPU, you should install the GPU drivers, CUDA, and cuDNN. I strongly recommend, as well, that you compile JAX from source.

You can find detailed instructions on how to install JAX on JAX's [installation guide](https://github.com/google/jax#installation). JAX can work in Windows Subsystem for Linux (WSL), but I didn't manage to make it work on GPU.

Here's what I did to install JAX on my machine:

1. Install [nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) with `sudo apt install nvidia-driver-XXX`, version `525.105.17` in my case.
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), versions `12.0` and `8.8.1.3` in my case.

I recommend checking the [compatibility matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html) to make sure you're installing the correct versions.

3. Install JAX with `pip install --upgrade "jax[cudaXX_cudnnYY]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`, where `XX` is the CUDA version you installed, and `YY` is the cuDNN version.

One can check if JAX-GPU is working by checking the output of `jax.devices()`.

## Getting Started

To create a new simulation, one can just make a new **.py** file anywhere, onto which one must add two functions: `waveFunction(x, t, constants) -> jax.numpy.ndarray`, which takes `x : jax.numpy.ndarray` and `t : float` as the space grid (in meters) and the time (in seconds), as well as `constants:dict` as a dictionary containing the constants for the simulation, and returns the complex wave function evaluated over all the position lattice x at time t; and `V(x, t, constants) -> jax.numpy.ndarray`, which also takes `x`, `t` and `constants` and shall return the real function for the potential evaluated over all x and at time t.

## Usage

One can then run the simulation by executing the following command:

```bash
python3 BEC-Simulations -i <path> -o <path> [-h] [-v <level>] [-a] [-cn <path>] [-sp] [-t] [-oc <list>] [-cpu] [-s]
```

Options:
* `-h, --help`
Show help message and exit
* `-v {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --verbose {DEBUG,INFO,WARNING,ERROR,CRITICAL}`
Setup logging level
* `-i INPUT, --input INPUT`
Python file with the wave function and potential functions
* `-o OUTPUT, --output OUTPUT`
Output folder for the simulation
* `-a, --animate`
Show the animation of the simulation
* `-cn CNMODULE, --crank-nicolson CNMODULE`
Python file with the Crank-Nicolson module
* `-sp, --show-parts`
Show imaginary and real parts of the wave function
* `-t, --show-theoretical`
Show the theoreical time dependent wave function in the animation
* `-oc OVERRIDECONSTANTS [OVERRIDECONSTANTS ...], --override-constants OVERRIDECONSTANTS [OVERRIDECONSTANTS ...]`
Override the constants with the provided values (provided as a list of key=value pairs)
* `-cpu, --cpu-only`
Run the simulation only on the CPU
* `-s, --simple-text`
Do not add colors to the text

Note that **BEC-Simulations** is the name of the folder, and is not a `.py` file, however it has a `__main__.py` file, which is the one that is executed when the command is run.

When a simulation is finished, the following script can be used to create an automatic **report** on the results. Run:

```bash
python3 utils/report.py -i <path> [-v <LEVEL>]
```
Where `-i/--input` sets the path to the folder containing all of the results and metadata for the simulation. This program will create a new folder named `report_images` containing all the graphics and diagrams of the results, and then create a `report.pdf` file **inside the input folder** with a summary of the parameters of the simulation, as well as the results and graphs.
