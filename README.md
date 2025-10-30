# Low-carbon Expansion Generation Optimization (LEGO) model

LEGO is a mixed integer quadratically constrained optimization problem and has been designed to be a multi-purpose tool,
like a Swiss army knife, that can be employed to study many different aspects of the energy sector. Ranging from
short-term unit commitment to long-term generation and transmission expansion planning. The underlying modeling
philosophies are: modularity and flexibility. LEGO is also composed of thematic modules that can be added or removed
from the model easily via data options depending on the scope of the study. You can use this code freely according to
the [LICENSE file](https://github.com/IEE-TUGraz/LEGO-Pyomo/blob/main/LICENSE), but please cite our
paper [[1]](https://doi.org/10.1016/j.softx.2022.101141) if you do.

## Setup

1. Install MPI implementation according to your OS
   (e.g., [MPICH](https://www.mpich.org/downloads/), [OpenMPI](https://www.open-mpi.org/software/ompi/v4.1/)
   or [Microsoft-MPI](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi))
2. Create environment from the 'environment.yml' file
    1. _For Pros_: Use whatever environment manager you like, e.g., for `conda`:
       ```bash
       conda env create -f environment.yml
       conda activate LEGO-Pyomo_env
       ```
    2. _For all others_: Use the `Conda-Activation-Scripts` provided in this repository:
        1. Download Anaconda to manage Python packages: https://www.anaconda.com/download
        2. Choose **"Just Me (recommended)"** when asked for the installation type
        3. Leave **all other options on default**
        4. Execute the activation script:
            - **Windows**: Execute the `activate_environment_windows.bat` file
            - **Unix**: Execute the `activate_environment_unix.sh` file
                - When executing it from terminal, use ```source activate_environment_unix.sh```
        5. In the now opened command line, you can use the activated environment. The working directory will be the
           **current folder** of the script
3. Test if it works
    - Run the following command in the now opened terminal:
      ```bash
      python LEGO.py data/example
      ```