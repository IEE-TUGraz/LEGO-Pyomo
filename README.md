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

## Usage

### Running a model
```bash
python LEGO.py data/example
```

### Running parallel jobs
```bash
python InOutModule/Caller.py jobs.txt
python InOutModule/Caller.py jobs.txt --spawn 4   # 4 parallel workers
```

### Testing
```bash
pytest                                                            # all tests
pytest tests/test_examples.py                                     # specific file
pytest tests/test_examples.py::test_comparisonAgainstMPS          # specific test
```

## Architecture

LEGO uses a modular architecture where constraint modules are conditionally loaded based on case study parameters:

- **`LEGO/`**: Core optimization model. `LEGO.py` builds and solves Pyomo models; `modules/` contains constraint definitions for thermal generators, renewables, storage, reserves, import/export, and soft line limits.
- **`InOutModule/`** (git submodule): Data I/O. `CaseStudy.py` loads all Excel input files and provides data manipulation methods; `SQLiteWriter.py` exports results to SQLite.
- **`research/`**: Standalone experiment scripts. Each subfolder has its own README with usage details.

## Data Structure

Case studies are defined by Excel files in a data directory (e.g. `data/example/`):

| File                                           | Contents                                               |
|------------------------------------------------|--------------------------------------------------------|
| `Global_Parameters.xlsx`                       | Solver, RMIP mode, global settings                     |
| `Global_Scenarios.xlsx`                        | Stochastic scenario definitions and weights            |
| `Power_Parameters.xlsx`                        | Enable/disable modules (thermal gen, VRES, storage, …) |
| `Power_BusInfo.xlsx`                           | Bus/node definitions, ZOI flag, zone column            |
| `Power_Network.xlsx`                           | Transmission lines, technical representation per line  |
| `Power_ThermalGen.xlsx`                        | Thermal generator characteristics                      |
| `Power_VRES.xlsx`                              | Renewable generator definitions                        |
| `Power_Storage.xlsx`                           | Storage unit specifications                            |
| `Power_Demand.xlsx`                            | Time-indexed demand profiles                           |
| `Power_Inflows.xlsx`                           | Hydro inflow time series                               |
| `Power_VRESProfiles.xlsx`                      | VRES capacity factor profiles                          |
| `Power_WeightsRP.xlsx` / `Power_WeightsK.xlsx` | Time period weights                                    |
| `Power_Hindex.xlsx`                            | Mapping between periods and representative periods     |

## Research Experiments

Standalone experiment scripts live in `research/`. Each subfolder has a README with usage details:

- [`research/TR/`](research/TR/README.md) — Network technical representation (DC-OPF vs. Transport Problem vs. Single Node)
- [`research/MK/`](research/MK/README.md) — Markov chain edge-handling for representative periods
- [`research/ID/`](research/ID/README.md) — Impact of inflow data aggregation on optimization results