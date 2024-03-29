# Numerical experiments: A fast algorithm with error bounds for Quadrature by Expansion

[![DOI](https://zenodo.org/badge/214558364.svg)](https://zenodo.org/badge/latestdoi/214558364)

This repository contains the code for (corrected) numerical experiments in the paper 'A fast
algorithm with error bounds for Quadrature by Expansion,' available at
[doi:10.1016/j.jcp.2018.05.006](https://doi.org/10.1016/j.jcp.2018.05.006) or on
[arXiv](https://arxiv.org/abs/1801.04070).

The code that is included reproduces the following parts of the paper:

* Tables 3, 4, 5, and 6
* Figures 14 and 15
* Data presented in Section 5.2.2

## Running Everything

Install the [Docker image](https://doi.org/10.5281/zenodo.3483367), and from a shell running in a
container, go to the code directory and type:
```
./run.sh
```
This script re-runs all experiments, and generates an output file
`summary/summary.pdf` containing all the generated figures, tables, and
data.

It's also possible to have more selective control over what gets run. See
[below](#running-experiments).

## Installation Hints

This code uses a special branch of
[Pytential](https://github.com/inducer/pytential), available
[here](https://github.com/mattwala/pytential/tree/fix-cost-model-for-2d-paper-code-v2).
Two options are available for obtaining the needed revisions of Pytential and
its dependencies.

### Docker Image

The simplest way to install is to use the
[Docker image](https://doi.org/10.5281/zenodo.3483367). The code
and software are installed in the image directory
`/home/inteq/gigaqbx-2d-results`.

### Manual Installation

If you don't want to use the Docker image, you can install necessary software
manually using the command:
```
./install.sh
```
This script downloads and installs software in an isolated environment in this
directory using Conda and pip.

For producing figures and outputs, you also need LaTeX and the
[latexrun](https://github.com/aclements/latexrun) script.

### Post-Installation

Before directly running the Python scripts in this directory, activate the
Conda environment and instruct PyOpenCL to use POCL, as follows:
```
source .miniconda3/bin/activate inteq
export PYOPENCL_TEST=portable
```

To ensure that everything works, you can run a short test:
```
py.test --disable-warnings utils.py
```

A more extensive set of tests can be found in the Pytential test suite (included
in `src/pytential/test`), which should also pass.

## Running Experiments

The scripts `generate-data.py` and `generate-figures-and-tables.py` can be used
to run individual experiments or groups of experiments, and postprocess
experimental outputs, respectively. Pass the `--help` option for more
documentation and the list of available experiments.

The `raw-data` directory is written to by `generate-data.py` and holds
experimental outputs. The `out` directory contains generated figures and tables
and is written to by `generate-figures-and-tables.py`.

To regenerate all outputs from the data that is already in the `raw-data`
directory, run

```
./generate-figures-and-tables.py --all
make -f makefile.summary
```

To run an individual experiment or to regenerate the data for a single
experiments, supply the command line option `-x experiment-name`. For instance, to
regenerate the results for the `bvp` experiment, run

```
./generate-data.py -x bvp
./generate-figures-and-tables.py -x bvp
```

## Contents

The following files and directories in this repository are included and/or
generated:

| Name | Description |
|----------------------------------|------------------------------------------------------------------------------------------------------------|
| `.miniconda3` | Conda install directory |
| `Dockerfile` | Used for generating the Docker image |
| `README.md` | This file |
| `env` | Files used by the installer to set up the Conda and pip environments |
| `generate-data.py` | Script for running experiments. Puts output in the `raw-data` directory |
| `generate-figures-and-tables.py` | Script for postprocessing experiments and producing figures and tables. Puts output in the `out` directory |
| `install.sh` | Installation script |
| `makefile.summary` | Makefile for generating the summary PDF |
| `out` | Holds generated figures and tables |
| `raw-data` | Holds data generated by experiments |
| `run.sh` | Script for re-running all experiments and generating all outputs |
| `src` | Pip install directory |
| `summary.tex` | Source code for summary PDF |
| `summary` | Holds generated summary PDF and auxiliary files |
| `utils.py` | Extra code used by `generate-data.py` |

## Citations

To cite the paper:
```
@article{gigaqbx2d,
  title = "A fast algorithm with error bounds for {Quadrature} by {Expansion}",
  journal = "Journal of Computational Physics",
  volume = "374",
  pages = "135 - 162",
  year = "2018",
  doi = "10.1016/j.jcp.2018.05.006",
  author = "Matt Wala and Andreas Klöckner",
}
```

To cite the repository and/or Docker image:
```
@software{gigaqbx2d_repo,
  author       = {Matt Wala and
                  Andreas Klöckner},
  title        = {mattwala/gigaqbx-2d-paper-code: Initial release},
  month        = nov,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.3530946},
  url          = {https://doi.org/10.5281/zenodo.3530946}
}

@software{gigaqbx2d_docker,
  author       = {Wala, Matt},
  title        = {{Docker image: A fast algorithm with error bounds 
                   for Quadrature by Expansion}},
  month        = nov,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.3483367},
  url          = {https://doi.org/10.5281/zenodo.3483367}
}
```
