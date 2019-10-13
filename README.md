# Instructions for Reproducing Results

## Installation Hints

### Docker Image

A [Docker image](http://dx.doi.org/10.5281/zenodo.3483367) preinstalled with
necessary software and the code in this repository is available.

### Manual Installation

The script `install.sh` downloads and installs all necessary software in an
isolated environment using Conda and pip.

For producing figures and outputs, you will also need LaTeX and the
[latexrun](https://github.com/aclements/latexrun) script.

### Post-Installation

Before running, activate the Conda environment and instruct PyOpenCL to use
POCL, as follows:
```
source .miniconda2/bin/activate inteq
export PYOPENCL_TEST=portable
```

To ensure that everything works, you can run a short test:
```
py.test --disable-warnings utils.py
```

A more extensive set of tests can be found in the Pytential test suite (included
in `src/pytential/test`), which should also pass.

## Running Experiments

To re-run all experiments and regenerate all experimental outputs, use the
command `./run.sh`. This may take a long time. It is also possible to
selectively run experiments, as detailed below.

The code in this directory generates the following outputs:

* experimental data stored in the `raw-data` directory;
* tables and figures stored in the `out` directory;
* a PDF file `summary/summary.pdf` which summarizes results.

The script `generate-data.py` generates the files in the `raw-data`
directory.

The script `generate-figures-and-tables.py` postprocesses the output in the
`raw-data` directory to generate figures and tables in the `out` directory.

The Makefile `makefile.summary` controls the build of the summary PDF, whose
source is `summary.tex`.

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

To see the list of available experiments, run `./generate-data.py --help`.
