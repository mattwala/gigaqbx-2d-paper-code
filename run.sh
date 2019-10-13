#!/bin/bash -e

# This script re-runs all experiments.

source install.sh

# Tell PyOpenCL to use POCL
export PYOPENCL_TEST=portable

# Run a simple test
py.test --disable-warnings utils.py

# Run code
OMP_NUM_THREADS=1 ./generate-data.py -x wall-time
nice ./generate-data.py --all --except wall-time
./generate-figures-and-tables.py --all
make -f makefile.summary
