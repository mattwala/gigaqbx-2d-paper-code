#!/bin/bash -e

if [ "$(uname)" = "Darwin" ]; then
  PLATFORM=MacOSX
  SPEC_FILE=env/osx-64-spec.txt
else
  PLATFORM=Linux
  SPEC_FILE=env/linux-64-spec.txt
fi

# Install miniconda

MINICONDA_VERSION=3
MINICONDA_INSTALL_DIR=.miniconda${MINICONDA_VERSION}
MINICONDA_INSTALL_SH=Miniconda${MINICONDA_VERSION}-latest-${PLATFORM}-x86_64.sh
curl -O https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_SH
rm -Rf "$MINICONDA_INSTALL_DIR"
bash $MINICONDA_INSTALL_SH -b -p "$MINICONDA_INSTALL_DIR"

# Install software

PATH="$MINICONDA_INSTALL_DIR/bin:$PATH" conda create --name inteq --file ${SPEC_FILE}.txt
source "$MINICONDA_INSTALL_DIR/bin/activate" inteq
pip install -r env/requirements.txt

# Tell PyOpenCL to use POCL

export PYOPENCL_TEST=portable

# Run a simple test

py.test --disable-warnings utils.py

# Run code

OMP_NUM_THREADS=1 ./generate-data.py -x wall-time
nice -n 1 ./generate-data.py --all --except wall-time
./generate-figures-and-tables.py --all
