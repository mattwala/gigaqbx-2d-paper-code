#!/bin/bash -e

# This script installs software and sets up the Conda environment.

# Find platform
if [ "$(uname)" = Darwin ] && [ "$(uname -m)" = x86_64 ]
then
    MINICONDA_INSTALL_SH=Miniconda3-4.7.10-MacOSX-x86_64.sh
    SPEC_FILE=env/osx-64-spec.txt
elif [ "$(uname)" = Linux ] && [ "$(uname -m)" = x86_64 ]
then
    MINICONDA_INSTALL_SH=Miniconda3-4.7.10-Linux-x86_64.sh
    SPEC_FILE=env/linux-64-spec.txt
else
    echo Unsupported platform.
    exit 1
fi

MINICONDA_INSTALL_DIR=.miniconda3

if [ -d $MINICONDA_INSTALL_DIR ] && [ -d src ]
then
    # Re-activate environment
    source "$MINICONDA_INSTALL_DIR/bin/activate" inteq
else
    # Install miniconda
    curl -O https://repo.continuum.io/miniconda/$MINICONDA_INSTALL_SH
    bash $MINICONDA_INSTALL_SH -b -p "$MINICONDA_INSTALL_DIR"

    # Install software
    PATH="$MINICONDA_INSTALL_DIR/bin:$PATH" conda create --yes --name inteq --file ${SPEC_FILE}
    source "$MINICONDA_INSTALL_DIR/bin/activate" inteq
    pip install -r env/requirements.txt
fi
