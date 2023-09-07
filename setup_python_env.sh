#!/bin/sh
#
# A shell script to setup a virtual environment. Run this script from the top
# directory of the project:
#
# $ ./setup_python_env.sh

set -x
set -e

if [ -d "env" ]; then
    echo "virtual environment exists"
    exit 0
fi

~/download/Python-3.9.17/python -m venv env

. ./env/bin/activate

pip install -U pip
pip install \
    black \
    gurobipy \
    isort \
    matplotlib \
    numpy \
    pandas \
    pyomo \
    scikit-learn \
    seaborn \
    torch
