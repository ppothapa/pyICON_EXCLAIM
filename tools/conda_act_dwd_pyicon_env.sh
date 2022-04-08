#!/bin/bash

# --- conda env settings
# script to working conda environment
path_conda_bin="/hpc/uhome/gboeloen/miniconda3/bin/"
# name of working conda environment
#conda_env="pyicon_env"
#conda_env="pyicon_env2"
#conda_env="pyicon_env3"
#conda_env="pyicon_env4"
#conda_env="pyicon_py37"
conda_env="pyicon_py38"
#conda_env="myenv_py3"

# --- add pyicon to PYTHONPATH
PYICON_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." >/dev/null 2>&1 && pwd )"  #problem: takes directory of source script as base
#PYICON_PATH="$( cd "$(pwd)/.." >/dev/null 2>&1 && pwd )"  # problem: takes directory as base from where this script is called
export PYTHONPATH="${PYICON_PATH}"
# use this if you have a harmless PYTHONPATH which you want to keep
#export PYTHONPATH="${PYICON_PATH}:${PYTHONPATH}"
echo "PYTHONPATH was modified to:" 
echo "${PYTHONPATH}"
echo ""

# --- activate conda environment
echo "Activate conda environment by:"
echo "${path_conda_bin}/activate ${conda_env}"
source ${path_conda_bin}/activate ${conda_env}
echo ""

# --- print some information
conda_path=`which conda`
echo "Active conda:"
echo "${conda_path}"
python_path=`which python`
echo "Active python:"
echo "${python_path}"
