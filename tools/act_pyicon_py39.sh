#!/bin/bash

t1=`date +%Y-%m-%d_%H-%M-%S`

# --- conda env settings
# script to working conda environment
path_conda_bin="/home/m/m300602/miniconda3/bin/"
# name of working conda environment
#conda_env="/home/m/m300602/miniconda3/envs/pyicon_py39"
conda_env="/home/m/m300602/.conda/envs/pyicon_py39_cartopy19"

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

t2=`date +%Y-%m-%d_%H-%M-%S`

echo "All done!"
echo "Start ${t1}"
echo "End   ${t2}"
