#! /bin/bash
#SBATCH --job-name=pyanim
#SBATCH --time=04:00:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=8
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033
set -x

module list

source /home/mpim/m300602/pyicon/tools/act_pyicon_py39.sh
which python

startdate=`date +%Y-%m-%d\ %H:%M:%S`
mpirun -np 8 python -u examp_animation.py --no_backend
enddate=`date +%Y-%m-%d\ %H:%M:%S`
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

