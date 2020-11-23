#! /bin/bash
#SBATCH --job-name=pysmt
#SBATCH --time=08:00:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#####SBATCH --account=bm0371
#####SBATCH --account=mh0033
#SBATCH --account=bm1102

#module load python/2.7.12
module list

#source /home/mpim/m300602/bin/myactcondenv.sh
source /home/mpim/m300602/pyicon/tools/conda_act_mistral_pyicon_env.sh
which python

startdate=`date +%Y-%m-%d\ %H:%M:%S`
python -u config_ckdtree_r2b11_oce.py --slurm
enddate=`date +%Y-%m-%d\ %H:%M:%S`
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

