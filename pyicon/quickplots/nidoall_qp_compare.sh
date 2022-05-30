#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:20:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033

module list
source ~/pyicon/tools/conda_act_mistral_pyicon_env.sh
which python

startdate=`date +%Y-%m-%d\ %H:%M:%S`

srun --exclusive -n 1 -c 1 python -u qp_compare.py ../../config_qp_compare/config_r2b6_idemix_levante.py &
wait

enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"
