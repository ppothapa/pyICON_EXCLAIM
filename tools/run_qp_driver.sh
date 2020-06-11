#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:30:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033

module list

source /home/mpim/m300602/pyicon/tools/conda_act_mistral_pyicon_env.sh
which python

path_pyicon=`(cd .. && pwd)`"/"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"
#config_file="${path_pyicon}config_qp/conf-slo1284.py"
config_file="${path_pyicon}config_qp/conf-exp.ocean_omip_long_tke_r2b6_20133-CIQ.py"

startdate=`date +%Y-%m-%d\ %H:%M:%S`
python -u ${qp_driver} --batch=True ${config_file}
enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

