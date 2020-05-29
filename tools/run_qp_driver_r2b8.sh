#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=02:00:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033

module list

source /home/mpim/m300602/pyicon/tools/conda_act_mistral_pyicon_env.sh
which python

path_pyicon="/mnt/lustre01/pf/zmaw/m300602/pyicon/"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"
config_file="${path_pyicon}config_qp/conf-exp.ocean_omip_long_tke_r2b8_20134-WWG.py"
path_qps="${path_pyicon}all_qps/"

startdate=`date +%Y-%m-%d\ %H:%M:%S`
# --- exp.ocean_omip_long_tke_r2b8_20134-WWG
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="exp.ocean_omip_long_tke_r2b8_20134-WWG" --tave_int='1932-02-01,1933-01-01'
enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

