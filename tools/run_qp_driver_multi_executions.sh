#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:20:00
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
config_file="${path_pyicon}config_qp/conf-ruby0_test.py"
path_qps="${path_pyicon}all_qps_2/"

startdate=`date +%Y-%m-%d\ %H:%M:%S`
# --- slo1284
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="slo1284" --tave_int='1570-02-01,1580-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="slo1284" --tave_int='1600-02-01,1610-01-01'
# --- slo1283
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="slo1283" --tave_int='1510-02-01,1520-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="slo1283" --tave_int='1570-02-01,1580-01-01'
enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

