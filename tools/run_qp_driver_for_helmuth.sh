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
config_file="${path_pyicon}config_qp/conf-hel20134-TOP.py"
path_qps="${path_pyicon}all_qps/"

startdate=`date +%Y-%m-%d\ %H:%M:%S`
# --- hel20134-TOP
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-TOP" --tave_int='1590-02-01,1600-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-TOP" --tave_int='1690-02-01,1700-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-TOP" --tave_int='1790-02-01,1800-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-TOP" --tave_int='1890-02-01,1900-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-TOP" --tave_int='1990-02-01,2000-01-01'
# --- hel20134-STR
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-STR" --tave_int='1590-02-01,1600-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-STR" --tave_int='1690-02-01,1700-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-STR" --tave_int='1790-02-01,1800-01-01'
python -u ${qp_driver} --batch=True --path_quickplots=${path_qps} ${config_file} --run="hel20134-STR" --tave_int='1890-02-01,1900-01-01'
enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

