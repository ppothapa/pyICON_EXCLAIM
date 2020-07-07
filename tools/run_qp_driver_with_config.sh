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

run="slo1284"
path_pyicon=`(cd .. && pwd)`"/"
config_file="${path_pyicon}/config_qp/config_qp_${run}.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
runname   = ''
run       = '${run}'

gname     = 'r2b6'
lev       = 'L64'
gname_atm = 'r2b4a'
lev_atm   = 'L84'

do_atmosphere_plots = True

tstep     = '????????????????'  # use this line for all data

path_data     = '/work/mh0469/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/'+run+'/'
path_grid     = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_grid_atm = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree  = 'auto'

fpath_initial_state = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_08/icon-oes/experiments/nib0004/initial_state.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = '_oce_def'
oce_moc = '_oce_moc'
oce_mon = '_oce_mon'
oce_monthly = '_oce_dbg'

atm_2d  = '_atm_2d_ml'
atm_3d  = '_atm_3d_ml'
atm_mon = '_atm_mon'

tave_ints = [
['1630-02-01', '1640-01-01'],
]
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`
#python -u ${qp_driver} --batch=True ${config_file}
python -u ${qp_driver} --batch=True ${config_file} --run="slo1284" --tave_int='1570-02-01,1580-01-01'
enddate=`date +%Y-%m-%d\ %H:%M:%S`

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

