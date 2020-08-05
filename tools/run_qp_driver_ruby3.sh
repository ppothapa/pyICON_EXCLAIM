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

rand=$(cat /dev/urandom | tr -dc 'A-Z' | fold -w 3 | head -n 1)

path_pyicon=`(cd .. && pwd)`"/"
config_file="./config_qp_${rand}.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_ocean_plots      = True

# --- grid information
gname     = 'r2b4_oce_r0004'
lev       = 'L40'
gname_atm = 'r2b3_atm_r0030'
lev_atm   = 'L47'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_grid_atm    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree     = path_grid+'/ckdtree/'
path_ckdtree_atm = path_grid_atm+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/icon_grid_0036_R02B04_O/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_tgrid_atm     = '/pool/data/ICON/grids/public/mpim/0030/icon_grid_0030_R02B03_G.nc'
fpath_ref_data_oce  = path_pool_oce + gnameu+lev+'_initial_state.nc'
fpath_ref_data_atm  = '/mnt/lustre01/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_fx            = path_pool_oce + gnameu+lev+'_fx.nc'

# --- nc file prefixes
oce_def     = '_oce_def'
oce_moc     = '_oce_moc'
oce_mon     = '_oce_mon'
oce_ice     = '_oce_ice'
oce_monthly = '_oce_dbg'

atm_2d      = '_atm_2d_ml'
atm_3d      = '_atm_3d_ml'
atm_mon     = '_atm_mon'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
['1630-02-01', '1640-01-01'],
]
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="ler0613"
path_data="/work/mh0727/m300732/icon-oes-switch2/experiments/${run}/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1110-02-01,1130-01-01'

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

