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
# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_ocean_plots      = True

# --- grid information
gname     = 'r2b6'
lev       = 'L64'
gname_atm = 'r2b4a'
lev_atm   = 'L84'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_grid_atm    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree     = path_grid+'/ckdtree/'
path_ckdtree_atm = path_grid_atm+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/'
gnameu = gname.upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_tgrid_atm     = '/pool/data/ICON/grids/public/mpim/0013/icon_grid_0013_R02B04_G.nc'
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

run="slo1284"
path_data="/work/mh0469/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/${run}/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1570-02-01,1580-01-01'
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1670-02-01,1680-01-01'

#run="dap1234"
#path_data="/different/path_data/${run}/"
#python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1570-02-01,1580-01-01'
#python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1670-02-01,1680-01-01'

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

