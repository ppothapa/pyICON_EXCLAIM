#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=02:00:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033

module list
source ./conda_act_mistral_pyicon_env.sh
which python

rand=$(cat /dev/urandom | tr -dc 'A-Z' | fold -w 3 | head -n 1)

path_pyicon=`(cd .. && pwd)`"/"
config_file="./config_qp_snap.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = False
do_ocean_plots      = True

# --- grid information
gname     = 'r2b6_oce_r0004'
lev       = 'L128'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_ref_data_oce  = '/home/mpim/m300029/zstar/ocean_omip_long_128_30184-QCY/initial_state.nc'
fpath_fx            = '/home/mpim/m300029/zstar/ocean_omip_long_128_30184-QCY/ocean_omip_long_128_30184-QCY_fx.nc'

# --- nc file prefixes
oce_def     = ''
oce_moc     = '_MOC'
oce_mon     = '_oceanMonitor'
oce_ice     = ''
oce_monthly = ''

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
['1630-02-01', '1640-01-01'],
]

red_list = []
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
#red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
#red_list += ['dens30w']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run='ocean_omip_long_128_30184-QCY'
path_data="/home/mpim/m300029/zstar/${run}/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='1930-02-01,1930-03-01'

enddate=`date +%Y-%m-%d\ %H:%M:%S`

#rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

