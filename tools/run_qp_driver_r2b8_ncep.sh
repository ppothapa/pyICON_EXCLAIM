#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=08:00:00
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
config_file="./config_qp_${rand}.py"
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
gname     = 'r2b8_oce_r0004'
lev       = 'L128'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_ref_data_oce  = path_pool_oce + gnameu+lev+'_initial_state.nc'
fpath_fx            = path_pool_oce + gnameu+lev+'_fx.nc'

# --- nc file prefixes
oce_def     = ''
oce_moc     = '_MOC'
oce_mon     = '_oceanMonitor'
oce_ice     = ''
oce_monthly = ''

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
['1950-02-01', '1951-01-01'],
['1960-02-01', '1961-01-01'],
['1970-02-01', '1971-01-01'],
['1980-02-01', '1981-01-01'],
['1990-02-01', '1991-01-01'],
['2000-02-01', '2000-01-01'],
['2010-02-01', '2010-01-01'],
['2019-02-01', '2019-01-01'],
]

red_list = []
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
red_list += ['dens30w']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="exp.ocean_ncep6h_r2b8_hel20160-NCP"
path_data="/work/mh0033/m211054/projects/icon/icon-oes-1.3.01/experiments/${run}/outdata/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

