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
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver_new_oce_output.py"

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
lev       = 'L128_zstar'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
#fpath_ref_data_oce  = path_pool_oce + gnameu+lev+'_initial_state.nc'
fpath_ref_data_oce  = '/work/bm1102/m211054/smtwave/initial/ts_oras5_icon_OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_sills_srtm30_1min_L128smt.nc'
#fpath_fx            = path_pool_oce + gnameu+lev+'_fx.nc'
fpath_fx            = path_grid + gname+'_'+lev+'_fx.nc'

# --- nc file prefixes
oce_def     = '_P1M_3d'
oce_moc     = '_P1M_moc'
oce_mon     = '_P1M_mon'
oce_ice     = '_P1M_2d'
oce_monthly = '_P1M_2d'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['2011-02-01', '2012-01-01'],
#['2012-02-01', '2013-01-01'],
#['2012-02-01', '2013-01-01'],
#['2015-02-01', '2016-01-01'],
#['2018-02-01', '2019-01-01'],
['2019-02-01', '2020-01-01'],
]

red_list = []
# uncomment this if ssh_variance is not there
red_list += ['ssh_variance']
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
red_list += ['dens30w']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="exp.ocean_era51h_zstar_r2b8_21070-SMT"
path_data="/work/bm1102/m211054/smtwave/zstar2/experiments/${run}/outdata/"
#path_data="/work/bm1102/m211054/smtwave/zstar2/experiments/${run}/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

