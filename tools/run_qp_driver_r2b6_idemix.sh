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
gname     = 'r2b6_oce_r0003'
lev       = 'L64'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data
fpath_tgrid         = path_grid + gname + '_tgrid.nc'
fpath_fx            = path_grid + gname+'_'+lev+'_fx.nc'
pd = "/home/mpim/m300602/work/proj_vmix/icon/icon_12/icon-oes/experiments/nib0004/"
fpath_ref_data_oce  = pd + 'initial_state.nc'

# --- nc file prefixes
oce_def     = ''
oce_moc     = '_MOC'
oce_mon     = '_oceanMonitor'
oce_ice     = ''
oce_monthly = ''

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['2011-02-01', '2012-01-01'],
#['2012-02-01', '2013-01-01'],
#['2012-02-01', '2013-01-01'],
#['2015-02-01', '2016-01-01'],
#['2018-02-01', '2019-01-01'],
['2490-02-01', '2500-01-01'],
]

red_list = []
# uncomment this if ssh_variance is not there
#red_list += ['ssh_variance']
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
#red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
#red_list += ['dens30w']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="nib0004"
path_data="/home/mpim/m300602/work/proj_vmix/icon/icon_12/icon-oes/experiments/${run}/"
#path_data="/work/bm1102/m211054/smtwave/zstar2/experiments/${run}/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

