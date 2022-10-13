#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:30:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute,compute2
#SBATCH --account=mh0033

module list
source ./conda_act_dwd_pyicon_env.sh
which python

rand=$(cat /dev/urandom | tr -dc 'A-Z' | fold -w 3 | head -n 1)

path_pyicon=`(cd .. && pwd)`"/"
config_file="./config_qp_${rand}.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = False

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_conf_dwd         = True
do_ocean_plots      = True

# --- grid information
gname     = 'R2B4-R2B4'
lev       = 'L40'
gname_atm = 'r2b4_atm_r0013'
lev_atm   = 'L90'

# --- path to interpolation files
path_grid        = '/hpc/uwork/gboeloen/pyICON/grids/'+gname+'/'
path_grid_atm    = '/hpc/uwork/gboeloen/pyICON/grids/'+gname_atm+'/'
path_ckdtree     = path_grid+'/ckdtree/'
path_ckdtree_atm = path_grid_atm+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/hpc/uwork/gboeloen/pyICON/grids/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_grid + gname+'_tgrid.nc'
fpath_tgrid_atm     = path_grid_atm + gname_atm+'_tgrid.nc'
fpath_ref_data_oce  = path_grid + 'ts_phc3.0_annual_icon_grid_0043_R02B04_G_L40.nc'
fpath_ref_data_atm  = path_grid_atm + 'pyicon_prepare_era.nc'
fpath_fx            = path_grid + 'oce_fx.19600102T000000Z.nc'

# --- mappings for ocean
D_variable_container = dict(
  default  = '_oce_3d',
  to       = '_oce_3d',
  so       = '_oce_3d',
  u        = '_oce_3d',
  v        = '_oce_3d',
  massflux = '_oce_3d',
  moc      = '_oce_moc',
  mon      = '_oce_mon',
  ice      = '_oce_2d',
  monthly  = '_oce_2d',
  sqr      = '_oce_2d',
)

# --- mappings for atmosphere
atm_2d      = '_atm_2d_ml'
atm_3d      = '_atm_3d_ml'
atm_mon     = '_atm_mon'

# --- nc output
save_data = False
path_nc = '/scratch/m/m300602/tmp/test_pyicon_output/'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['1630-02-01', '1640-01-01'],
['4450-02-01', '4500-01-01'],
]
ave_freq = 12

# --- what to plot and what not?
# --- not to plot:
#red_list = ['']
# --- to plot:
#green_list = ['']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="cpl01"
path_data="/hpc/uwork/gboeloen/ICON-Seamless/chain/scratch/${run}/output/icon/"
python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='2000-01-01,2011-01-01'

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"
