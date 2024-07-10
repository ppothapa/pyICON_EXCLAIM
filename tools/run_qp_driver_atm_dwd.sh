#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:30:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=d121



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
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_conf_dwd         = True
do_djf              = False
do_jja              = False
do_ocean_plots      = False

# --- grid information
gname     = 'R2B4-R2B4'
lev       = 'L40'
gname_atm = 'r2b4_atm_r0013'
lev_atm   = 'L90'

# --- path to interpolation files
#  Here is the DWD Path
#path_grid        = '/hpc/uwork/icon-sml/pyICON/grids/'+gname+'/'
#path_grid_atm    = '/hpc/uwork/icon-sml/pyICON/grids/'+gname_atm+'/'
#path_ckdtree     = path_grid+'/ckdtree/'
#path_ckdtree_atm = path_grid_atm+'/ckdtree/'
#End of DWD Path

# PPK Comment, as far as I can see, these are all Ocean files- and is difficult to understand them.


path_grid        = '/project/d121/ppothapa/data_for_pyicon/r2b6_oce_r0004/'
path_grid_atm    = '/project/d121/ppothapa/data_for_pyicon/r2b4_atm_r0013/'
path_ckdtree     = '/project/d121/ppothapa/data_for_pyicon/r2b6_oce_r0004/ckdtree/'
path_ckdtree_atm = '/project/d121/ppothapa/data_for_pyicon/r2b4_atm_r0013/ckdtree/'



# --- grid files and reference data
#path_pool_oce       = '/hpc/uwork/icon-sml/pyICON/grids/'
path_pool_oce       = '/pool/data/ICON/grids/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid                 = '/scratch/snx3000/ppothapa/for_praveen/0013/icon_grid_0013_R02B04_G.nc'
fpath_tgrid_atm             = '/scratch/snx3000/ppothapa/for_praveen/0013/icon_grid_0013_R02B04_G.nc'
fpath_ref_data_oce          = path_grid + 'ts_phc3.0_annual_icon_grid_0043_R02B04_G_L40.nc'
fpath_ref_data_atm          = path_grid_atm + 'era5_pyicon_2001-2010_1.5x1.5deg.nc'
fpath_ref_data_atm          = '/project/d121/ppothapa/data_for_pyicon/era/pyicon_prepare_era.nc'
fpath_ref_data_atm_djf      = path_grid_atm + 'era5_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_jja      = path_grid_atm + 'era5_pyicon_2001-2010_jja_1.5x1.5deg.nc'
fpath_ref_data_atm_rad      = path_grid_atm + 'ceres_pyicon_2001-2010_1.5x1.5deg.nc'
fpath_ref_data_atm_rad_djf  = path_grid_atm + 'ceres_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_rad_jja  = path_grid_atm + 'ceres_pyicon_2001-2010_jja_1.5x1.5deg.nc'
fpath_ref_data_atm_prec     = path_grid_atm + 'gpm_pyicon_2001-2010_1.5x1.5deg.nc'
fpath_ref_data_atm_prec_djf = path_grid_atm + 'gpm_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_prec_jja = path_grid_atm + 'gpm_pyicon_2001-2010_jja_1.5x1.5deg.nc'
fpath_fx                    = path_grid + 'oce_fx.19600102T000000Z.nc'

# --- nc file prefixes ocean
oce_def     = '_oce_def'
oce_moc     = '_oce_moc'
oce_mon     = '_oce_mon'
oce_ice     = '_oce_ice'
oce_monthly = '_oce_dbg'

# --- nc file prefixes atmosphere
atm_2d      = '_atm_2d_ml'
atm_3d      = '_atm_3d_ml'
atm_mon     = '_atm_mon'

# --- nc output
save_data = False
path_nc = '/scratch/m/m300602/tmp/test_pyicon_output/'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['1630-02-01', '1640-01-01'],
['2004-03-01', '2004-05-01'],
]

ave_freq = 12

#output_freq = 'daily'


# --- decide if time-series (ts) plots are plotted for all the 
#     available data or only for the intervall defined by tave_int

use_tave_int_for_ts = False

# --- what to plot and what not?
# --- not to plot:
#red_list = ['']
# --- to plot:
#green_list = ['atm_t2m']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

run="exclaim_uncoupled_R02B04L90_test"
#path_data="/hpc/uwork/gboeloen/ICON-Seamless/chain/scratch/${run}/output/icon/"
path_data="/scratch/snx3000/ppothapa/icon-dsl_v0.3.0-rc/experiments/exclaim_uncoupled_R02B04L90_test/"
#python -W ignore -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='2004-02-15,2004-05-15'

python -W ignore -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run --tave_int='2004-03-01,2005-01-01' 


enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

