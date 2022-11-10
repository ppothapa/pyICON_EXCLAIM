#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=00:30:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --account=mh0033

#cd /home/m/m211054/pyicon/tools
#cd /home/m/m211054/pyicon_helmuth_levante/tools
cd /home/m/m300602/pyicon/tools/

module list
#source ./conda_act_mistral_pyicon_env.sh
source /home/m/m300602/pyicon/tools/act_pyicon_py39.sh 
which python

#rand=$(cat /dev/urandom | tr -dc 'A-Z' | fold -w 3 | head -n 1)
rand="easyms22"

path_pyicon=`(cd .. && pwd)`"/"
config_file="./config_qp_${rand}.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
run = "sfx0080"
path_data = f"/work/mh0287/m300879/icon-les/experiments/{run}/outdata/"

# --- path to quickplots
path_quickplots = '../../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_ocean_plots      = True
do_hamocc_plots     = True

# --- grid information
gname     = 'r2b4_oce_r0004'
lev       = 'L40'
gname_atm = 'r2b3_atm_r0030'
lev_atm   = 'L47'

# --- path to interpolation files
path_grid        = '/work/mh0033/m300602/icon/grids/'+gname+'/'
path_grid_atm    = '/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree     = path_grid+'/ckdtree/'
path_ckdtree_atm = path_grid_atm+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/icon_grid_0036_R02B04_O/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_tgrid_atm     = '/pool/data/ICON/grids/public/mpim/0030/icon_grid_0030_R02B03_G.nc'
fpath_ref_data_oce  = '/pool/data/ICON/oes/input/r0004/icon_grid_0036_R02B04_O/R2B4L40_initial_state.nc'
#fpath_ref_data_atm  = '/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_ref_data_atm  = '/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_fx            = path_grid + gname+'_'+lev+'_fx.nc'

# --- nc file prefixes
D_variable_container = dict(
  default  = '_oce_P1M_3d',
  to       = '_oce_P1M_3d',
  so       = '_oce_P1M_3d',
  u        = '_oce_P1M_3d',
  v        = '_oce_P1M_3d',
  massflux = '_oce_P1M_3d',
  moc      = '_oce_P1M_moc',
  mon      = '_oce_P1M_mon',
  ice      = '_oce_P1M_2d',
  monthly  = '_oce_P1M_2d',
  sqr      = '_oce_P1M_sqr',
)

ham_inv = '_hamocc'
ham_2d = '_hamocc_2d_tendencies'
ham_mon = '_hamocc_monitor'

atm_2d = '_atm_2d_ml'
atm_3d = '_atm_3d_ml'
atm_mon = '_atm_mon'

tstep_ts  = '????????'
tstep     = '29??????'

time_at_end_of_interval = False

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['2011-02-01', '2012-01-01'],
#['2012-02-01', '2013-01-01'],
#['2013-02-01', '2014-01-01'],
#['2014-02-01', '2015-01-01'],
#['2015-02-01', '2016-01-01'],
#['2016-02-01', '2017-01-01'],
#['2017-02-01', '2018-01-01'],
#['2018-02-01', '2019-01-01'],
#['2019-02-01', '2020-01-01'],
#['2900-01-01', '2909-12-31'],
['2900-01-01', '2949-12-31'],
]

red_list = []
# from dwd; does not work for MPI-ICON
red_list += ['ts_pme_gmean']
# uncomment this if ssh_variance is not there
#red_list += ['ssh']
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

run="sfx0080"
path_data="/work/mh0287/m300879/icon-les/experiments/sfx0080/outdata/"

python -u ${qp_driver} --batch=True ${config_file} --path_data=$path_data --run=$run

enddate=`date +%Y-%m-%d\ %H:%M:%S`

#rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"


