#!/bin/bash
#SBATCH --job-name=pyicon_qp
#SBATCH --time=08:00:00
#SBATCH --output=log.o-%j.out
#SBATCH --error=log.o-%j.out
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --account=mh0033

module list
source ./act_pyicon_py39.sh
which python

rand=$(cat /dev/urandom | tr -dc 'A-Z' | fold -w 3 | head -n 1)

path_pyicon=`(cd .. && pwd)`"/"
config_file="./config_qp_${rand}.py"
qp_driver="${path_pyicon}pyicon/quickplots/qp_driver.py"

cat > ${config_file} << %eof%
run = 'nib2201'
path_data = f'/work/mh0033/m300602/proj_vmix/icon/icon_22/icon-oes-zstar3/experiments/{run}/'

# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = False
do_ocean_plots      = True

# --- grid information
gname     = 'r2b6_oce_r0004'
lev       = 'L64'

# --- grid files and reference data
path_grid           = f'/work/mh0033/m300602/icon/grids/{gname}/'
path_ckdtree        = f'{path_grid}/ckdtree/'
fpath_tgrid         = f'{path_grid}/{gname}_tgrid.nc'
fpath_fx            = f'{path_grid}/{gname}_{lev}_fx.nc'
fpath_ref_data_oce  = f'{path_data}/initial_state.nc'

# --- mappings
D_variable_container = dict(
  default  = '_P1Y_3d',
  to       = '_P1Y_3d',
  so       = '_P1Y_3d',
  u        = '_P1Y_3d',
  v        = '_P1Y_3d',
  massflux = '_P1Y_3d',
  moc      = '_P1M_moc',
  mon      = '_P1M_mon',
  ice      = '_P1M_2d',
  monthly  = '_P1M_2d',
  sqr      = '_P1M_sqr',
)

# --- time average information (can be overwritten by qp_driver call)
tstep = "*"
tave_ints = [
['2090-02-01', '2100-01-01'],
]

red_list = []
# uncomment this if ssh_variance is not there
#red_list += ['ssh']
red_list += ['ssh_variance']
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
#red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
#red_list += ['dens30w']
%eof%

# --- start qp_driver
startdate=`date +%Y-%m-%d\ %H:%M:%S`

python -u ${qp_driver} --batch=True ${config_file} 

enddate=`date +%Y-%m-%d\ %H:%M:%S`

rm ${config_file}

echo "--------------------------------------------------------------------------------"
echo "Started at ${startdate}"
echo "Ended at   ${enddate}"

