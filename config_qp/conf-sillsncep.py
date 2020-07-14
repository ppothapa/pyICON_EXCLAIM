# --- runname and path to data
runname   = ''
run       = 'sillsncep'
path_data = f'/mnt/lustre01/work/mh0033/m214056/RESOLUTION_EXPS/NEW_RUN_PAPER2019/icon-oes-1.3.01/experiments/{run}/'

# --- path to quickplots
path_quickplots = '../../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = False
do_ocean_plots      = True

# --- grid information
gname     = 'smt20km'
lev       = 'L128'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data
fpath_tgrid         = 'auto'
fpath_ref_data_oce  = '/mnt/lustre01/work/mh0033/m214056/RESOLUTION_EXPS/startfile_Helmuth/ts_phc3.0_annual_icon_OceanOnly_Global_NAtalntic_0020km_modified_srtm30_1min.nc_L128.nc'
fpath_fx            = 'auto'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
['1950-02-01', '1952-01-01'],
]
