run = 'slo1325'
runname = ''
path_data = '/work/mh0287/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/slo1325/'

# --- path to quickplots
path_quickplots = '/mnt/lustre01/pf/zmaw/m300602/pyicon/all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- decide which set of plots to do
do_ocean_plots      = True
do_atmosphere_plots = True
do_hamocc_plots     = False

# --- grid information
gname     = 'r2b6_oce_r0004'
lev       = 'L64'
gname_atm = 'r2b4_atm_r0013'
lev_atm   = 'L84'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b6_oce_r0004/'
path_grid_atm    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b4_atm_r0013/'
path_ckdtree     = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b6_oce_r0004//ckdtree/'
path_ckdtree_atm = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b4_atm_r0013//ckdtree/'

# --- grid files and reference data
fpath_tgrid         = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/R2B6_ocean-grid.nc'
fpath_tgrid_atm     = '/pool/data/ICON/grids/public/mpim/0013/icon_grid_0013_R02B04_G.nc'
fpath_ref_data_oce  = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/R2B6L64_initial_state.nc'
fpath_ref_data_atm  = '/mnt/lustre01/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_fx            = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/R2B6L64_fx.nc'

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
tave_ints = [['4840-02-01', '4850-01-01']]
# --- decide which data files to take for time series plots
tstep     = '????????????????'
# --- set this to 12 for yearly averages in timeseries plots, set to 0 for no averaging
ave_freq = 12

# --- xarray usage
xr_chunks = None
load_xarray_dset = False
save_data = True
path_nc = '/home/mpim/m300602/work/tmp/test_pyicon_output/'

# --- information for re-gridding
sec_name_30w   = '30W_300pts'
rgrid_name     = 'global_0.3'
rgrid_name_atm = 'global_1.0_era'

verbose = False
do_write_final_config = False

# --- list containing figures which should not be plotted
red_list = []
# --- list containing figures which should be plotted 
# (if empty all figures will be plotted)
green_list = []

