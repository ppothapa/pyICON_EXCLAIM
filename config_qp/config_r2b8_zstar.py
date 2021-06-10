run = 'exp.ocean_era51h_zstar_r2b8_21043-SMT'
runname = ''
path_data = '/work/bm1102/m211054/smtwave/zstar2/experiments/exp.ocean_era51h_zstar_r2b8_21043-SMT/outdata/'

# --- path to quickplots
path_quickplots = '/mnt/lustre01/pf/zmaw/m300602/pyicon/all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- decide which set of plots to do
do_ocean_plots      = True
do_atmosphere_plots = False
do_hamocc_plots     = False

# --- grid information
gname     = 'r2b8_oce_r0004'
lev       = 'L128_zstar'
gname_atm = ''
lev_atm   = ''

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b8_oce_r0004/'
path_grid_atm    = ''
path_ckdtree     = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b8_oce_r0004//ckdtree/'
path_ckdtree_atm = 'auto'

# --- grid files and reference data
fpath_tgrid         = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/R2B8_ocean-grid.nc'
fpath_tgrid_atm     = ''
#fpath_ref_data_oce  = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/R2B8L128_zstar_initial_state.nc'
fpath_ref_data_oce  = '/work/bm1102/m211054/smtwave/initial/ts_oras5_icon_OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_sills_srtm30_1min_L128smt.nc'
fpath_ref_data_atm  = '/mnt/lustre01/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_fx            = '/mnt/lustre01/work/mh0033/m300602/icon/grids/r2b8_oce_r0004/r2b8_oce_r0004_L128_zstar_fx.nc'

# --- nc file prefixes
oce_def     = '_P1M_3d'
oce_moc     = '_P1M_moc'
oce_mon     = '_P1M_mon'
oce_ice     = '_P1M_2d'
oce_monthly = '_P1M_2d'

atm_2d      = ''
atm_3d      = ''
atm_mon     = ''

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [['2011-02-01', '2012-01-01']]
# --- decide which data files to take for time series plots
tstep     = '????????????????'
# --- set this to 12 for yearly averages in timeseries plots, set to 0 for no averaging
ave_freq = 12

# --- xarray usage
xr_chunks = None
load_xarray_dset = False
save_data = False
path_nc = './'

# --- information for re-gridding
sec_name_30w   = '30W_300pts'
rgrid_name     = 'global_0.3'
rgrid_name_atm = 'global_1.0_era'

verbose = False
do_write_final_config = False

# --- list containing figures which should not be plotted
red_list = ['ssh_variance', 'bstr', 'arctic_budgets', 'passage_transports', 'arctic_budgets', 'dens30w']
# --- list containing figures which should be plotted 
# (if empty all figures will be plotted)
green_list = []
