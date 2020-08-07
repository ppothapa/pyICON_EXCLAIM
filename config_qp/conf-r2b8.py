run="exp.ocean_ncep6h_r2b8_hel20160-NCP"
path_data=f"/work/mh0033/m211054/projects/icon/icon-oes-1.3.01/experiments/{run}/outdata"
verbose = True

# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = False

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = False
do_ocean_plots      = True

# --- grid information
gname     = 'r2b8'
lev       = 'L128'
gname_atm = 'r2b4a'
lev_atm   = 'L84'

# --- path to interpolation files
path_grid        = '/mnt/lustre01/work/mh0033/m211054/projects/icon/grids/'+gname+'/'
path_grid_atm    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree     = path_grid+'/ckdtree/'
path_ckdtree_atm = path_grid_atm+'/ckdtree/'

# --- grid files and reference data
path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/'
gnameu = gname.upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
fpath_tgrid_atm     = '/pool/data/ICON/grids/public/mpim/0013/icon_grid_0013_R02B04_G.nc'
fpath_ref_data_oce  = path_pool_oce + gnameu+lev+'_initial_state.nc'
fpath_ref_data_atm  = '/mnt/lustre01/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'
fpath_fx            = path_pool_oce + gnameu+lev+'_fx.nc'

# --- nc file prefixes
oce_def     = ''
oce_moc     = '_MOC'
oce_mon     = '_oceanMonitor'
oce_ice     = ''
oce_monthly = ''

atm_2d      = '_atm_2d_ml'
atm_3d      = '_atm_3d_ml'
atm_mon     = '_atm_mon'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
['1948-02-01', '1949-01-01'],
]
