run = 'exp.ocean_era51h_zstar_r2b6_22144-ERA'
runname = ''
#path_data = f'/work/mh0287/m211054/nextgems/nextgems_cycle2/experiments/{run}/outdata/'
path_data = f'/scratch/m/m211054/exp.ocean_era51h_zstar_r2b6_22144-ERA/run_*/'

# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = False
do_ocean_plots      = True

# --- grid information
gname     = 'r2b6_oce_r0004'
lev       = 'L128_zstar'

# --- path to interpolation files
path_grid        = '/work/mh0033/m211054/projects/icon/grids/'+gname+'/'
path_ckdtree     = path_grid+'/ckdtree/'

# --- grid files and reference data

path_pool_oce       = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid         = path_pool_oce + gnameu+'_ocean-grid.nc'
#fpath_ref_data_oce  = path_pool_oce + gnameu+lev+'_initial_state.nc'
fpath_ref_data_oce  = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/ZSTAR/ts_phc3.0_annual_icon_OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_sills_srtm30_1min_L128smt.nc'

#fpath_ref_data_oce  = '/work/bm1102/m211054/smtwave/initial/ts_oras5_icon_OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_sills_srtm30_1min_L128smt.nc'
#fpath_fx            = path_pool_oce + gnameu+lev+'_fx.nc'
#fpath_fx            = path_grid + gname+'_'+lev+'_fx.nc'
fpath_fx            = '/pool/data/ICON/oes/input/r0004/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/ZSTAR/R2B6L128_fx.nc'

# --- nc file prefixes
# --- nc file prefixes
#oce_def     = '_P1M_to'
#oce_to     = '_P1M_to'
#oce_so     = '_P1M_so'
#oce_u     = '_P1M_u'
#oce_v     = '_P1M_v'
#oce_massflux   = '_P1M_mass_flux'
#oce_moc     = '_P1M_moc'
#oce_mon     = '_P1M_mon'
#oce_ice     = '_P1M_2d'
#oce_monthly = '_P1M_2d'
#oce_sqr     = '_P1M_sqr'

# --- time average information (can be overwritten by qp_driver call)
#tstep = "202[0-1]*"
tstep = "*"
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
['2018-01-01', '2018-03-01'],
]

red_list = []
# uncomment this if ssh_variance is not there
#red_list += ['ssh']
red_list += ['ssh_variance']
# uncomment this to omit plots which require loading 3D mass_flux
#red_list += ['bstr', 'arctic_budgets', 'passage_transports']
# uncomment this to omit plots which require loading 3D u, v
red_list += ['arctic_budgets']
# uncomment this to omit plots which require loading 3D density
red_list += ['dens30w']
red_list += [
  'ice_concentration_nh', 'ice_thickness_nh', 'snow_thickness_nh', 
  'ice_concentration_sh', 'ice_thickness_sh', 'snow_thickness_sh',
]

