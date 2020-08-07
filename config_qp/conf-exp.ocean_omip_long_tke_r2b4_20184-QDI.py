runname   = ''
run       = 'exp.ocean_omip_long_tke_r2b4_20184-QDI'

gname     = 'r2b4'
lev       = 'L40'
gname_atm = 'r2b4a'
lev_atm   = 'L84'

do_atmosphere_plots = False

#tstep    = '17800101T000000Z'
tstep     = '????????????????'  # use this line for all data


#path_data     = '/work/mh0287/users/juergen/icon-oes/experiments/'+run+'/'
#path_data     = '/work/mh0469/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/'+run+'/'
path_data     = f'/work/mh0033/m211054/projects/icon/input/master/experiments/{run}/'
path_grid     = '/work/mh0033/m211054/projects/icon/grids/'+gname+'/'
path_grid_atm = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree  = 'auto'

fpath_ref_data_oce = '/pool/data/ICON/oes/input/r0004/icon_grid_0036_R02B04_O/R2B4L40_initial_state.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = ''
oce_moc = '_MOC'
oce_mon = '_oceanMonitor'
oce_monthly = ''

atm_2d  = '_atm_2d_ml'
atm_3d  = '_atm_3d_ml'
atm_mon = '_atm_mon'

tave_ints = [
#['2040-02-01', '2050-01-01'],
#['2090-02-01', '2100-01-01'],
#['2140-02-01', '2150-01-01'],
['2190-02-01', '2200-01-01'],
]
