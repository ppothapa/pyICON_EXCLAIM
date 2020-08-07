runname   = ''
run       = 'slo1284'

gname     = 'r2b6'
lev       = 'L64'
gname_atm = 'r2b4a'
lev_atm   = 'L84'

do_atmosphere_plots = True

#tstep    = '17800101T000000Z'
tstep     = '????????????????'  # use this line for all data

#path_data     = '/work/mh0287/users/juergen/icon-oes/experiments/'+run+'/'
path_data     = '/work/mh0469/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/'+run+'/'
path_grid     = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_grid_atm = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'
path_ckdtree  = 'auto'

fpath_ref_data_oce = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_08/icon-oes/experiments/nib0004/initial_state.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = '_oce_def'
oce_moc = '_oce_moc'
oce_mon = '_oce_mon'
oce_ice = '_oce_ice'
oce_monthly = '_oce_dbg'

atm_2d  = '_atm_2d_ml'
atm_3d  = '_atm_3d_ml'
atm_mon = '_atm_mon'

tave_ints = [
#['1480-02-01', '1490-01-01'],
#['1580-02-01', '1590-01-01'],
['1630-02-01', '1640-01-01'],
#['1780-02-01', '1790-01-01'],
]
