runname = ''
run     = 'jkr0043'
gname   = 'r2b6'
lev     = 'L64'
#tstep   = '17800101T000000Z'
tstep   = '????????????????'  # use this line for all data

path_data    = '/work/mh0287/users/juergen/icon-oes/experiments/'+run+'/'
path_grid    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree = 'auto'

fpath_initial_state = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_08/icon-oes/experiments/nib0004/initial_state.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = '_oce_def'
oce_moc = '_oce_moc'
oce_mon = '_oce_mon'
oce_monthly = '_oce_dbg'

tave_ints = [
['1480-02-01', '1490-01-01'],
['1580-02-01', '1590-01-01'],
['1680-02-01', '1690-01-01'],
['1780-02-01', '1790-01-01'],
]
