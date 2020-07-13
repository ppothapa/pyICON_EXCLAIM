runname   = ''
run       = 'sillsncep'

gname     = 'smt20km'
lev       = 'L128'

do_atmosphere_plots = False

tstep     = '????????????????'  # use this line for all data

path_data     = f'/mnt/lustre01/work/mh0033/m214056/RESOLUTION_EXPS/NEW_RUN_PAPER2019/icon-oes-1.3.01/experiments/{run}/'
path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
path_ckdtree  = 'auto'

fpath_ref_data_oce = f'{path_data}/initial_state.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = ''
oce_moc = '_MOC'
oce_mon = '_oceanMonitor'
oce_monthly = ''

tave_ints = [
['1950-02-01', '1951-01-01'],
]
