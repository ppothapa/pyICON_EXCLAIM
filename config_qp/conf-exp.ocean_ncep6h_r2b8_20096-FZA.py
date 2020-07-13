runname = ''
run     = 'exp.ocean_ncep6h_r2b8_20096-FZA'
gname   = 'r2b8'
lev     = 'L128'
#tstep   = '17800101T000000Z'
tstep   = '????????????????'  # use this line for all data

path_data    = '/work/mh0287/users/helmuth/icon-oes-1.3.00/experiments/'+run+'/'
path_grid    = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree = 'auto'

fpath_ref_data_oce = 'auto'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = ''
oce_moc = '_oce_moc'
oce_mon = '_oce_mon'
oce_monthly = ''

tave_ints = [
['1948-03-01', '1949-02-01'],
]
