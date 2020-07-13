runname = ''
run = 'exp.ocean_omip_long_tke_r2b8_20134-WWG'

gname     = 'r2b8'
lev       = 'L128'

do_atmosphere_plots = False

tstep     = '????????????????'  # use this line for all data

path_data     = '/mnt/lustre01/work/mh0033/m211054/projects/icon/icon-oes-1.3.01/experiments/'+run+'/'
path_grid     = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname+'/'
path_ckdtree  = 'auto'

fpath_ref_data_oce = '/pool/data/ICON/oes/input/r0003/OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min/ts_phc3.0_annual_icon_OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min_L128.nc'
fpath_tgrid  = 'auto'
fpath_fx     = 'auto'

oce_def = ''
oce_moc = '_MOC'
oce_mon = '_oceanMonitor'
oce_monthly = ''

omit_last_file = True

tave_ints = [
#['1480-02-01', '1490-01-01'],
#['1580-02-01', '1590-01-01'],
['1932-02-01', '1933-01-01'],
#['1780-02-01', '1790-01-01'],
]
