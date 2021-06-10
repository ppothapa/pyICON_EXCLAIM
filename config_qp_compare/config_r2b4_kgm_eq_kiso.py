# Settings
# --------
Set.name = 'r2b4_kgm_eq_kiso'
Set.path_base = f'/mnt/lustre01/pf/zmaw/m300602/qp_compare_test/{Set.name}/'
Set.path_pics = Set.path_base+'pics/'
Set.do_diff = False
Set.compare_with_reference = True

#path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/old_experiments/'
path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/experiments/'

# Simulations
# -----------
Sims = []

runs = [
#'nib0021', 'nib0022', 'nib0012',
#'nib0026', 'nib0023', 'nib0027',
'nib0021', 'nib0026', # 400 / 400
'nib0012', 'nib0023', # 1000 / 1000
'nib0022', 'nib0027', # 800 / 1200
]
for run in runs:
  S = Simulation()
  S.run = run
  Sims.append(S)

for nn, S in enumerate(Sims):
  S.gname = 'r2b4_oce_r0004'
  gname = S.gname
  S.lev = 'L40'
  path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'
  S.path_data = f'{path_base}/{S.run}/'
  S.tave_int = ['2090', '2100']
  #S.tave_int = ['2040', '2050']
  S.name = S.run
  S.fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res1.00_180W-180E_90S-90N.npz' 
  #S.fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz' 
  S.fpath_tgrid   = f'{path_grid}/{gname}/{gname}_tgrid.nc'
  #S.fpath_fx      = f'{path_grid}/{gname}/{gname}_{lev}_fx.nc')
  S.fpath_fx      = f'{S.path_data}{S.run}_fx.nc'
  S.namelist_oce  = f'{S.path_data}NAMELIST_{S.run}'
  S.fpath_ref     = f'{S.path_data}initial_state.nc'
