# Settings
# --------
Set.path_base = '/mnt/lustre01/pf/zmaw/m300602/qp_compare_test/'
Set.path_pics = Set.path_base+'pics/'

Set.tstr = '????????????????'
Set.prfx_3d      = ''
Set.prfx_2d      = ''
Set.prfx_monitor = '_oceanMonitor'
Set.prfx_moc     = '_MOC'

path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/old_experiments/'

# Simulations
# -----------
Sims = []
S = Simulation()
S.run = 'nib0001'
Sims.append(S)

S = Simulation()
S.run = 'nib0002'
Sims.append(S)

S = Simulation()
S.run = 'nib0003'
Sims.append(S)

S = Simulation()
S.run = 'nib0004'
Sims.append(S)

S = Simulation()
S.run = 'nib0005'
Sims.append(S)

for nn, S in enumerate(Sims):
  gname = 'r2b4_oce_r0004'
  lev = 'L40'
  path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'
  S.path_data = f'{path_base}/{S.run}/'
  S.tave_int = ['2090', '2100']
  S.name = S.run
  S.fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res1.00_180W-180E_90S-90N.npz' 
  S.fpath_tgrid   = f'{path_grid}/{gname}/{gname}_tgrid.nc'
  #S.fpath_fx      = f'{path_grid}/{gname}/{gname}_{lev}_fx.nc')
  S.fpath_fx      = f'{S.path_data}{S.run}_fx.nc'
