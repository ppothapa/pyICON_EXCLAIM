# Settings
# --------
Set.name = 'r2b4_diff_GM'
Set.path_base = f'/mnt/lustre01/pf/zmaw/m300602/qp_compare_test/{Set.name}/'
Set.path_pics = Set.path_base+'pics/'
Set.do_diff = False
Set.compare_with_reference = True

#path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/old_experiments/'
path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/experiments/'

# Simulations
# -----------
Sims = []

# GM=1000
S = Simulation()
S.run = 'nib0012'
Sims.append(S)

# GM=800 / 1000
S = Simulation()
S.run = 'nib0016'
Sims.append(S)

# GM=1200 / 1000
S = Simulation()
S.run = 'nib0017'
Sims.append(S)

# GM=400 / 1000
S = Simulation()
S.run = 'nib0018'
Sims.append(S)

# GM=0 / 1000
S = Simulation()
S.run = 'nib0019'
Sims.append(S)

# GM=0 / 0
S = Simulation()
S.run = 'nib0020'
Sims.append(S)

# GM=400 / 400
S = Simulation()
S.run = 'nib0021'
Sims.append(S)

# GM=800 / 800
S = Simulation()
S.run = 'nib0022'
Sims.append(S)

for nn, S in enumerate(Sims):
  gname = 'r2b4_oce_r0004'
  lev = 'L40'
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
