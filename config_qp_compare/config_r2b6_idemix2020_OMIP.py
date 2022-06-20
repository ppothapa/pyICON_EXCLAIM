# Settings
# --------
Set.name = 'r2b6_idemix2020_OMIP'
Set.omit_last_file = True
Set.path_base = f'/home/m/m300602/pyicon/all_qp_compare/{Set.name}/'
Set.path_pics = Set.path_base+'pics/'
Set.do_diff = False
Set.compare_with_reference = True

Set.mfdset_kwargs = dict(combine='nested', concat_dim='time', use_cftime=True)
#Set.mfdset_kwargs = dict(combine='nested', concat_dim='time')

path_base = '/work/mh0033/m300602/proj_vmix/icon/icon_17_levante/icon-oes/experiments/'
path_grid = '/work/mh0033/m300602/icon/grids/'

# Simulations
# -----------
Sims = []

# c_k=0.3, tke_min=1e-6
S = Simulation()
S.run = 'nib0037'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.3, IDEMIX Jayne
S = Simulation()
S.run = 'nib0043'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.3, IDEMIX Nycander Falahat
S = Simulation()
S.run = 'nib0048'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.1, IDEMIX Stormtide Nycander
S = Simulation()
S.run = 'nib0049'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

for nn, S in enumerate(Sims):
  S.path_data = f'{path_base}/{S.run}/'
  S.tave_int = ['2140', '2150']
  S.fpath_fx      = f'{S.path_data}{S.run}_fx.nc'
  S.namelist_oce  = f'{S.path_data}NAMELIST_{S.run}'
  S.fpath_ref     = f'{S.path_data}initial_state.nc'
