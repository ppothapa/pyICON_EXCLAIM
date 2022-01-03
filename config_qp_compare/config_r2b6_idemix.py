# Settings
# --------
Set.name = 'r2b6_idemix'
Set.path_base = f'/mnt/lustre01/pf/zmaw/m300602/qp_compare_test/{Set.name}/'
Set.path_pics = Set.path_base+'pics/'
Set.do_diff = False
Set.compare_with_reference = True

#Set.mfdset_kwargs = dict(combine='nested', concat_dim='time', use_cftime=True)
Set.mfdset_kwargs = dict(combine='nested', concat_dim='time')

path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/experiments/'

# Simulations
# -----------
Sims = []

path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'

#S = Simulation()
#S.run = 'nib0012'
#S.name = S.run
#S.gname = 'r2b4_oce_r0004'
#S.lev = 'L40'
#S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
#S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
#Sims.append(S)

# c_k=0.1
S = Simulation()
S.run = 'nib0033'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

#S = Simulation()
#S.run = 'nib0035'
#S.name = S.run
#S.gname = 'r2b6_oce_r0004'
#S.lev = 'L64'
#S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
#S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
#Sims.append(S)
#
#S = Simulation()
#S.run = 'nib0036'
#S.name = S.run
#S.gname = 'r2b6_oce_r0004'
#S.lev = 'L64'
#S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
#S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
#Sims.append(S)

# c_k=0.3, tke_min=1e-6
S = Simulation()
S.run = 'nib0037'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.1, tke_min=5e-6
S = Simulation()
S.run = 'nib0038'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

#S = Simulation()
#S.run = 'nib0039'
#S.name = S.run
#S.gname = 'r2b6_oce_r0004'
#S.lev = 'L64'
#S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
#S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
#Sims.append(S)
#
#S = Simulation()
#S.run = 'nib0040'
#S.name = S.run
#S.gname = 'r2b6_oce_r0004'
#S.lev = 'L64'
#S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
#S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
#Sims.append(S)

# c_k=0.2, IDEMIX Olbers/Eden parameters
S = Simulation()
S.run = 'nib0041'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.3, IDEMIX Olbers/Eden parameters
S = Simulation()
S.run = 'nib0042'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

# c_k=0.3, IDEMIX Pollmann parameters
S = Simulation()
S.run = 'nib0043'
S.name = S.run
S.gname = 'r2b6_oce_r0004'
S.lev = 'L64'
S.fpath_ckdtree = f'{path_grid}/{S.gname}/ckdtree/rectgrids/{S.gname}_res1.00_180W-180E_90S-90N.npz' 
S.fpath_tgrid = f'{path_grid}/{S.gname}/{S.gname}_tgrid.nc'
Sims.append(S)

for nn, S in enumerate(Sims):
  S.path_data = f'{path_base}/{S.run}/'
  S.tave_int = ['2090', '2100']
  S.fpath_fx      = f'{S.path_data}{S.run}_fx.nc'
  S.namelist_oce  = f'{S.path_data}NAMELIST_{S.run}'
  S.fpath_ref     = f'{S.path_data}initial_state.nc'
