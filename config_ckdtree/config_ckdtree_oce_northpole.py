import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload
import pyicon as pyic

path_ckdtree  = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_ckdtree/misk/'

all_grids = [
  'np_60N-90N_10km',
            ]

#all_grids = []

gnames = dict()
path_tgrids = dict()

# --- ocean grids
path_tgrid_oce = '/pool/data/ICON/oes/input/r0003/' 
# ---
name = 'r2b4'
gnames[name] = 'OceanOnly_Icos_0158km_etopo40'
path_tgrids[name] = path_tgrid_oce
# ---
name = 'r2b6'
gnames[name] = 'OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min'
path_tgrids[name] = path_tgrid_oce
# ---
name =  'r2b8'
gnames[name] = 'OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min'
path_tgrids[name] = path_tgrid_oce
# ---
name =  'r2b9'
gnames[name] = 'OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min'
path_tgrids[name] = path_tgrid_oce
# ---
name = 'r2b4a'
gnames[name] = 'icon_grid_0013_R02B04_G'
path_tgrids[name] = '/pool/data/ICON/grids/public/mpim/0013/'

if not os.path.exists(path_ckdtree): 
  os.makedirs(path_ckdtree)

names_process = gnames.keys()
#names_process = ['r2b4', 'r2b4a']

for name in names_process:
  print(name)
  path_tgrid = path_tgrids[name]
  gname = gnames[name]

  if name.endswith('a'):
    fpath_tgrid  = path_tgrid+'/'+gname+'.nc'
  else:
    fpath_tgrid  = path_tgrid+gname+'/'+gname+'.nc'
  Drgrid = pyic.identify_grid(path_tgrid, fpath_tgrid)

  # --- grids
  sname = 'np_60N-90N_10km'
  print(gname+': '+sname)
  if sname in all_grids:
    Lon_np, Lat_np = pyic.calc_north_pole_interp_grid_points(lat_south=60., res=10e3)
    fpath_ckdtree = path_ckdtree+Drgrid['name']+'_'+sname+'.npz'

    # --- calculate ckdtree
    Dind_dist = pyic.ckdtree_points(fpath_tgrid, Lon_np.flatten(), Lat_np.flatten(), 
      load_cgrid=True, load_egrid=True, load_vgrid=True)

    # --- save grid
    print('Saving grid file: %s' % (fpath_ckdtree))
    np.savez(fpath_ckdtree,
              Lon_np=Lon_np,
              Lat_np=Lat_np,
              sname=sname,
              gname=gname,
              **Dind_dist,
             )

print('make_ckdtree.py: All done!')
