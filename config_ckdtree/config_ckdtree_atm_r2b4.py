import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload

import pyicon as pyic
reload(pyic)

#path_tgrid    = '/pool/data/ICON/oes/input/r0002/' 
path_tgrid    = '/pool/data/ICON/grids/public/mpim/0015/'
#icon_grid_0015_R02B09_G.nc
path_ckdtree  = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_ckdtree/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
  'global_1.0',
  'global_1.0_era',
  'global_0.3',
#  'global_0.1',
            ]

all_secs = [
#  '30W_100pts',
#  '30W_200pts',
#  '170W_100pts',
            ]

#all_grids = []
#all_secs = []

gnames = []
path_tgrids = []

# r2b9
#gnames += ['icon_grid_0015_R02B09_G']
#path_tgrids += ['/pool/data/ICON/grids/public/mpim/0015/']
# r2b4
gnames += ['icon_grid_0013_R02B04_G']
path_tgrids += ['/pool/data/ICON/grids/public/mpim/0013/']

if not os.path.exists(path_rgrid): 
  os.makedirs(path_rgrid)
if not os.path.exists(path_sections): 
  os.makedirs(path_sections)

for gname, path_tgrid in zip(gnames, path_tgrids):
  print(path_tgrid, gname)

  # --- grids
  sname = 'global_1.0_era'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-179.,181.], lat_reg=[-89.5,90.], res=1.0,
                      fname_tgrid  = gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )

  sname = 'global_1.0'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  
  sname = 'global_0.3'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  
  sname = 'global_0.1'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.1,
                      fname_tgrid  = gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  

  # --- sections
  sname = '30W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )

  sname = '30W_200pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=200,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )
    
  sname = '170W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )

print('make_ckdtree.py: All done!')
