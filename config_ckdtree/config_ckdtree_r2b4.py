import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload

import pyicon as pyic
reload(pyic)

ts = pyic.timing([0], 'start')

tgname        = 'r2b4'
gname         = 'icon_grid_0036_R02B04_O'
path_tgrid    = f'/pool/data/ICON/oes/input/r0004/{gname}/'
fname_tgrid   = f'{gname}.nc'
path_ckdtree  = f'/mnt/lustre01/work/mh0033/m211054/projects/icon/grids/{tgname}/ckdtree/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
  'global_1.0',
  'global_0.3',
  'global_0.1',
            ]

all_secs = [
  '30W_200pts',
  '170W_200pts',
  '30W_300pts',
  '170W_300pts',
            ]

#all_grids = []
#all_secs = []

gnames = [gname]

if not os.path.exists(path_rgrid): 
  os.makedirs(path_rgrid)
if not os.path.exists(path_sections): 
  os.makedirs(path_sections)

for gname in gnames:
  ts = pyic.timing(ts, gname)
  print(gname)

  # --- grids
  sname = 'global_1.0'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )
  
  sname = 'global_0.3'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )
  
  sname = 'global_0.1'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.1,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )

  sname = 'global_0.02'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.02,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )
  

  # --- sections
  sname = '30W_200pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=200,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )
    
  sname = '170W_200pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=200,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )

  sname = '30W_300pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=300,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )
    
  sname = '170W_300pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=300,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      )

print('make_ckdtree.py: All done!')
ts = pyic.timing(ts, 'All done!')
