import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload

import pyicon as pyic
reload(pyic)

ts = pyic.timing([0], 'start')

tgname        = 'r2b4_atm_r0012'
gname         = 'icon_grid_0012_R02B04_G'
path_tgrid    = f'/hpc/uwork/gboeloen/pyICON/grids/'
fname_tgrid   = f'{gname}.nc'
path_ckdtree  = f'/hpc/uwork/gboeloen/pyICON/grids/{tgname}/ckdtree/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
  'global_1.0',
  'global_1.0_era',
  'global_1.5_era5',
  'regional_1.0_era',
  'global_0.3',
            ]

all_secs = [
  '30W_100pts',
  '30W_200pts',
  '170W_100pts',
           ]

#all_grids = []
#all_secs = []

gnames = [gname]

if not os.path.exists(path_rgrid): 
  os.makedirs(path_rgrid)
if not os.path.exists(path_sections): 
  os.makedirs(path_sections)
fpath = f'{path_ckdtree}/../{tgname}_tgrid.nc'
if not os.path.exists(fpath):
  os.symlink(path_tgrid+fname_tgrid, fpath)

for gname in gnames:
  ts = pyic.timing(ts, gname)
  print(gname)

  # --- grids
  sname = 'global_1.0_era'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-179.,181.], lat_reg=[-89.5,90.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )

  sname = 'global_1.5_era5'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180], lat_reg=[-90,91.5], res=1.5,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )

  sname = 'regional_1.0_era'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-30.,40.], lat_reg=[30.,80.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )


  sname = 'global_1.0'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
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
                      save_as_nc=False,
                      )
  
  # --- sections
  sname = '30W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=100,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )

  sname = '30W_200pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=200,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )
    
  sname = '170W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=100,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      save_as_nc=False,
                      )

print('make_ckdtree.py: All done!')
ts = pyic.timing(ts, 'All done!')
