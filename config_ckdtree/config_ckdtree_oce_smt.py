import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload

import pyicon as pyic
reload(pyic)

ts = pyic.timing([0], 'start')

tgname        = 'smt'
gname         = 'OceanOnly_SubmesoNA_2500m_srtm30'
path_tgrid    = f'/pool/data/ICON/oes/grids/OceanOnly/'
fname_tgrid   = f'{gname}.nc'
path_ckdtree  = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{tgname}/ckdtree/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
#  'global_1.0',
  'global_0.3',
  #'global_0.3_nn5',
  #'global_0.3_nn20',
  #'global_0.3_nn500',
#  'global_0.1',
#  'global_0.02',
            ]

all_secs = [
  '30W_100pts',
  '30W_200pts',
  '170W_100pts',
            ]

#all_grids = []
all_secs = []

gnames = [gname]

if not os.path.exists(path_rgrid): 
  os.makedirs(path_rgrid)
if not os.path.exists(path_sections): 
  os.makedirs(path_sections)

for gname in gnames:
  ts = pyic.timing(ts, gname)
  #print(gname)

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
                      load_egrid=False,
                      load_vgrid=False,
                      )

  sname = 'global_0.3_nn500'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      load_egrid=False,
                      load_vgrid=False,
                      n_nearest_neighbours=500,
                      )

  sname = 'global_0.3_nn20'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      load_egrid=False,
                      load_vgrid=False,
                      n_nearest_neighbours=20,
                      )
  
  sname = 'global_0.3_nn5'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      load_egrid=False,
                      load_vgrid=False,
                      n_nearest_neighbours=5,
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
                      load_egrid=False,
                      load_vgrid=False,
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
                      load_egrid=False,
                      load_vgrid=False,
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
                      load_egrid=False,
                      load_vgrid=False,
                      )
  

if False:
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
                      load_egrid=False,
                      load_vgrid=False,
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
                      load_egrid=False,
                      load_vgrid=False,
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
                      load_egrid=False,
                      load_vgrid=False,
                      )

print('make_ckdtree.py: All done!')
ts = pyic.timing(ts, 'All done!')
