import numpy as np
from netCDF4 import Dataset
import sys
from importlib import reload

import pyicon as pyic
reload(pyic)

path_tgrid    = 
path_ckdtree  = 
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
  'global_1.0',
  'global_0.3',
  'global_0.1',
            ]

all_secs = [
  '30W_100pts',
  '30W_200pts',
  '170W_100pts',
            ]

all_grids = []
all_secs = []

gnames = ['OceanOnly_Icos_0158km_etopo40']

for gname in gnames:
  # --- grids
  sname = 'global_1.0'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname
                      )
  
  sname = 'global_0.3'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname
                      )
  
  sname = 'global_0.1'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.1,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname
                      )
  

  # --- sections
  sname = '30W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname
                      )

  sname = '30W_200pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=200,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname
                      )
    
  sname = '170W_100pts'
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname
                      )

return('make_ckdtree.py: All done!')
