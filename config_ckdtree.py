import numpy as np
from netCDF4 import Dataset
import sys, os
from importlib import reload

import pyicon as pyic
reload(pyic)

path_tgrid    = '/pool/data/ICON/oes/input/r0003/' 
path_ckdtree  = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/icon_ckdtree/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
  'global_1.0',
  'global_0.3',
  'global_0.1',
  'global_0.05',
            ]

all_secs = [
  '30W_100pts',
  '30W_200pts',
  '170W_100pts',
            ]

#all_grids = []
#all_secs = []

gnames = []
# r2b4
#gnames += ['OceanOnly_Icos_0158km_etopo40'] 
# r2b6
gnames += ['OCEANINP_pre04_LndnoLak_039km_editSLOHH2017_G']
#gnames += ['OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min']
# r2b8
#gnames += ['OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min']
# r2b9
#gnames += ['OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min']

if not os.path.exists(path_rgrid): 
  os.makedirs(path_rgrid)
if not os.path.exists(path_sections): 
  os.makedirs(path_sections)

for gname in gnames:

  # --- grids
  sname = 'global_1.0'
  print(gname+': '+sname)
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  
  sname = 'global_0.3'
  print(gname+': '+sname)
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.3,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  
  sname = 'global_0.1'
  print(gname+': '+sname)
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.1,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  
  sname = 'global_0.05'
  print(gname+': '+sname)
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=0.05,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      )
  

  # --- sections
  sname = '30W_100pts'
  print(gname+': '+sname)
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )

  sname = '30W_200pts'
  print(gname+': '+sname)
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-30,-80], p2=[-30,80], npoints=200,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )
    
  sname = '170W_100pts'
  print(gname+': '+sname)
  if sname in all_secs:
    dckdtree, ickdtree, lon_sec, lat_sec, dist_sec = pyic.ckdtree_section(p1=[-170,-80], p2=[-170,80], npoints=100,
                      fname_tgrid  = gname+'/'+gname+'.nc',
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_sections,
                      sname = sname,
                      gname = gname,
                      )

print('make_ckdtree.py: All done!')
