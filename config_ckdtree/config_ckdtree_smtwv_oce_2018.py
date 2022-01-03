import numpy as np
from netCDF4 import Dataset
import sys, os
import matplotlib
if len(sys.argv)==2:
  matplotlib.use('Agg')
import pyicon as pyic

ts = pyic.timing([0], 'start')
#path_tgrid    = '/mnt/lustre01/work/mh0287/users/leonidas/icon/submeso/grid/'
#fname_tgrid   = 'OceanOnly_submeso_WalvisRidge_2500m_srtm30.nc'

rev           = f''
tgname        = f'smtwv_oce_2018'
gname         = f'OceanOnly_WalvisRidge_submeso_srtm15_2018'
path_tgrid    = f'/pool/data/ICON/oes/grids/OceanOnly/' 
fname_tgrid   = f'{gname}.nc'
path_ckdtree  = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{tgname}/ckdtree_new/'
path_rgrid    = path_ckdtree + 'rectgrids/' 
path_sections = path_ckdtree + 'sections/' 

all_grids = [
#  'global_1.0',
  'global_0.3',
  'global_0.1',
  'global_0.02',
  #'global_0.3_nn5',
  #'global_0.3_nn20',
#  'global_0.3_nn500',
            ]

all_secs = [
  '30W_200pts',
  '170W_200pts',
  '30W_300pts',
  '170W_300pts',
            ]

#all_grids = []
all_secs = []

load_cgrid = True,
load_vgrid = True
load_egrid = False

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
  sname = 'global_1.0'
  if sname in all_grids:
    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
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
                      load_cgrid=load_cgrid,
                      load_egrid=load_egrid,
                      load_vgrid=load_vgrid,
                      )
  

if False:
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
