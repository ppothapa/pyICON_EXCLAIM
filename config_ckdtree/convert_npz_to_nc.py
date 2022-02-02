import sys, os
import glob
import numpy as np
import pyicon as pyic
import xarray as xr

#gname = 'r2b4_oce_r0004'
#gname = 'r2b6_oce_r0004'
#gname = 'r2b8_oce_r0004'
#gname = 'r2b9_oce_r0004'
gname = 'r2b10_oce'
#gname = 'smtwv_oce_2018'
#gname = 'smt'
#gname = 'smt20km'
#
#gname = 'r2b9_atm_r0015'
#gname = 'r2b4_atm_r0013'

# --- rectgrids
path_grid = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/ckdtree/rectgrids/'
flist = glob.glob(path_grid+'/*.npz')
flist.sort()

for fpath_ckdtree in flist:
  print(f'{fpath_ckdtree}')
  ddnpz = np.load(fpath_ckdtree)
  lon = ddnpz['lon']
  lat = ddnpz['lat']
  nx, ny = lon.size, lat.size

  ds = xr.Dataset(coords=dict(lon=lon, lat=lat))
  for var in ddnpz.keys():
    if var.startswith('d') or var.startswith('i'):
      ds[var] = xr.DataArray(ddnpz[var].reshape(ny, nx), dims=['lat', 'lon'])
  fpath_nc = fpath_ckdtree.split('.npz')[0]+'.nc'
  print(f'Saving file {fpath_nc}')
  ds.to_netcdf(fpath_nc)

# --- sections
path_grid = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/ckdtree/sections/'
flist = glob.glob(path_grid+'/*.npz')
flist.sort()

for fpath_ckdtree in flist:
  print(f'{fpath_ckdtree}')
  ddnpz = np.load(fpath_ckdtree)
  lon = ddnpz['lon_sec']
  lat = ddnpz['lat_sec']
  dist = ddnpz['dist_sec']

  ds = xr.Dataset(coords=dict(lon=lon, lat=lat, dist=dist))
  ds['lon'] = xr.DataArray(lon, dims=['isec'])
  ds['lat'] = xr.DataArray(lat, dims=['isec'])
  ds['dist'] = xr.DataArray(dist, dims=['isec'])
  for var in ddnpz.keys():
    if var.startswith('d') or var.startswith('i'):
      ds[var] = xr.DataArray(ddnpz[var], dims=['isec'])
  fpath_nc = fpath_ckdtree.split('.npz')[0]+'.nc'
  print(f'Saving file {fpath_nc}')
  ds.to_netcdf(fpath_nc)
