import sys, glob, os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from netCDF4 import Dataset   
import pyicon as pyic
import cartopy.crs as ccrs 

def load_era3d_var(fpath, var, isort):
  f = Dataset(fpath, 'r')
  data = f.variables[var][:].mean(axis=0)
  f.close()
  data = data[:,:,isort]
  return data

path_datao = '/mnt/lustre01/work/mh0033/m300602/icon/era/'
path_datai_3d = '/pool/data/ICON/post/QuickPlots_1x1/ERAin/' 
path_datai_2d = '/pool/data/ICON/post/QuickPlots_1x1_1.3.0.0/ERAin/'
fnamei_2d = 'ERAinBil_1x1_atm_2d_1979-2016_MM.nc'
y1 = 1979
y2 = 2016
type = 'MM'
name = 'ERAinBil_1x1'

# --- load the grid
var = 'ta'
fpath = f'{path_datai_3d}{name}_{var}_{y1}-{y2}_{type}.nc'
f = Dataset(fpath, 'r')
lon = f.variables['lon'][:]
lat = f.variables['lat'][:]
plev = f.variables['plev'][:]
ncv_time = f.variables['time']
time = pyic.nctime_to_datetime64(ncv_time, time_mode='num2date')
f.close()

# --- shift grid
lon[lon>180.] += -360.
iw = (lon>0).sum()+1
isort = np.arange(lon.size)
isort = np.concatenate((isort[iw:], isort[:iw]))
lon = lon[isort]

#var = 'ta'
#print(var)
#fpath = f'{path_datai}{name}_{var}_{y1}-{y2}_{type}.nc'
#ta = load_era3d_var(fpath, var, isort)
#
#var = 'ua'
#print(var)
#fpath = f'{path_datai}{name}_{var}_{y1}-{y2}_{type}.nc'
#ua = load_era3d_var(fpath, var, isort)
#
#var = 'va'
#print(var)
#fpath = f'{path_datai}{name}_{var}_{y1}-{y2}_{type}.nc'
#va = load_era3d_var(fpath, var, isort)

fnameo = __file__.split('/')[-1].split('.')[0]+'.nc'
fo = Dataset(path_datao+fnameo, 'w', format='NETCDF4')
fo.createDimension('lon', lon.size)
fo.createDimension('lat', lat.size)
fo.createDimension('plev', plev.size)
ncv = fo.createVariable('lon','f4',('lon',))
ncv[:] = lon
ncv = fo.createVariable('lat','f4',('lat',))
ncv[:] = lat
ncv = fo.createVariable('plev','f4',('plev',))
ncv[:] = plev

# --- 2d data
ave_vars = ['tas', 'prw', 'psl', 'tauu', 'tauv', 'sfcWind']
f = Dataset(path_datai_2d+fnamei_2d, 'r')
for var in ave_vars:
  print(var)
  data = f.variables[var][:].mean(axis=0)
  data = data[:,isort]
  ncv = fo.createVariable(var, 'f4', ('lat', 'lon'))
  ncv[:] = data
f.close()

# --- 3d data
ave_vars = ['ua', 'va', 'ta', 'hur', 'hus']
for var in ave_vars:
  print(var)
  fpath = f'{path_datai_3d}{name}_{var}_{y1}-{y2}_{type}.nc'
  data = load_era3d_var(fpath, var, isort)
  ncv = fo.createVariable(var, 'f4', ('plev', 'lat', 'lon'))
  ncv[:] = data

fo.close()
print(f'Done with writing {path_datao+fnameo}')
