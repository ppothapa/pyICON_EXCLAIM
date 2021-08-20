#!/usr/bin/env python

print('start modules')
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import glob
import os
import sys
sys.path.append('/home/mpim/m300602/pyicon/')
import pyicon as pyic  
#from pyicon import arrange_axes, shade, plot_settings
#from pyicon import interp_to_rectgrid
#import pandas as pd
#import matplotlib.transforms as transforms
#import matplotlib
import argparse
from pathlib import Path
from netCDF4 import Dataset
print('Done modules.')

arrange_axes = pyic.arrange_axes
shade = pyic.shade
plot_settings = pyic.plot_settings
interp_to_rectgrid = pyic.interp_to_rectgrid

help_text = """
Makes a figure from ICON data

Usage notes:
------------


Argument list:
--------------
"""

# --- default arguments
#fpath = '/mnt/lustre02/work/bm1102/m300761/Projects/smt2/icon-oes/experiments/r7_smt_wave/r7_smt_wave_P1D_2d_20190701T093000Z.nc'
#it = 0
#iz = 0
#projection = 'pc'
#lon_reg = None
#lat_reg = None
#cmap = 'auto' 
#clim = [None, None]
#var = 'auto'
#res = 0.3
#gname = 'smtwv_oce_2018'

# --- read input arguments
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

# --- necessary arguments
parser.add_argument('fpath_data', nargs='+', metavar='fpath_data', type=str,
                    help='Path to ICON data file.')
parser.add_argument('var', metavar='var', type=str,
                    help='Name of variable which should be plotted.')
# --- optional arguments
parser.add_argument('--fpath_fig', type=str, default='none',
                    help='Path to save the figure.')
parser.add_argument('--show', default=False,
                    action='store_true',
                    help='If specified, the plot is directly shown')
parser.add_argument('--use_tgrid', default=False,
                    action='store_true',
                    help='If specified, the plot is made on the original triangular grid.')
parser.add_argument('--gname', type=str, default='auto',
                    help='Grid name of the ICON data.')
parser.add_argument('--it', type=int, default=0,
                    help='Time index which should be plotted.')
parser.add_argument('--time', type=str, default='none',
                    help='Time string \'yyyy-mm-dd\' wich should be plotted (if specified overwrites it).')
parser.add_argument('--iz', type=int, default=0,
                    help='Depth index which should be plotted.')
parser.add_argument('--depth', type=float, default=-1.,
                    help='Depth value in m which should be plotted (if specified overwrites iz).')
parser.add_argument('--projection', type=str, default='pc',
                    help='Map projection, choose \'pc\' or None.')
parser.add_argument('--res', type=float, default=0.3,
                    help='Resolution of the interpolated data which will be plotted. So far, 1.0, 0.3, 0.1 are supported.')
parser.add_argument('--fpath_tgrid', type=str, default='auto',
                    help='Path to triangular grid file. If \'auto\' the path is guessed automatically. Only necessary if \'--use_tgrid\' is used.')
parser.add_argument('--lon_reg', type=str, default=None,
                    help='Longitude range of the plot.')
parser.add_argument('--lat_reg', type=str, default=None,
                    help='Latitude range of the plot.')
parser.add_argument('--cmap', type=str, default='auto',
                    help='Colormap used for plot.')
parser.add_argument('--clim', type=str, default='auto',
                    help='Color limits of the plot. Either specify one or two values.If one value is specified color limits are taken symetrically around zero. If \'auto\' is specified color limits are derived automatically.')

iopts = parser.parse_args()

fpath_data = iopts.fpath_data
var = iopts.var
#output = iopts.output
gname = iopts.gname
it = iopts.it
time = iopts.time
iz = iopts.iz
depth = iopts.depth
projection = iopts.projection
res = iopts.res
lon_reg = iopts.lon_reg
lat_reg = iopts.lat_reg
cmap = iopts.cmap
clim = iopts.clim
fpath_fig = iopts.fpath_fig
use_tgrid = iopts.use_tgrid
fpath_tgrid = iopts.fpath_tgrid

# --- variable replacements
if var=='auto':
  var = list(ds.keys())[-1]

if clim!='auto':
  clim = clim.replace(' ', '')
  clim = np.array(clim.split(','), dtype=float)
if lon_reg:
  lon_reg = lon_reg.replace(' ', '')
  lon_reg = np.array(lon_reg.split(','), dtype=float)
if lat_reg:
  lat_reg = lat_reg.replace(' ', '')
  lat_reg = np.array(lat_reg.split(','), dtype=float)
if isinstance(projection, str) and projection=='pc':
  projection = ccrs.PlateCarree()
path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'
if isinstance(fpath_data, list):
  fpath = fpath_data[0]
else:
  fpath = fpath_data
if gname=='auto':
  Dgrid = pyic.identify_grid(path_grid, fpath)
  gname = Dgrid['name']
if fpath_tgrid=='auto':
  Dgrid = pyic.identify_grid(path_grid, fpath)
  fpath_tgrid = Dgrid['fpath_grid']
fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res{res:3.2f}_180W-180E_90S-90N.npz'
print(fpath_ckdtree)

# --- open dataset
mfdset_kwargs = dict(combine='nested', concat_dim='time', 
                     data_vars='minimal', coords='minimal', 
                     compat='override', join='override',)
ds = xr.open_mfdataset(fpath_data, **mfdset_kwargs)

# --- reduce time and depth dimension
data = ds[var]
if 'depth' in data.dims:
  depth_name = 'depth'
elif 'depth_2' in data.dims:
  depth_name = 'depth_2'
else:
  depth_name = 'none'

if depth_name!='none':
  if depth!=-1:
    data = data.sel({depth_name: depth}, method='nearest')
  else:
    data = data.isel({depth_name: iz})
if 'time' in data.dims:
  if time=='none':
    data = data.isel(time=it)
  else:
    data = data.sel(time=time, method='nearest')

# --- interpolate and cut to region
if not use_tgrid:
  lon, lat, datai = interp_to_rectgrid(data, fpath_ckdtree, lon_reg=lon_reg, lat_reg=lat_reg)
else:
  print('Deriving triangulation object, this can take a while...')
  f = Dataset(fpath_tgrid, 'r')
  clon = f.variables['clon'][:] * 180./np.pi
  clat = f.variables['clat'][:] * 180./np.pi
  vlon = f.variables['vlon'][:] * 180./np.pi
  vlat = f.variables['vlat'][:] * 180./np.pi
  vertex_of_cell = f.variables['vertex_of_cell'][:].transpose()-1
  f.close()
  
  ind_reg = np.where(   (clon>lon_reg[0])
                      & (clon<=lon_reg[1])
                      & (clat>lat_reg[0])
                      & (clat<=lat_reg[1]) )[0]
  vertex_of_cell_reg = vertex_of_cell[ind_reg,:]
  Tri = matplotlib.tri.Triangulation(vlon, vlat, triangles=vertex_of_cell_reg)
  data_reg = data.compute().data[ind_reg]
  print('Done deriving triangulation object.')

# -- start plotting
plt.close('all')
asp = (lat_reg[1]-lat_reg[0])/(lon_reg[1]-lon_reg[0])
hca, hcb = arrange_axes(1,1, plot_cb=True, asp=asp, fig_size_fac=2,
                             sharex=True, sharey=True, xlabel="", ylabel="",
                             projection=projection, axlab_kw=None, dfigr=0.5,
                            )
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
if not use_tgrid:
  hm = shade(lon, lat, datai, ax=ax, cax=cax, clim=clim, projection=projection)
else:
  hm = shade(Tri, data_reg, ax=ax, cax=cax, clim=clim, projection=projection)
#ax.set_title(f'{data.long_name} [{data.units}]', loc='left')
cax.set_ylabel(f'{data.long_name} [{data.units}]')
tstr = str(data.time.data)
#tstr = tstr.split('T')[0].replace('-', '')+'T'+tstr.split('T')[1].split('.')[0].replace(':','')+'Z'
tstr = tstr.split('.')[0]
if 'time' in ds[var].dims:
  ht = ax.set_title(tstr, loc='right')
if depth_name!='none':
  ht = ax.set_title(f'depth = {data[depth_name].data:.1f}m', loc='left')

if projection is None:
  ax.set_xlabel('longitude')
  ax.set_ylabel('latitude')

if not projection:
  ax.set_facecolor('0.7')

if lon_reg is None:
  xlim = 'none'
else:
  xlim = lon_reg
if lat_reg is None:
  ylim = 'none'
else:
  ylim = lat_reg

if (lon_reg is None) and (lat_reg is None):
  plot_settings(ax, template='global')
else:
  plot_settings(ax, xlim=xlim, ylim=ylim)

if fpath_fig!='none':
  if fpath_fig.startswith('~'):
    home = str(Path.home())+'/'
    fpath_fig = home + fpath_fig[1:]
  print(f'Saving figure {fpath_fig}...')
  plt.savefig(fpath_fig)

if iopts.show:
  plt.show()
