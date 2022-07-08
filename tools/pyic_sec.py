#!/usr/bin/env python

import argparse

help_text = """
Makes a figure from ICON data

Usage notes:
------------
Basic usage:
pyic_sec.py netcdf_file.nc var_name [options]

Change color limits, colorbar:
pyic_sec.py netcdf_file.nc var_name --clim=-10,32 --cmap=viridis

Select time step by indices:
pyic_sec.py netcdf_file.nc var_name --it=3 

Select date:
pyic_sec.py netcdf_files_*.nc var_name --time=2010-03-02 --depth=1000

Change x/y-limits:
pyic_sec.py netcdf_file.nc var_name ---xlim=-30,80 --ylim=0,3000

Add contours:
pyic_sec.py netcdf_file.nc var_name --clim=16 --cincr=2 --conts=auto --clabel

Save the figure:
pyic_sec.py netcdf_file.nc var_name --fpath_fig=/path/to/figure.png

Argument list:
--------------
"""

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
parser.add_argument('--dontshow', action='store_true', default=False,
                    help='If dontshow is specified, the plot is not shown')
parser.add_argument('--invert_yaxis', action='store_true', default=True,
                    help='Invert y-axis, starting with largest and ending with smalles value.')
parser.add_argument('--gname', type=str, default='auto',
                    help='Grid name of the ICON data.')
parser.add_argument('--it', type=int, default=0,
                    help='Time index which should be plotted.')
parser.add_argument('--time', type=str, default='none',
                    help='Time string \'yyyy-mm-dd\' wich should be plotted (if specified overwrites \'it\').')
parser.add_argument('--fpath_tgrid', type=str, default='auto',
                    help='Path to triangular grid file. If \'auto\' the path is guessed automatically. Only necessary if \'--use_tgrid\' is used.')
parser.add_argument('--xlim', type=str, default=None,
                    help='Limits for x-axis.')
parser.add_argument('--ylim', type=str, default=None,
                    help='Limits for y-axis.')
parser.add_argument('--cmap', type=str, default='auto',
                    help='Colormap used for plot.')
parser.add_argument('--clim', type=str, default='auto',
                    help='Color limits of the plot. Either specify one or two values.If one value is specified color limits are taken symetrically around zero. If \'auto\' is specified color limits are derived automatically.')
parser.add_argument('--cincr', type=float, default=-1.0,
                    help='Increment for pcolor plot to specify levels between clims.')
parser.add_argument('--clevs', type=str, default=None,
                    help='Color levels for pcolor plot.')
parser.add_argument('--conts', type=str, default=None,
                    help='Contour levels for monochromatic contours.')
parser.add_argument('--contfs', type=str, default=None,
                    help='Contour levels for filled contour patches.')
parser.add_argument('--clabel', action='store_true', default=False,
                    help='If clabel is specified, color labels will be shown.')
parser.add_argument('--facecolor', type=str, default='0.7',
                    help='Background color')
parser.add_argument('--title_center', type=str, default='auto',
                    help='Title string center.')
parser.add_argument('--title_left', type=str, default='auto',
                    help='Title string left.')
parser.add_argument('--title_right', type=str, default='auto',
                    help='Title string right.')
parser.add_argument('--xlabel', type=str, default='auto',
                    help='String for xlabel.')
parser.add_argument('--ylabel', type=str, default='depth [m]',
                    help='String for ylabel.')
parser.add_argument('--cbar_str', type=str, default='auto',
                    help='String for colorbar. Default is name of variable and its units.')
parser.add_argument('--cbar_pos', type=str, default='bottom',
                    help='Position of colorbar. It is possible to choose between \'right\' and \'bottom\'.')
parser.add_argument('--logplot', default=False,
                    action='store_true',
                    help='Plot logarithm of the data.')
parser.add_argument('--factor', type=float, default=None,
                    help='Factor to mulitply data with.')
parser.add_argument('--xdim', type=str, default='auto',
                    help='Dimension of x-axes of the plot. Choose between \{\'auto\'\}, \'lon\', or \'lat\'.')
parser.add_argument('--section', type=str, default='auto',
                    help='Section which is used for interpolation. At the moment, the following sections are supported: \'30W\', \'170W\'.')
parser.add_argument('--fpath_fx', type=str, default='none',
                    help='Path to an fx file fitting to the data.')

iopts = parser.parse_args()

#fpath_data = iopts.fpath_data
#var = iopts.var
#gname = iopts.gname
#it = iopts.it
#time = iopts.time
#iz = iopts.iz
#depth = iopts.depth
#projection = iopts.projection
#res = iopts.res
#cmap = iopts.cmap
#clim = iopts.clim
#fpath_fig = iopts.fpath_fig
#use_tgrid = iopts.use_tgrid
#fpath_tgrid = iopts.fpath_tgrid

print('start modules')
import matplotlib
if iopts.dontshow:
  matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xarray as xr
import glob
import os
import sys
sys.path.append('/home/mpim/m300602/pyicon/')
import pyicon as pyic  
from pathlib import Path
from netCDF4 import Dataset
print('Done modules.')

arrange_axes = pyic.arrange_axes
shade = pyic.shade
plot_settings = pyic.plot_settings

def str_to_array(string):
  string = string.replace(' ', '')
  array = np.array(string.split(','), dtype=float)
  return array

# --- limits
if iopts.clim!='auto':
  iopts.clim = str_to_array(iopts.clim)
if iopts.xlim:
  iopts.xlim = str_to_array(iopts.xlim)
if iopts.ylim:
  iopts.ylim = str_to_array(iopts.ylim)

# --- contour values
if iopts.conts and iopts.conts!='auto':
  iopts.conts = str_to_array(iopts.conts)
if iopts.contfs and iopts.contfs!='auto':
  iopts.contfs = str_to_array(iopts.contfs)
if iopts.clevs:
  iopts.clevs = str_to_array(iopts.clevs)

# --- grid files and interpolation
path_grid = '/work/mh0033/m300602/icon/grids/'
#path_grid = '/home/m/m300602/icon/grids/'
if isinstance(iopts.fpath_data, list):
  fpath = iopts.fpath_data[0]
else:
  fpath = iopts.fpath_data
if iopts.gname=='auto':
  try:
    Dgrid = pyic.identify_grid(path_grid, fpath)
    iopts.gname = Dgrid['name']
  except:
    iopts.gname = 'none'
if iopts.fpath_tgrid=='auto':
  try:
    Dgrid = pyic.identify_grid(path_grid, fpath)
    iopts.fpath_tgrid = Dgrid['fpath_grid']
  except:
    iopts.fpath_tgrid = 'from_file'
fpath_ckdtree = f'{path_grid}/{iopts.gname}/ckdtree/sections/{iopts.gname}_nps300_{iopts.section}80S_{iopts.section}80N.nc'

# --- open dataset
mfdset_kwargs = dict(combine='nested', concat_dim='time', 
                     data_vars='minimal', coords='minimal', 
                     compat='override', join='override',)
ds = xr.open_mfdataset(iopts.fpath_data, **mfdset_kwargs)

# --- reduce time and depth dimension
data = ds[iopts.var]
if 'depth' in data.dims:
  depth_name = 'depth'
elif 'depth_2' in data.dims:
  depth_name = 'depth_2'
else:
  depth_name = 'none'

if 'time' in data.dims:
  if iopts.time=='none':
    data = data.isel(time=iopts.it)
  else:
    data = data.sel(time=iopts.time, method='nearest')

if 'ncells' in data.dims:
  coordinates = 'clat clon'
  interp = True
elif 'ncells_2' in data.dims:
  data = data.rename(ncells_2='ncells') 
  coordinates = 'vlat vlon'
  interp = True
else:
  interp = False

if 'zave' in iopts.section:
  interp = False

if interp:
  ds_ckdt = xr.open_dataset(fpath_ckdtree)
  if 'clat' in coordinates:
    inds = ds_ckdt.ickdtree_c.data
  elif 'vlat' in coordinates:
    inds = ds_ckdt.ickdtree_v.data
  data = data.isel(ncells=inds)

if 'zave' in iopts.section:
  ds_fx = xr.open_dataset(iopts.fpath_fx)
  clat = data.clat * 180./np.pi
  lat_group = np.round(clat/0.3)*0.3
  data = data.where(data!=0)
  if iopts.section=='gzave':
    data = data.groupby(lat_group).mean()
    iopts.xlim = [-80, 90]
  elif iopts.section=='azave':
    data = data.where(ds_fx.basin_c==1.).groupby(lat_group).mean()
    iopts.xlim = [-30, 90]
  elif iopts.section=='ipzave':
    data = data.where((ds_fx.basin_c==3.) | (ds_fx.basin_c==7.)).groupby(lat_group).mean()
    iopts.xlim = [-30, 70]
  data = data.compute()
  xdim = data.clat
  xdim = xdim.assign_attrs(long_name='latitude')
  iopts.xdim = 'none' # do not need this information, avoid redefinition of xdim

if iopts.factor:
  data *= iopts.factor
data = data.squeeze()

if iopts.xdim=='auto':
  xdim = data[data.dims[1]]
elif 'lat' in iopts.xdim:
  xdim = ds_ckdt.lat_sec
  xdim = xdim.assign_attrs(long_name='latitude')
elif 'lon' in iopts.xdim:
  xdim = 'longitude'
  xdim = xdim.assign_attrs(long_name='longitude')
ydim = data[data.dims[0]]

data = data.where(data!=0)

# --- aspect ratio of the plot
asp = 0.5

# --- title, colorbar, and x/y label  strings
if iopts.cbar_str=='auto':
  try:
    units = data.units
  except:
    units = 'NA'
  try:
    long_name = data.long_name
  except:
    long_name = iopts.var
  if iopts.logplot:
    iopts.cbar_str = f'log_10({long_name}) [{units}]'
  else:
    iopts.cbar_str = f'{long_name} [{units}]'
if (iopts.title_right=='auto') and ('time' in ds[iopts.var].dims):
  tstr = str(data.time.data)
  #tstr = tstr.split('T')[0].replace('-', '')+'T'+tstr.split('T')[1].split('.')[0].replace(':','')+'Z'
  tstr = tstr.split('.')[0]
  iopts.title_right = tstr
if (iopts.title_center=='auto'):
  iopts.title_center = ''
if (iopts.xlabel=='auto'):
  try:
    iopts.xlabel = xdim.long_name
  except:
    iopts.xlabel = 'xlabel'
if (iopts.title_left=='auto') and (iopts.section!='auto'):
  iopts.title_left = iopts.section
#  iopts.title_left = f'depth = {data[depth_name].data:.1f}m'
#elif iopts.title_left=='auto':
#  iopts.title_left = ''

# -- start plotting
plt.close('all')
hca, hcb = arrange_axes(1,1, plot_cb=iopts.cbar_pos, asp=asp, fig_size_fac=2,
                             sharex=True, sharey=True, xlabel="", ylabel="",
                             axlab_kw=None, dfigr=0.5,
                            )
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade_kwargs = dict(ax=ax, cax=cax, 
                    logplot=iopts.logplot, 
                    clim=iopts.clim, 
                    cmap=iopts.cmap, 
                    cincr=iopts.cincr,
                    clevs=iopts.clevs,
                    conts=iopts.conts,
                    contfs=iopts.contfs,
                   )
hm = shade(xdim, ydim, data.data, **shade_kwargs)

if iopts.clabel:
  Cl = ax.clabel(hm[1], colors='k', fontsize=6, fmt='%.1f', inline=False)
  for txt in Cl:
    txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))

if iopts.cbar_pos=='bottom':
  cax.set_xlabel(iopts.cbar_str)
else:
  cax.set_ylabel(iopts.cbar_str)
ht = ax.set_title(iopts.title_right, loc='right')
ht = ax.set_title(iopts.title_center, loc='center')
ht = ax.set_title(iopts.title_left, loc='left')

ax.set_xlabel(iopts.xlabel)
ax.set_ylabel(iopts.ylabel)

ax.set_facecolor(iopts.facecolor)

if not iopts.xlim is None:
  ax.set_xlim(iopts.xlim)
if not iopts.ylim is None:
  ax.set_ylim(iopts.ylim)

if iopts.invert_yaxis:
  ax.invert_yaxis()

if iopts.fpath_fig!='none':
  if iopts.fpath_fig.startswith('~'):
    home = str(Path.home())+'/'
    iopts.fpath_fig = home + iopts.fpath_fig[1:]
  print(f'Saving figure {iopts.fpath_fig}...')
  plt.savefig(iopts.fpath_fig)

if not iopts.dontshow:
  plt.show()
