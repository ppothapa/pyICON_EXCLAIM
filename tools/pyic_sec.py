#!/usr/bin/env python

import argparse

help_text = """
Makes a figure from ICON data

Usage notes:
------------
Basic usage:
pyic_fig netcdf_file.nc var_name [options]

Change color limits, colorbar:
pyic_fig netcdf_file.nc var_name --clim=-10,32 --cmap=viridis

Select time step and depth level by indices:
pyic_fig netcdf_file.nc var_name --it=3 --iz=0

Select date and depth:
pyic_fig netcdf_files_*.nc var_name --time=2010-03-02 --depth=1000

Change region:
pyic_fig netcdf_file.nc var_name --lon_reg=-20,30 --lat_reg=-45,-20

Plot on original triangle grid (it is recommended to cut the domain otherwise, it takes a long time):
pyic_fig netcdf_file.nc var_name --use_tgrid --lon_reg=-72,-68 --lat_reg=33,35

Change projection to North Polar Stereographic projection:
pyic_fig netcdf_file.nc var_name --projection=np

Save the figure:
pyic_fig netcdf_file.nc var_name --fpath_fig=/path/to/figure.png

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
parser.add_argument('--it', type=int, default=0,
                    help='Time index which should be plotted.')
parser.add_argument('--time', type=str, default='none',
                    help='Time string \'yyyy-mm-dd\' wich should be plotted (if specified overwrites \'it\').')
parser.add_argument('--xlim', type=str, default=None,
                    help='Limits for x-axis.')
parser.add_argument('--ylim', type=str, default=None,
                    help='Limits for y-axis.')
parser.add_argument('--cmap', type=str, default='auto',
                    help='Colormap used for plot.')
parser.add_argument('--clim', type=str, default='auto',
                    help='Color limits of the plot. Either specify one or two values.If one value is specified color limits are taken symetrically around zero. If \'auto\' is specified color limits are derived automatically.')
parser.add_argument('--facecolor', type=str, default=None,
                    help='Background color')
parser.add_argument('--title_center', type=str, default='auto',
                    help='Title string center.')
parser.add_argument('--title_left', type=str, default=None,
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

# --- limits
if iopts.clim!='auto':
  iopts.clim = iopts.clim.replace(' ', '')
  iopts.clim = np.array(iopts.clim.split(','), dtype=float)
if iopts.xlim:
  iopts.xlim = iopts.xlim.replace(' ', '')
  iopts.xlim = np.array(iopts.xlim.split(','), dtype=float)
if iopts.ylim:
  iopts.ylim = iopts.ylim.replace(' ', '')
  iopts.ylim = np.array(iopts.ylim.split(','), dtype=float)

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

if iopts.factor:
  data *= iopts.factor
data = data.squeeze()
xdim = data[data.dims[1]]
ydim = data[data.dims[0]]

# --- aspect ratio of the plot
asp = 0.5

# --- title, colorbar, and x/y label  strings
if iopts.cbar_str=='auto':
  try:
    units = data.units
  except:
    units = 'NA'
  if iopts.logplot:
    iopts.cbar_str = f'log_10({data.long_name}) [{units}]'
  else:
    iopts.cbar_str = f'{data.long_name} [{units}]'
if (iopts.title_right=='auto') and ('time' in ds[iopts.var].dims):
  tstr = str(data.time.data)
  #tstr = tstr.split('T')[0].replace('-', '')+'T'+tstr.split('T')[1].split('.')[0].replace(':','')+'Z'
  tstr = tstr.split('.')[0]
  iopts.title_right = tstr
if (iopts.title_center=='auto'):
  iopts.title_center = ''
if (iopts.xlabel=='auto'):
  iopts.xlabel = xdim.long_name
#if (iopts.title_left=='auto') and (depth_name!='none'):
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
shade_kwargs = dict(ax=ax, cax=cax, clim=iopts.clim, cmap=iopts.cmap, logplot=iopts.logplot)
hm = shade(xdim, ydim, data.data, **shade_kwargs)

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
