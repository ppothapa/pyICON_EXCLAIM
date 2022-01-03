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
parser.add_argument('--use_tgrid', default=False,
                    action='store_true',
                    help='If specified, the plot is made on the original triangular grid.')
parser.add_argument('--gname', type=str, default='auto',
                    help='Grid name of the ICON data.')
parser.add_argument('--it', type=int, default=0,
                    help='Time index which should be plotted.')
parser.add_argument('--time', type=str, default='none',
                    help='Time string \'yyyy-mm-dd\' wich should be plotted (if specified overwrites \'it\').')
parser.add_argument('--iz', type=int, default=0,
                    help='Depth index which should be plotted.')
parser.add_argument('--depth', type=float, default=-1.,
                    help='Depth value in m which should be plotted (if specified overwrites \'iz\').')
parser.add_argument('--projection', type=str, default='pc',
                    help='Map projection, choose \'None\' to deactivate cartopy, \'pc\' for normal lon/lat projection and \'np\' or \'sp\' for Norh- South-pole stereographic projections.')
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
parser.add_argument('--title_center', type=str, default='auto',
                    help='Title string center.')
parser.add_argument('--title_left', type=str, default='auto',
                    help='Title string left.')
parser.add_argument('--title_right', type=str, default='auto',
                    help='Title string right.')
parser.add_argument('--xlabel', type=str, default='',
                    help='String for xlabel.')
parser.add_argument('--ylabel', type=str, default='',
                    help='String for ylabel.')
parser.add_argument('--cbar_str', type=str, default='auto',
                    help='String for colorbar. Default is name of variable and its units.')
parser.add_argument('--cbar_pos', type=str, default='bottom',
                    help='Position of colorbar. It is possible to choose between \'right\' and \'bottom\'.')
parser.add_argument('--coastlines_color', type=str, default='k',
                    help='Color of coastlines. Default is \'k\'. To disable set to \'none\'.')
parser.add_argument('--land_facecolor', type=str, default='0.7',
                    help='Color of land masses. Default is \'0.7\'. To disable set to \'none\'.')
parser.add_argument('--lonlat_for_mask', default=False,
                    action='store_true',
                    help='If specified, mask for triangles which are swapped at periodic boundaries is calculated from clon and clat (and not only from clon). Relevant for torus setup.')
parser.add_argument('--logplot', default=False,
                    action='store_true',
                    help='Plot logarithm of the data.')

iopts = parser.parse_args()

fpath_data = iopts.fpath_data
var = iopts.var
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
interp_to_rectgrid = pyic.interp_to_rectgrid
interp_to_rectgrid_xr = pyic.interp_to_rectgrid_xr
triangulation = pyic.triangulation

# --- variable replacements
if var=='auto':
  var = list(ds.keys())[-1]

# --- limits
if clim!='auto':
  clim = clim.replace(' ', '')
  clim = np.array(clim.split(','), dtype=float)
if lon_reg:
  lon_reg = lon_reg.replace(' ', '')
  lon_reg = np.array(lon_reg.split(','), dtype=float)
if lat_reg:
  lat_reg = lat_reg.replace(' ', '')
  lat_reg = np.array(lat_reg.split(','), dtype=float)

# --- projections
do_xyticks = True
if isinstance(projection, str) and projection=='pc':
  ccrs_proj = ccrs.PlateCarree()
  shade_proj = ccrs.PlateCarree()
elif isinstance(projection, str) and projection=='np':
  ccrs_proj = ccrs.NorthPolarStereo()
  shade_proj = ccrs.PlateCarree()
  do_xyticks = False
  if lon_reg==None:
    lon_reg = [-180, 180]
  if lat_reg==None:
    lat_reg = [60, 90]
  extent = [lon_reg[0], lon_reg[1], lat_reg[0], lat_reg[1]]
  lat_reg[0] += -15 # increase data range to avoid white corners
elif isinstance(projection, str) and projection=='sp':
  ccrs_proj = ccrs.SouthPolarStereo()
  shade_proj = ccrs.PlateCarree()
  do_xyticks = False
  if lon_reg==None:
    lon_reg = [-180, 180]
  if lat_reg==None:
    lat_reg = [-90, -50]
  extent = [lon_reg[0], lon_reg[1], lat_reg[0], lat_reg[1]]
  lat_reg[1] += 15 # increase data range to avoid white corners
else:
  ccrs_proj = None
  shade_proj = None

# --- grid files and interpolation
path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'
if isinstance(fpath_data, list):
  fpath = fpath_data[0]
else:
  fpath = fpath_data
if gname=='auto':
  try:
    Dgrid = pyic.identify_grid(path_grid, fpath)
    gname = Dgrid['name']
  except:
    gname = 'none'
if fpath_tgrid=='auto':
  try:
    Dgrid = pyic.identify_grid(path_grid, fpath)
    fpath_tgrid = Dgrid['fpath_grid']
  except:
    fpath_tgrid = 'from_file'
#fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res{res:3.2f}_180W-180E_90S-90N.npz'
fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res{res:3.2f}_180W-180E_90S-90N.nc'

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

if var in ['mld', 'mlotst']:
  data = data.where(data!=data.min())

# --- aspect ratio of the plot
if (lon_reg is None) or (lat_reg is None):
  asp = 0.5
else:
  asp = (lat_reg[1]-lat_reg[0])/(lon_reg[1]-lon_reg[0])

if projection in ['np', 'sp']:
  asp = 1.

# --- interpolate and cut to region
if not use_tgrid:
  try:
    datai = interp_to_rectgrid_xr(data, fpath_ckdtree, lon_reg=lon_reg, lat_reg=lat_reg)
    lon = datai.lon
    lat = datai.lat
  except:
    lon, lat, datai = interp_to_rectgrid(data, fpath_ckdtree, lon_reg=lon_reg, lat_reg=lat_reg)
else:
  print('Deriving triangulation object, this can take a while...')
    
  if fpath_tgrid != 'from_file':
    ds_tgrid = xr.open_dataset(fpath_tgrid)
  else:
    ds_tgrid = xr.Dataset()
    ntr = ds.clon.size
    vlon = ds.clon_bnds.data.reshape(ntr*3)
    vlat = ds.clat_bnds.data.reshape(ntr*3)
    vertex_of_cell = np.arange(ntr*3).reshape(ntr,3)
    vertex_of_cell = vertex_of_cell.transpose()+1
    ds_tgrid['clon'] = xr.DataArray(ds.clon.data, dims=['cell'])
    ds_tgrid['clat'] = xr.DataArray(ds.clat.data, dims=['cell'])
    ds_tgrid['vlon'] = xr.DataArray(vlon, dims=['vertex'])
    ds_tgrid['vlat'] = xr.DataArray(vlat, dims=['vertex'])
    ds_tgrid['vertex_of_cell'] = xr.DataArray(vertex_of_cell, dims=['nv', 'cell'])

  if iopts.lonlat_for_mask:
    only_lon = False
  else:
    only_lon = True
  ind_reg, Tri = pyic.triangulation(ds_tgrid, lon_reg, lat_reg, only_lon=only_lon)

  if lon_reg is not None and lat_reg is not None:
    data = data[ind_reg]
  data = data.compute()
  print('Done deriving triangulation object.')

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
if (iopts.title_right=='auto') and ('time' in ds[var].dims):
  tstr = str(data.time.data)
  #tstr = tstr.split('T')[0].replace('-', '')+'T'+tstr.split('T')[1].split('.')[0].replace(':','')+'Z'
  tstr = tstr.split('.')[0]
  iopts.title_right = tstr
if (iopts.title_center=='auto'):
  iopts.title_center = ''
if (iopts.title_left=='auto') and (depth_name!='none'):
  iopts.title_left = f'depth = {data[depth_name].data:.1f}m'
elif iopts.title_left=='auto':
  iopts.title_left = ''

# -- start plotting
plt.close('all')
hca, hcb = arrange_axes(1,1, plot_cb=iopts.cbar_pos, asp=asp, fig_size_fac=2,
                             sharex=True, sharey=True, xlabel="", ylabel="",
                             projection=ccrs_proj, axlab_kw=None, dfigr=0.5,
                            )
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
shade_kwargs = dict(ax=ax, cax=cax, clim=clim, projection=shade_proj, cmap=cmap, logplot=iopts.logplot)
if not use_tgrid:
  hm = shade(lon, lat, datai.data, **shade_kwargs)
else:
  hm = shade(Tri, data.data, **shade_kwargs)

if iopts.cbar_pos=='bottom':
  cax.set_xlabel(iopts.cbar_str)
else:
  cax.set_ylabel(iopts.cbar_str)
ht = ax.set_title(iopts.title_right, loc='right')
ht = ax.set_title(iopts.title_center, loc='center')
ht = ax.set_title(iopts.title_left, loc='left')

ax.set_xlabel(iopts.xlabel)
ax.set_ylabel(iopts.ylabel)

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

if projection in ['np', 'sp']: 
   ax.set_extent(extent, ccrs.PlateCarree())
   ax.gridlines()
   ax.add_feature(cartopy.feature.LAND)
   ax.coastlines()
else:
  if (lon_reg is None) and (lat_reg is None):
    plot_settings(ax, template='global', 
                  do_xyticks=do_xyticks, 
                  land_facecolor=iopts.land_facecolor, 
                  coastlines_color=iopts.coastlines_color,
    )
  else:
    plot_settings(ax, xlim=xlim, ylim=ylim, 
                  do_xyticks=do_xyticks,
                  land_facecolor=iopts.land_facecolor, 
                  coastlines_color=iopts.coastlines_color,
    )

if fpath_fig!='none':
  if fpath_fig.startswith('~'):
    home = str(Path.home())+'/'
    fpath_fig = home + fpath_fig[1:]
  print(f'Saving figure {fpath_fig}...')
  plt.savefig(fpath_fig)

if not iopts.dontshow:
  plt.show()
