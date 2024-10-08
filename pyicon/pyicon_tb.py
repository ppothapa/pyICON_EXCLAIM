print('sys')
import sys, glob, os
print('json')
import json
# --- calculations
print('numpy')
import numpy as np
print('scipy')
from scipy import interpolate
from scipy.spatial import cKDTree
# --- reading data 
print('netcdf datetime')
from netCDF4 import Dataset, num2date, date2num
import datetime
# --- plotting
print('matplotlib')
import matplotlib.pyplot as plt
import matplotlib
# --- debugging
print('mybreak')
#from ipdb import set_trace as mybreak  
print('pnadas')
import pandas as pd
print('xarray')
import xarray as xr
print('done xarray')

"""
pyicon
#  icon_to_regular_grid
#  icon_to_section
  apply_ckdtree
  ckdtree_hgrid
  ckdtree_section
  calc_ckdtree
  haversine_dist
  derive_section_points
  timing
  conv_gname
  identify_grid
  crop_tripolar_grid
  crop_regular_grid
  get_files_of_timeseries
  get_varnames
  get_timesteps

  ?load_data
  ?load_grid

  ?hplot
  ?update_hplot
  ?vplot
  ?update_vplot

  #IconDataFile

  IconData
  IP_hor_sec_rect

  QuickPlotWebsite

  IDa: Icon data set (directory of files)
    - info about tsteps
    - info about vars
    - info about grid
    - IGr: Icon grid
    - IVa: Icon variable if loaded
  IIn: Icon interpolator class

  IPl: Icon plot class

IDa = pyic.IconData(fpath or path)
IDa.load_grid()
IDa.show()

IPl = pyic.hplot(IDa, 'var', iz, tstep, IIn)

"""

class pyicon_configure(object):
  def __init__(self, fpath_config):
    with open(fpath_config) as file_json:
      Dsettings = json.load(file_json)
    for key in Dsettings.keys():
      setattr(self, key, Dsettings[key])
    return

#def icon_to_regular_grid(data, shape, distances=None, \
#                  inds=None, radius_of_influence=1000e3):
#  """
#  """
#  data_interpolated = apply_ckdtree(data, distances=distances, inds=inds, 
#                                    radius_of_influence=radius_of_influence)
#  data_interpolated = data_interpolated.reshape(shape)
#  return data_interpolated
#
#def icon_to_section(data, distances=None, \
#                  inds=None, radius_of_influence=1000e3):
#  """
#  """
#  data_interpolated = apply_ckdtree(data, distances=distances, inds=inds, 
#                                    radius_of_influence=radius_of_influence)
#  return data_interpolated

"""
Routines to apply interpolation weights
"""
def apply_ckdtree_base(data, inds, distances, radius_of_influence=1000e3):
  if distances.ndim == 1:
    #distances_ma = np.ma.masked_greater(distances, radius_of_influence)
    if data.ndim==1:
      if isinstance(data, xr.core.dataarray.DataArray):
        data_interpolated = data.load()[inds]
      else:
        data_interpolated = data[inds]
      data_interpolated[distances>=radius_of_influence] = np.nan
    elif data.ndim==2:
      if isinstance(data, xr.core.dataarray.DataArray):
        data_interpolated = data.load()[:,inds]
      else:
        data_interpolated = data[:,inds]
      data_interpolated[:,distances>=radius_of_influence] = np.nan
  else:
    #raise ValueError("::: distances.ndim>1 is not properly supported yet. :::")
    #distances_ma = np.ma.masked_greater(distances, radius_of_influence)
    weights = 1.0 / distances**2
    if data.ndim==1:
      data_interpolated = np.ma.sum(weights * data[inds], axis=1) / np.ma.sum(weights, axis=1)
      #data_interpolated[distances>=radius_of_influence] = np.nan
    elif data.ndim==2:
      data_interpolated = np.ma.sum(weights[np.newaxis,:,:] * data[:,inds], axis=2) / np.ma.sum(weights[np.newaxis,:,:], axis=2)
      #data_interpolated[:,distances>=radius_of_influence] = np.nan
  data_interpolated = np.ma.masked_invalid(data_interpolated)
  return data_interpolated

def apply_ckdtree(data, fpath_ckdtree, mask=None, coordinates='clat clon', radius_of_influence=1000e3):
  """
  * credits
    function modified from pyfesom (Nikolay Koldunov)
  """
  if fpath_ckdtree.endswith('npz'):
    #print("::: Warning: using npz ckdtree files is outdated, use nc-files instead! :::")
    ddnpz = np.load(fpath_ckdtree)
  else:
    ddnpz = Dataset(fpath_ckdtree, 'r')
  #if coordinates=='clat clon':
  if ('clon' in coordinates) or (coordinates==''):
    distances = ddnpz['dckdtree_c']
    inds = ddnpz['ickdtree_c'] 
  #elif coordinates=='elat elon':
  elif 'elon' in coordinates:
    distances = ddnpz['dckdtree_e']
    inds = ddnpz['ickdtree_e'] 
  #elif coordinates=='vlat vlon':
  elif 'vlon' in coordinates:
    distances = ddnpz['dckdtree_v']
    inds = ddnpz['ickdtree_v'] 
  else:
    raise ValueError('::: Error: Unsupported coordinates: %s! ::: ' % (coordinates))
  if fpath_ckdtree.endswith('nc'):
    distances = distances[:]
    inds = inds[:]
    ddnpz.close()

  if mask is not None:
    #if data.ndim==1:
    #  data = data[mask]
    #elif data.ndim==2:
    #  data = data[:,mask]
    if inds.ndim==1:
      inds = inds[mask]
      distances = distances[mask]
    elif inds.ndim==2:
      #raise ValueError('::: Warning: This was never checked! Please check carefully and remove this warning.:::')
      inds = inds[:,mask]
      distances = distances[:,mask]

  data_interpolated = apply_ckdtree_base(data, inds, distances, radius_of_influence)
  return data_interpolated

def interp_to_rectgrid(data, fpath_ckdtree, 
                       lon_reg=None, lat_reg=None,             # for new way of cropping
                       indx='all', indy='all', mask_reg=None,  # for old way of cropping
                       coordinates='clat clon'):
  if fpath_ckdtree.endswith('npz'):
    #print("::: Warning: using npz ckdtree files is outdated, use nc-files instead! :::")
    ddnpz = np.load(fpath_ckdtree)
    lon = ddnpz['lon']
    lat = ddnpz['lat']
  else:
    f = Dataset(fpath_ckdtree, 'r')
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    f.close()
  # --- old way of cropping
  if not isinstance(indx, str):
    lon = lon[indx]
    lat = lat[indy]
  # --- prepare cropping the data to a region
  if lon_reg is not None:
    indx = np.where((lon>=lon_reg[0]) & (lon<lon_reg[1]))[0]
    indy = np.where((lat>=lat_reg[0]) & (lat<lat_reg[1]))[0]
    Lon, Lat = np.meshgrid(lon, lat) # full grid
    lon = lon[indx]
    lat = lat[indy]
    ind_reg = ((Lon>=lon_reg[0]) & (Lon<lon_reg[1]) & (Lat>=lat_reg[0]) & (Lat<lat_reg[1])).flatten()
    mask_reg = ind_reg
    Lon, Lat = np.meshgrid(lon, lat) # cropped grid
  datai = apply_ckdtree(data, fpath_ckdtree, mask=mask_reg, coordinates=coordinates)
  if datai.ndim==1:
    datai = datai.reshape(lat.size, lon.size)
  else:
    datai = datai.reshape([data.shape[0], lat.size, lon.size])
  datai[datai==0.] = np.ma.masked
  return lon, lat, datai

def interp_to_rectgrid_xr(arr, fpath_ckdtree='auto', 
                          lon_reg=None, lat_reg=None,
                          coordinates='clat clon',
                          radius_of_influence=1000e3,
                          compute=True,
                          mask_out_of_range=True,
                          mask_out_of_range_before=False,
                         ):

  # --- rename dimensions if necessary
  for dim in ['cell', 'edge', 'vertex', 'ncells_2', 'ncells_3']:
    if dim in arr.dims:
      arr = arr.rename({dim: 'ncells'})

  # --- get fpath_ckdtree if necessary
  if fpath_ckdtree=='auto':
    path_grid = '/work/mh0033/m300602/icon/grids/'
    Dgrid = identify_grid(path_grid, arr) 
    fpath_ckdtree = Dgrid['Drectgrids']['res0.30_180W-180E_90S-90N']
    print(fpath_ckdtree)

  # --- load interpolation indices
  ds_ckdt = xr.open_dataset(fpath_ckdtree)
  if ('clon' in coordinates) or (coordinates==''):
    inds = ds_ckdt.ickdtree_c
    dist = ds_ckdt.dckdtree_c
  elif 'elon' in coordinates:
    inds = ds_ckdt.ickdtree_e
    dist = ds_ckdt.dckdtree_e
  elif 'vlon' in coordinates:
    inds = ds_ckdt.ickdtree_v
    dist = ds_ckdt.dckdtree_v
  else:
    raise ValueError('::: Error: Unsupported coordinates: %s! ::: ' % (coordinates))
  lon = ds_ckdt.lon.compute().data
  lat = ds_ckdt.lat.compute().data
  if lon_reg is not None:
    indx = np.where((lon>=lon_reg[0]) & (lon<lon_reg[1]))[0]
    indy = np.where((lat>=lat_reg[0]) & (lat<lat_reg[1]))[0]
    lon = lon[indx]
    lat = lat[indy]
    dist = dist.isel(lon=indx, lat=indy)
    inds = inds.isel(lon=indx, lat=indy)
    
  #if lon_reg is not None:
  #  indx = np.where((lon>=lon_reg[0]) & (lon<lon_reg[1]))[0]
  #  indy = np.where((lat>=lat_reg[0]) & (lat<lat_reg[1]))[0]
  #  Lon, Lat = np.meshgrid(lon, lat) # full grid
  #  lon = lon[indx]
  #  lat = lat[indy]
  #  ind_reg = ((Lon>=lon_reg[0]) & (Lon<lon_reg[1]) & (Lat>=lat_reg[0]) & (Lat<lat_reg[1])).flatten()
  #  mask_reg = ind_reg
  #  Lon, Lat = np.meshgrid(lon, lat) # cropped grid
  dist = dist.compute()
  inds = inds.compute().data.flatten()

  # --- interpolate by nearest neighbor
  arr_interp = arr.isel(ncells=inds)

  # --- reshape
  arr_interp = arr_interp.assign_coords(ncells=pd.MultiIndex.from_product([lat, lon], names=("lat", "lon"))
                                ).unstack()

  # --- mask values where nearest neighbor is too far away
  # (doing this after compute seems to be faster) FIXME check that!
  if mask_out_of_range_before:
    arr_interp = arr_interp.where(dist<radius_of_influence)

  # --- compute data otherwise a lazy object is returned
  if compute:
    arr_interp = arr_interp.compute()

  # --- mask values where nearest neighbor is too far away
  # (doing this after compute seems to be faster) FIXME check that!
  if mask_out_of_range:
    arr_interp = arr_interp.where(dist<radius_of_influence)

  return  arr_interp

def interp_to_section(data, fpath_ckdtree, coordinates='clat clon'):
  ddnpz = np.load(fpath_ckdtree)
  lon_sec = ddnpz['lon_sec'] 
  lat_sec = ddnpz['lat_sec'] 
  dist_sec = ddnpz['dist_sec'] 
  datai = apply_ckdtree(data, fpath_ckdtree, coordinates=coordinates)
  datai[datai==0.] = np.ma.masked
  return lon_sec, lat_sec, dist_sec, datai

""" 
Routines for zonal averaging
"""
def zonal_average(fpath_data, var, basin='global', it=0, fpath_fx='', fpath_ckdtree=''):

  for fp in [fpath_data, fpath_fx, fpath_ckdtree]:
    if not os.path.exists(fp):
      raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  f = Dataset(fpath_fx, 'r')
  basin_c = f.variables['basin_c'][:]
  mask_basin = np.zeros(basin_c.shape, dtype=bool)
  if basin.lower()=='atlantic' or basin=='atl':
    mask_basin[basin_c==1] = True 
  elif basin.lower()=='pacific' or basin=='pac':
    mask_basin[basin_c==3] = True 
  elif basin.lower()=='southern ocean' or basin=='soc' or basin=='so':
    mask_basin[basin_c==6] = True 
  elif basin.lower()=='indian ocean' or basin=='ind' or basin=='io':
    mask_basin[basin_c==7] = True 
  elif basin.lower()=='global' or basin=='glob' or basin=='glo':
    mask_basin[basin_c!=0] = True 
  elif basin.lower()=='indopacific' or basin=='indopac':
    mask_basin[(basin_c==3) | (basin_c==7)] = True 
  elif basin.lower()=='indopacso':
    mask_basin[(basin_c==3) | (basin_c==7) | (basin_c==6)] = True 
  f.close()
  
  ddnpz = np.load(fpath_ckdtree)
  lon = ddnpz['lon'] 
  lat = ddnpz['lat'] 
  shape = [lat.size, lon.size]
  lat_sec = lat
  
  f = Dataset(fpath_data, 'r')
  nz = f.variables[var].shape[1]
  coordinates = f.variables[var].coordinates
  data_zave = np.ma.zeros((nz,lat_sec.size))
  for k in range(nz):
    #print('k = %d/%d'%(k,nz))
    # --- load data
    data = f.variables[var][it,k,:]
    # --- mask land points
    data[data==0] = np.ma.masked
    # --- mask not-this-basin points
    data[mask_basin==False] = np.ma.masked
    # --- go to normal np.array (not np.ma object)
    if isinstance(data, np.ma.core.MaskedArray):
      data = data.filled(0.)
    # --- interpolate to rectangular grid
    datai = apply_ckdtree(data, fpath_ckdtree, coordinates=coordinates)
    datai = datai.reshape(shape)
    # --- go back to masked array
    datai = np.ma.array(datai, mask=datai==0.)
    # --- do zonal average
    data_zave[k,:] = datai.mean(axis=1)
  f.close()
  return lat_sec, data_zave

def zonal_average_3d_data(data3d, basin='global', it=0, coordinates='clat clon', fpath_fx='', fpath_ckdtree=''):
  """ Like zonal_average but here data instead of path to data is given. This can only work if the whole data array fits into memory.
  """

  for fp in [fpath_fx, fpath_ckdtree]:
    if not os.path.exists(fp):
      raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  f = Dataset(fpath_fx, 'r')
  basin_c = f.variables['basin_c'][:]
  mask_basin = np.zeros(basin_c.shape, dtype=bool)
  if basin.lower()=='atlantic' or basin=='atl':
    mask_basin[basin_c==1] = True 
  elif basin.lower()=='pacific' or basin=='pac':
    mask_basin[basin_c==3] = True 
  elif basin.lower()=='southern ocean' or basin=='soc' or basin=='so':
    mask_basin[basin_c==6] = True 
  elif basin.lower()=='indian ocean' or basin=='ind' or basin=='io':
    mask_basin[basin_c==7] = True 
  elif basin.lower()=='global' or basin=='glob' or basin=='glo':
    mask_basin[basin_c!=0] = True 
  elif basin.lower()=='indopacific' or basin=='indopac':
    mask_basin[(basin_c==3) | (basin_c==7)] = True 
  elif basin.lower()=='indopacso':
    mask_basin[(basin_c==3) | (basin_c==7) | (basin_c==6)] = True 
  f.close()
  
  ddnpz = np.load(fpath_ckdtree)
  #dckdtree = ddnpz['dckdtree']
  #ickdtree = ddnpz['ickdtree'] 
  lon = ddnpz['lon'] 
  lat = ddnpz['lat'] 
  shape = [lat.size, lon.size]
  lat_sec = lat
  
  nz = data3d.shape[0]
  data_zave = np.ma.zeros((nz,lat_sec.size))
  for k in range(nz):
    data = 1.*data3d[k,:]
    #print('k = %d/%d'%(k,nz))
    # --- mask land points
    data[data==0] = np.ma.masked
    # --- mask not-this-basin points
    data[mask_basin==False] = np.ma.masked
    # --- go to normal np.array (not np.ma object)
    if isinstance(data, np.ma.core.MaskedArray):
      data = data.filled(0.)
    # --- interpolate to rectangular grid
    datai = apply_ckdtree(data, fpath_ckdtree, coordinates=coordinates)
    datai = datai.reshape(shape)
    # --- go back to masked array
    datai = np.ma.array(datai, mask=datai==0.)
    # --- do zonal average
    data_zave[k,:] = datai.mean(axis=1)
  return lat_sec, data_zave

def zonal_average_atmosphere(data3d, ind_lev, fac, fpath_ckdtree='', coordinates='clat clon',):
  icall = np.arange(data3d.shape[1],dtype=int)
  datavi = data3d[ind_lev,icall]*fac+data3d[ind_lev+1,icall]*(1.-fac)
  lon, lat, datavihi = interp_to_rectgrid(datavi, fpath_ckdtree, coordinates=coordinates)
  data_zave = datavihi.mean(axis=2)
  return lat, data_zave

def zonal_section_3d_data(data3d, fpath_ckdtree, coordinates):
  """
  (
   lon_sec, lat_sec, dist_sec, data_sec 
  ) = pyic.zonal_section_3d_data(tbias, 
    fpath_ckdtree=path_ckdtree+'sections/r2b4_nps100_30W80S_30W80N.npz')
  """
  # --- load ckdtree
  ddnpz = np.load(fpath_ckdtree)
  #dckdtree = ddnpz['dckdtree']
  #ickdtree = ddnpz['ickdtree'] 
  lon_sec = ddnpz['lon_sec'] 
  lat_sec = ddnpz['lat_sec'] 
  dist_sec = ddnpz['dist_sec'] 

  nz = data3d.shape[0]
  data_sec = np.ma.zeros((nz,dist_sec.size))
  for k in range(nz):
    data_sec[k,:] = apply_ckdtree(data3d[k,:], fpath_ckdtree, coordinates=coordinates)
  return lon_sec, lat_sec, dist_sec, data_sec

def lonlat2str(lon, lat):
  if lon<0:
    lon_s = '%gW'%(-lon)
  else:
    lon_s = '%gE'%(lon)
  if lat<0:
    lat_s = '%gS'%(-lat)
  else:
    lat_s = '%gN'%(lat)
  return lon_s, lat_s

"""
Routines to calculate interpolation weights:

  | ckdtree_hgrid
  | ckdtree_section
  |-->| ckdtree_points
      |--> calc_ckdtree
"""

def ckdtree_hgrid(lon_reg, lat_reg, res, 
                 #fpath_grid_triangular='', 
                 fname_tgrid='',
                 path_tgrid='',
                 path_ckdtree='',
                 sname='',
                 gname='',
                 tgname='',
                 load_cgrid=True,
                 load_egrid=True,
                 load_vgrid=True,
                 n_nearest_neighbours=1,
                 n_jobs=1,
                 save_as_nc=True,
                 ):
  """
  """
  if tgname=='':
    Drgrid = identify_grid(path_tgrid, path_tgrid+fname_tgrid) 
    tgname = Drgrid['name']
  lon1str, lat1str = lonlat2str(lon_reg[0], lat_reg[0])
  lon2str, lat2str = lonlat2str(lon_reg[1], lat_reg[1])

  if save_as_nc:
    ending='nc'
  else:
    ending='npz'

  if n_nearest_neighbours==1:
    fname = f'{tgname}_res{res:3.2f}_{lon1str}-{lon2str}_{lat1str}-{lat2str}.{ending}'
  else:
    fname = f'{tgname}_res{res:3.2f}_{n_nearest_neighbours:d}_{lon1str}-{lon2str}_{lat1str}-{lat2str}.{ending}'
  fpath_ckdtree = path_ckdtree+fname
  fpath_tgrid   = path_tgrid+fname_tgrid

  # --- make rectangular grid 
  lon = np.arange(lon_reg[0],lon_reg[1],res)
  lat = np.arange(lat_reg[0],lat_reg[1],res)
  Lon, Lat = np.meshgrid(lon, lat)

  lon_o = Lon.flatten()
  lat_o = Lat.flatten()
  
  # --- calculate ckdtree
  Dind_dist = ckdtree_points(fpath_tgrid, lon_o, lat_o, load_cgrid=load_cgrid, load_egrid=load_egrid, load_vgrid=load_vgrid,
                             n_nearest_neighbours=n_nearest_neighbours, n_jobs=n_jobs)

  # --- save grid
  print('Saving grid file: %s' % (fpath_ckdtree))
  if save_as_nc:
    ds = xr.Dataset(coords=dict(lat=lat, lon=lon))
    for var in Dind_dist.keys(): 
      ds[var] = xr.DataArray(Dind_dist[var].reshape(lat.size, lon.size), dims=['lat', 'lon'])
    ds.to_netcdf(fpath_ckdtree)
  else:
    np.savez(fpath_ckdtree,
              lon=lon,
              lat=lat,
              sname=sname,
              gname=gname,
              tgname='test',
              **Dind_dist,
             )
  return

def ckdtree_section(p1, p2, npoints=101, 
                 fname_tgrid='',
                 path_tgrid='',
                 path_ckdtree='',
                 sname='auto',
                 gname='',
                 tgname='',
                 n_nearest_neighbours=1,
                 n_jobs=1,
                 load_cgrid=True,
                 load_egrid=True,
                 load_vgrid=True,
                 save_as_nc=True,
                 ):
  """
  """
  if tgname=='':
    Drgrid = identify_grid(path_tgrid, path_tgrid+fname_tgrid) 
    tgname = Drgrid['name']
  lon1str, lat1str = lonlat2str(p1[0], p1[1])
  lon2str, lat2str = lonlat2str(p2[0], p2[1])

  if sname=='auto':
    sname = fpath_ckdtree.split('/')[-1][:-4]

  if save_as_nc:
    ending='nc'
  else:
    ending='npz'
  #fname = '%s_nps%d_%s%s_%s%s.npz'%(tgname, npoints, lon1str, lat1str, lon2str, lat2str) 
  fname = f'{tgname}_nps{npoints:d}_{lon1str}{lat1str}_{lon2str}{lat2str}.{ending}'
  fpath_ckdtree = path_ckdtree+fname
  fpath_tgrid   = path_tgrid+fname_tgrid

  # --- derive section points
  lon_sec, lat_sec, dist_sec = derive_section_points(p1, p2, npoints)
  lon_o = lon_sec
  lat_o = lat_sec

  # --- calculate ckdtree
  Dind_dist = ckdtree_points(fpath_tgrid, lon_o, lat_o, load_cgrid=load_cgrid, load_egrid=load_egrid, load_vgrid=load_vgrid, n_nearest_neighbours=n_nearest_neighbours,
                             n_jobs=n_jobs)

  # --- save grid
  print('Saving grid file: %s' % (fpath_ckdtree))
  if save_as_nc:
    ds = xr.Dataset()
    ds['lon_sec'] = xr.DataArray(lon_sec, dims=['isec'])
    ds['lat_sec'] = xr.DataArray(lat_sec, dims=['isec'])
    ds['dist_sec'] = xr.DataArray(dist_sec, dims=['isec'])
    for var in Dind_dist.keys(): 
      ds[var] = xr.DataArray(Dind_dist[var], dims=['isec'])
    ds.to_netcdf(fpath_ckdtree)
  else:
    np.savez(fpath_ckdtree,
              lon_sec=lon_sec,
              lat_sec=lat_sec,
              dist_sec=dist_sec,
              sname=sname,
              gname=gname,
              **Dind_dist
             )
  return Dind_dist['dckdtree_c'], Dind_dist['ickdtree_c'], lon_sec, lat_sec, dist_sec

def ckdtree_points(fpath_tgrid, lon_o, lat_o, load_cgrid=True, load_egrid=True, load_vgrid=True, n_nearest_neighbours=1, n_jobs=1):
  """
  """
  # --- load triangular grid
  f = Dataset(fpath_tgrid, 'r')
  if load_cgrid:
    clon = f.variables['clon'][:] * 180./np.pi
    clat = f.variables['clat'][:] * 180./np.pi
  if load_egrid:
    elon = f.variables['elon'][:] * 180./np.pi
    elat = f.variables['elat'][:] * 180./np.pi
  if load_vgrid:
    vlon = f.variables['vlon'][:] * 180./np.pi
    vlat = f.variables['vlat'][:] * 180./np.pi
  f.close()

  # --- ckdtree for cells, edges and vertices
  if load_cgrid:
    dckdtree_c, ickdtree_c = calc_ckdtree(lon_i=clon, lat_i=clat,
                                          lon_o=lon_o, lat_o=lat_o,
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          n_jobs=n_jobs,
                                          )
  if load_egrid:
    dckdtree_e, ickdtree_e = calc_ckdtree(lon_i=elon, lat_i=elat,
                                          lon_o=lon_o, lat_o=lat_o,
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          n_jobs=n_jobs,
                                          )
  if load_vgrid:
    dckdtree_v, ickdtree_v = calc_ckdtree(lon_i=vlon, lat_i=vlat,
                                          lon_o=lon_o, lat_o=lat_o,
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          n_jobs=n_jobs,
                                          )

  # --- save dict
  Dind_dist = dict()
  if load_cgrid: 
    Dind_dist['dckdtree_c'] = dckdtree_c
    Dind_dist['ickdtree_c'] = ickdtree_c
  if load_egrid: 
    Dind_dist['dckdtree_e'] = dckdtree_e
    Dind_dist['ickdtree_e'] = ickdtree_e
  if load_vgrid: 
    Dind_dist['dckdtree_v'] = dckdtree_v
    Dind_dist['ickdtree_v'] = ickdtree_v
  return Dind_dist

def calc_ckdtree(lon_i, lat_i, lon_o, lat_o, n_nearest_neighbours=1, n_jobs=1, use_npconcatenate=True):
  """
  """
  # --- do ckdtree
  if False:
    lzip_i = list(zip(lon_i, lat_i))
    tree = cKDTree(lzip_i)
    lzip_o = list(zip(lon_o, lat_o))
    dckdtree, ickdtree = tree.query(lzip_o , k=n_nearest_neighbours, n_jobs=1)
  else:
    #print('calc_ckdtree by cartesian distances')
    xi, yi, zi = spherical_to_cartesian(lon_i, lat_i)
    xo, yo, zo = spherical_to_cartesian(lon_o, lat_o)

    if not use_npconcatenate:
      lzip_i = list(zip(xi, yi, zi))
      lzip_o = list(zip(xo, yo, zo))
    else:
      # This option seems to be much faster but needs to be tested also for big grids
      lzip_i = np.concatenate((xi[:,np.newaxis],yi[:,np.newaxis],zi[:,np.newaxis]), axis=1)
      lzip_o = np.concatenate((xo[:,np.newaxis],yo[:,np.newaxis],zo[:,np.newaxis]), axis=1) 
    tree = cKDTree(lzip_i)
    #dckdtree, ickdtree = tree.query(lzip_o , k=n_nearest_neighbours, n_jobs=n_jobs)
    dckdtree, ickdtree = tree.query(lzip_o , k=n_nearest_neighbours)
  return dckdtree, ickdtree

def calc_vertical_interp_weights(zdata, levs, increases_along_axes=True):
  """ Calculate vertical interpolation weights and indices.

Call example:
icall, ind_lev, fac = calc_vertical_interp_weights(zdata, levs)

Afterwards do interpolation like this:
datai = data[ind_lev,icall]*fac+data[ind_lev+1,icall]*(1.-fac)
  """
  nza = zdata.shape[0]
  # --- initializations
  ind_lev = np.zeros((levs.size,zdata.shape[1]),dtype=int)
  icall = np.arange(zdata.shape[1],dtype=int)
  icall = icall[np.newaxis,:]
  fac = np.ma.zeros((levs.size,zdata.shape[1]))
  for k, lev in enumerate(levs):
    #print(f'k = {k}')
    # --- find level below critical level
    if increases_along_axes:
      ind_lev[k,:] = (zdata<levs[k]).sum(axis=0)-1
    else:
      ind_lev[k,:] = (zdata>levs[k]).sum(axis=0)-1
    ind_lev[k,ind_lev[k,:]==(nza-1)]=-1
    # --- zdata below and above lev 
    zd1 = zdata[ind_lev[k,:],icall]
    zd2 = zdata[ind_lev[k,:]+1,icall]
    # --- linear interpolation to get weight (fac=1 if lev=zd1)
    fac[k,:] = (0.-1.)/(zd2-zd1)*(levs[k]-zd1)+1.
  # --- mask values which are out of range
  fac[ind_lev==-1] = np.ma.masked 
  return icall, ind_lev, fac

"""
Routines to calculate grids and sections
"""

def derive_section_points(p1, p2, npoints=101,):
  # --- derive section points
  if p1[0]==p2[0]:
    lon_sec = p1[0]*np.ones((npoints)) 
    lat_sec = np.linspace(p1[1],p2[1],npoints)
  else:
    lon_sec = np.linspace(p1[0],p2[0],npoints)
    lat_sec = (p2[1]-p1[1])/(p2[0]-p1[0])*(lon_sec-p1[0])+p1[1]
  dist_sec = haversine_dist(lon_sec[0], lat_sec[0], lon_sec, lat_sec)
  return lon_sec, lat_sec, dist_sec

def calc_north_pole_interp_grid_points(lat_south=60., res=100e3):
  """
  Compute grid points optimized for plotting the North Pole area.

  Parameters:
  -----------
  lat_south : float
      Southern latitude of target grid.
  res : float
      resolution of target grid

  Returns:
  --------
  Lon_np, Lat_np: ndarray
      Longitude and latitude of target grid as 2d array.

  Examples:
  ---------
  Lon_np, Lat_np = calc_north_pole_interp_grid_points(lat_south=60., res=100e3)

  """
  R = 6371e3
  x1, y1, z1 = spherical_to_cartesian(  0., lat_south)
  x2, y2, z2 = spherical_to_cartesian( 90., lat_south)
  x3, y3, z3 = spherical_to_cartesian(180., lat_south)
  x4, y4, z4 = spherical_to_cartesian(270., lat_south)

  lon1, lat1 = cartesian_to_spherical(x1, y1, z1)
  lon2, lat2 = cartesian_to_spherical(x2, y2, z2)
  lon3, lat3 = cartesian_to_spherical(x3, y3, z3)
  lon4, lat4 = cartesian_to_spherical(x4, y4, z4)

  #x1 = R * np.cos(  0.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #y1 = R * np.sin(  0.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #z1 = R * np.sin(lat_south*np.pi/180.)
  #x2 = R * np.cos( 90.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #y2 = R * np.sin( 90.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #z2 = R * np.sin(lat_south*np.pi/180.)
  #x3 = R * np.cos(180.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #y3 = R * np.sin(180.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #z3 = R * np.sin(lat_south*np.pi/180.)
  #x4 = R * np.cos(270.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #y4 = R * np.sin(270.*np.pi/180.) * np.cos(lat_south*np.pi/180.)
  #z4 = R * np.sin(lat_south*np.pi/180.)
  #
  #lat1 = np.arcsin(z1/np.sqrt(x1**2+y1**2+z1**2)) * 180./np.pi
  #lon1 = np.arctan2(y1,x1) * 180./np.pi
  #lat2 = np.arcsin(z2/np.sqrt(x2**2+y2**2+z2**2)) * 180./np.pi
  #lon2 = np.arctan2(y2,x2) * 180./np.pi
  #lat3 = np.arcsin(z3/np.sqrt(x3**2+y3**2+z3**2)) * 180./np.pi
  #lon3 = np.arctan2(y3,x3) * 180./np.pi
  #lat4 = np.arcsin(z4/np.sqrt(x4**2+y4**2+z4**2)) * 180./np.pi
  #lon4 = np.arctan2(y4,x4) * 180./np.pi
  
  xnp = np.arange(x3, x1+res, res)
  ynp = np.arange(y4, y2+res, res)
  
  Xnp, Ynp = np.meshgrid(xnp, ynp)
  Znp = R * np.sin(lat1*np.pi/180.) * np.ones((ynp.size,xnp.size))
  Lon_np = np.arctan2(Ynp,Xnp) * 180./np.pi
  Lat_np = np.arcsin(Znp/np.sqrt(Xnp**2+Ynp**2+Znp**2)) * 180./np.pi
  return Lon_np, Lat_np

"""
Routines related to spherical geometry
"""
def haversine_dist(lon_ref, lat_ref, lon_pts, lat_pts, degree=True):
  # for details see http://en.wikipedia.org/wiki/Haversine_formula
  r = 6378.e3
  if degree:
    lon_ref = lon_ref * np.pi/180.
    lat_ref = lat_ref * np.pi/180.
    lon_pts = lon_pts * np.pi/180.
    lat_pts = lat_pts * np.pi/180.
  arg = np.sqrt(   np.sin(0.5*(lat_pts-lat_ref))**2 
                 + np.sin(0.5*(lon_pts-lon_ref))**2
                 * np.cos(lat_ref)*np.cos(lat_pts) )
  dist = 2*r * np.arcsin(arg)
  return dist

def spherical_to_cartesian(lon, lat):
  earth_radius = 6371e3
  x = earth_radius * np.cos(lon*np.pi/180.) * np.cos(lat*np.pi/180.)
  y = earth_radius * np.sin(lon*np.pi/180.) * np.cos(lat*np.pi/180.)
  z = earth_radius * np.sin(lat*np.pi/180.)
  return x, y, z

def cartesian_to_spherical(x, y, z):
  lat = np.arcsin(z/np.sqrt(x**2+y**2+z**2)) * 180./np.pi
  lon = np.arctan2(y,x) * 180./np.pi
  return lon, lat

"""
Routines to load data
"""
def load_hsnap(fpath, var, it=0, iz=0, iw=None, fpath_ckdtree='', verbose=True):
  f = Dataset(fpath, 'r')
  if verbose:
    print("Loading %s from %s" % (var, fpath))
  if f.variables[var].ndim==2:
    data = f.variables[var][it,:]
  else:
    data = f.variables[var][it,iz,:]
  if iw is not None:
    data = np.concatenate((data[:,iw:],data[:,:iw]),axis=1)
  f.close()

  data[data==0.] = np.ma.masked
  return data

def datetime64_to_float(dates):
  years  = dates.astype('datetime64[Y]').astype(int) + 1970
  months = int((str(dates.astype('datetime64[M]'))[5:7]))
  days   = int((str(dates.astype('datetime64[D]'))[8:10]))
  return years, months, days

def get_averaging_interval(times, output_freq, end_of_interval=True):
  """ Determine the length of the averaging time interval in seconds.

  times: numpy array containing numpy.datetime64 objects: time points which are associated with averaing interval
  output_freq: bool: pick either 'monthly', 'yearly', 'daily' or 'hourly'
  end_of_interal: bool: sets whether time points of times are interpreted at the end of the averaging interval (default ICON output) or at the beginning or in the middle (mkexp time shift)
  """
  dt = np.zeros((times.size))
  dt64type = times[0].dtype
  if output_freq=='yearly':
    for nn in range(times.size):
      yy, mm, dd = datetime64_to_float(times[nn])
      if end_of_interval:
        y1, y2 = yy-1, yy
      else:
        y1, y2 = yy, yy+1
      t1 = np.datetime64(f'{y1:04d}-01-01').astype(dt64type)
      t2 = np.datetime64(f'{y2:04d}-01-01').astype(dt64type)
      dt[nn] = t2-t1
      print(dt)
  elif output_freq=='monthly':
    for nn in range(times.size):
      yy, mm, dd = datetime64_to_float(times[nn])
      if end_of_interval:
        if mm==1:
          m1, y1 = 12, yy-1
          m2, y2 = 1,  yy
        else:
          m1, y1 = mm-1, yy
          m2, y2 = mm, yy
      else:
        if mm==12:
          m1, y1 = 12, yy
          m2, y2 = 1,  yy+1
        else:
          m1, y1 = mm,   yy
          m2, y2 = mm+1, yy
      t1 = np.datetime64(f'{y1:04d}-{m1:02d}-01').astype(dt64type)
      t2 = np.datetime64(f'{y2:04d}-{m2:02d}-01').astype(dt64type)
      dt[nn] = t2-t1
  elif output_freq=='daily':
    dt += 86400.
  elif output_freq=='hourly':
    dt += 3600.
  elif output_freq=='6hourly':
    dt += 21600.

  else:
    raise ValueError(f'::: Error: Unsupported output_freq = {output_freq}!:::')
  return dt

def time_average(IcD, var, t1='none', t2='none', it_ave=[], iz='all', always_use_loop=False, verbose=False, use_xr=False, load_xr_data=False, dimension_from_file='first'):
  it_ave = np.array(it_ave)
  # --- if no it_ave is given use t1 and t2 to determine averaging indices it_ave
  if it_ave.size==0:
    # --- if t2=='none' set t2=t1 and no time average will be applied
    if isinstance(t2, str) and t2=='none':
      t2 = t1

    # --- convert to datetime64 objects if necessary
    if isinstance(t1, str):
      t1 = np.datetime64(t1)
    if isinstance(t2, str):
      t2 = np.datetime64(t2)

    # --- determine averaging interval
    it_ave = np.where( (IcD.times>=t1) & (IcD.times<=t2) )[0]
  else:
    t1 = IcD.times[it_ave[0]]
    t2 = IcD.times[it_ave[-1]]

  if it_ave.size==0:
    raise ValueError(f'::: Could not find any time steps in interval t1={t1} and t2={t2}! :::')
  
  ## --- decide whether the file consists of monthly or yearly averages (or something else)
  #dt1 = (IcD.times[it_ave][1]-IcD.times[it_ave][0]).astype(float)/(86400)
  #if dt1==365 or dt1==366:
  #  ave_mode = 'yearly'
  #elif dt1==28 or dt1==29 or dt1==30 or dt1==31:
  #  ave_mode = 'monthly'
  #else:
  #  ave_mode = 'unknown'
       
  # --- determine the length of the time average interval (in seconds)
  #dt64type = IcD.times[0].dtype
  #time_bnds = IcD.times[it_ave]
  #yy, mm, dd = datetime64_to_float(time_bnds[0])
  #if t1!=t2:
  #  if IcD.output_freq=='yearly':
  #    time_bnds = np.concatenate(([np.datetime64(f'{yy-1:04d}-{mm:02d}-{dd:02d}').astype(dt64type)],time_bnds))
  #  elif IcD.output_freq=='monthly':
  #    if mm==1:
  #      yy += -1
  #      mm = 13
  #    time_bnds = np.concatenate(([np.datetime64(f'{yy:04d}-{mm-1:02d}-{dd:02d}').astype(dt64type)],time_bnds))
  #  elif IcD.output_freq=='unknown':
  #    time_bnds = np.concatenate(([time_bnds[0]-(time_bnds[1]-time_bnds[0])], time_bnds))
  #  dt = np.diff(time_bnds).astype(IcD.dtype)
  #else:
  #  # load single time instance
  #  dt = np.array([1])
  ##dt = np.ones((it_ave.size), dtype=IcD.dtype)
  ##print('Warning dt set to ones!!!')
  dt = get_averaging_interval(IcD.times[it_ave], IcD.output_freq, end_of_interval=IcD.time_at_end_of_interval)

  # --- get dimensions to allocate data
  if dimension_from_file=='first':
    dimension_from_file = IcD.flist_ts[0]
  elif dimension_from_file=='last':
    dimension_from_file = IcD.flist_ts[-1]
  f = Dataset(dimension_from_file, 'r')
  # FIXME: If == ('time', 'lat', 'lon') works well use it everywhere
  load_hfl_type = False
  load_moc_type = False
  if f.variables[var].dimensions == ('time', 'lat', 'lon'): # e.g. for heat fluxes
    nt, nc, nx = f.variables[var].shape
    nz = 0
    load_hfl_type = True
  elif f.variables[var].dimensions == ('time', 'depth', 'lat', 'lon'): # e.g. for MOC 
    nt, nz, nc, ndummy = f.variables[var].shape 
    load_moc_type = True
  elif f.variables[var].ndim==3:
    nt, nz, nc = f.variables[var].shape
  elif f.variables[var].ndim==2: # e.g. for 2D variables like zos and mld
    nt, nc = f.variables[var].shape
    nz = 0
  f.close()

  # --- set iz to all levels
  if isinstance(iz,str) and iz=='all':
    iz = np.arange(nz)
  #else:
  #  iz = np.array([iz])

  # --- if all data is coming from one file take faster approach
  fpaths = np.unique(IcD.flist_ts[it_ave])
  if use_xr:
    #print(dt)
    if load_hfl_type:
      data_ave = (IcD.ds[var][it_ave,:,0]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif load_moc_type:
      data_ave = (IcD.ds[var][it_ave,:,:,0]*dt[:,np.newaxis,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif nz>0 and isinstance(iz,(int,np.integer)): # data has no depth dim afterwards
      #data_ave = (IcD.ds[var][it_ave,iz,:]*dt[:,np.newaxis]).sum(axis=0)/dt.sum()
      data_ave = (IcD.ds[var][it_ave,iz,:]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif nz>0 and not isinstance(iz,(int,np.integer)): # data has depth dim afterwards
      data_ave = (IcD.ds[var][it_ave,iz,:]*dt[:,np.newaxis,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    else:
      data_ave = (IcD.ds[var][it_ave,:]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    #dataxr = dsxr[var][it_ave,:,:].mean(axis=0)
    if load_xr_data:
      data_ave = data_ave.load().data
  elif (fpaths.size==1) and not always_use_loop:
    f = Dataset(fpaths[0], 'r')
    if load_hfl_type:
      data_ave = (f.variables[var][IcD.its[it_ave],:,0]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif load_moc_type:
      data_ave = (f.variables[var][IcD.its[it_ave],:,:,0]*dt[:,np.newaxis,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif nz>0 and isinstance(iz,(int,np.integer)): # data has no depth dim afterwards
      data_ave = (f.variables[var][IcD.its[it_ave],iz,:]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    elif nz>0 and not isinstance(iz,(int,np.integer)): # data has depth dim afterwards
      data_ave = (f.variables[var][IcD.its[it_ave],iz,:]*dt[:,np.newaxis,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    else:
      data_ave = (f.variables[var][IcD.its[it_ave],:]*dt[:,np.newaxis]).sum(axis=0, dtype='float64')/dt.sum()
    f.close()
  # --- otherwise loop ovar all files is needed
  else:
    # --- allocate data
    if isinstance(iz,(int,np.integer)) or nz==0:
      data_ave = np.ma.zeros((nc), dtype=IcD.dtype)
    else:
      data_ave = np.ma.zeros((iz.size,nc), dtype=IcD.dtype)

    # --- average by looping over all files and time steps
    for ll, it in enumerate(it_ave):
      f = Dataset(IcD.flist_ts[it], 'r')
      if load_hfl_type:
        data_ave += f.variables[var][IcD.its[it],:,0]*dt[ll]/dt.sum()
      elif load_moc_type:
        data_ave += f.variables[var][IcD.its[it],:,:,0]*dt[ll]/dt.sum()
      elif nz>0:
        data_ave += f.variables[var][IcD.its[it],iz,:]*dt[ll]/dt.sum()
      else:
        data_ave += f.variables[var][IcD.its[it],:]*dt[ll]/dt.sum()
      f.close()
  data_ave = data_ave.astype(IcD.dtype)
  if verbose:
    #print(f'pyicon.time_average: var={var}: it_ave={it_ave}')
    print(f'pyicon.time_average: var={var}: it_ave={IcD.times[it_ave]}')
  return data_ave, it_ave

def timing(ts, string='', verbose=True):
  if ts[0]==0:
    ts = np.array([datetime.datetime.now()])
  else:
    ts = np.append(ts, [datetime.datetime.now()])
    if verbose:
      print(ts[-1]-ts[-2], ' ', (ts[-1]-ts[0]), ' '+string)
  return ts

def conv_gname(gname):
  gname = gname[:-4]

  ogrid = gname.split('_')[0]
  res = float(gname.split('_')[1][1:])

  lo1 = gname.split('_')[2]
  if lo1[-1]=='w':
    lo1 = -float(lo1[:-1])
  else:
    lo1 = float(lo1[:-1])
  lo2 = gname.split('_')[3]
  if lo2[-1]=='w':
    lo2 = -float(lo2[:-1])
  else:
    lo2 = float(lo2[:-1])

  la1 = gname.split('_')[4]
  if la1[-1]=='s':
    la1 = -float(la1[:-1])
  else:
    la1 = float(la1[:-1])
  la2 = gname.split('_')[5]
  if la2[-1]=='s':
    la2 = -float(la2[:-1])
  else:
    la2 = float(la2[:-1])

  lon_reg = [lo1, lo2]
  lat_reg = [la1, la2]
  return ogrid, res, lon_reg, lat_reg

"""
Grid related functions
"""
def identify_grid(path_grid, fpath_data):
  """ Identifies ICON grid from fpath_data which can be either a path to 
  a file containing a 'clon' variable or an xarray dataset or array where
  the cell dimension name is either 'ncells' or 'cell'.
  
  
  r2b4:  160km:    15117: OceanOnly_Icos_0158km_etopo40.nc
  r2b4a: 160km:    20480: /pool/data/ICON/grids/public/mpim/0013/icon_grid_0013_R02B04_G.nc
  r2b6:   40km:   327680: OCEANINP_pre04_LndnoLak_039km_editSLOHH2017_G.nc
  r2b8:   10km:  3729001: OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min.nc
  r2b9:    5km: 14886338: OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min.nc
  r2b9a:   5km: 20971520: /pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc
  """
  
  Dgrid_list = dict()

  grid_name = 'r2b3_atm_r0030'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '320km'
  Dgrid_list[grid_name]['long_name'] = 'icon_grid_0030_R02B03_G'
  Dgrid_list[grid_name]['size'] = 5120
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b4_oce_r0003'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '160km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Icos_0158km_etopo40'
  Dgrid_list[grid_name]['size'] = 15117
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'
  
  grid_name = 'r2b4_oce_r0004'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '160km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Icos_0158km_etopo40'
  Dgrid_list[grid_name]['size'] = 15105
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'
 
  grid_name = 'r2b4_atm_r0013'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '160km'
  Dgrid_list[grid_name]['long_name'] = 'icon_grid_0013_R02B04_G'
  Dgrid_list[grid_name]['size'] = 20480
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b6old'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '40km'
  Dgrid_list[grid_name]['long_name'] = 'OCEANINP_pre04_LndnoLak_039km_editSLOHH2017_G'
  Dgrid_list[grid_name]['size'] = 327680
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'
  
  grid_name = 'r2b6_oce_r0004'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '40km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 235403 
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b8_oce_r0004'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '10km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 3729001
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'
  
  grid_name = 'r2b9_oce_r0004'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '5km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 14886338
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b9_atm_r0015'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '5km'
  Dgrid_list[grid_name]['long_name'] = 'icon_grid_0015_R02B09_G'
  Dgrid_list[grid_name]['size'] = 20971520
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b10_oce'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '2.5km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 59359799
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b10_atm_r0039'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '2.5km'
  Dgrid_list[grid_name]['long_name'] = 'icon_grid_0015_R02B09_G.nc'
  Dgrid_list[grid_name]['size'] = 20971520
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'r2b11_oce'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '1.25km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 237316813 
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'smt'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '500m-11km'
  Dgrid_list[grid_name]['long_name'] = 'SMT NATL grid'
  Dgrid_list[grid_name]['size'] = 59799625
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'smtwv_oce_2018'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '500m-11km'
  Dgrid_list[grid_name]['long_name'] = 'SMT WAVE grid'
  Dgrid_list[grid_name]['size'] = 66859144
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'smtwv_oce_2022'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '500m-11km'
  Dgrid_list[grid_name]['long_name'] = 'SMT WAVE grid'
  Dgrid_list[grid_name]['size'] = 66877252
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'

  grid_name = 'icon_244_0158km60'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = 'XXXkm'
  Dgrid_list[grid_name]['long_name'] = 'icon_244_0158km60'
  Dgrid_list[grid_name]['size'] = 862265
  #Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  Dgrid_list[grid_name]['fpath_grid'] = f'{path_grid}/{grid_name}/{grid_name}_tgrid.nc'
  
  if isinstance(fpath_data, str):
    try:
      f = Dataset(fpath_data, 'r')
      gsize = f.variables['clon'].size
      f.close()
    except:
      ds = xr.open_dataset(fpath_data)
      gsize = ds.ncells.size
  else:
    # assume fpath_data is an xarray dataset or array
    try:
      gsize = fpath_data.ncells.size
    except:
      try:
        gsize = fpath_data.cell.size
      except:
        raise ValueError('::: Error: Could not read numer of cells from fpath_data! :::')

  for grid_name in Dgrid_list.keys():
    if gsize == Dgrid_list[grid_name]['size']:
      Dgrid = Dgrid_list[grid_name]
      break

  # --- get all available interpolation files
  glist = glob.glob(f"{path_grid}/{Dgrid['name']}/ckdtree/rectgrids/*.nc")
  glist.sort()
  Drectgrids = dict()
  for gg in glist:
    key = gg.split('.nc')[0].split(Dgrid['name']+'_')[1]
    Drectgrids[key] = gg
  Dgrid['Drectgrids'] = Drectgrids
  glist = glob.glob(f"{path_grid}/{Dgrid['name']}/ckdtree/sections/*.nc")
  glist.sort()
  Dsections = dict()
  for gg in glist:
    key = gg.split('.nc')[0].split(Dgrid['name']+'_')[1]
    Dsections[key] = gg
  Dgrid['Drectgrids'] = Drectgrids
  Dgrid['Dsections'] = Dsections
  #fpath_grid = '/pool/data/ICON/oes/input/r0003/' + Dgrid['long_name'] +'/' + Dgrid['long_name'] + '.nc'
  return Dgrid

def mask_big_triangles(vlon, vlat, vertex_of_cell, Tri, only_lon=True):
  if only_lon:
    mask_bt = (
        (np.abs(  vlon[vertex_of_cell[:,0]] 
                - vlon[vertex_of_cell[:,1]])>180.)
      | (np.abs(  vlon[vertex_of_cell[:,0]] 
                - vlon[vertex_of_cell[:,2]])>180.)
                  )
  else:
    mask_bt = (
        (np.abs(  vlon[vertex_of_cell[:,0]] 
                - vlon[vertex_of_cell[:,1]])>180.)
      | (np.abs(  vlon[vertex_of_cell[:,0]] 
                - vlon[vertex_of_cell[:,2]])>180.)
      | (np.abs(  vlat[vertex_of_cell[:,0]] 
                - vlat[vertex_of_cell[:,1]])>90.)
      | (np.abs(  vlat[vertex_of_cell[:,0]] 
                - vlat[vertex_of_cell[:,2]])>90.)
                  )
  Tri.set_mask(mask_bt)
  return Tri, mask_bt

def triangulation(ds_tgrid, lon_reg=None, lat_reg=None, do_mask_big_triangles=True, only_lon=True, shift_to_zero_dateline=False):
  vlon = ds_tgrid.vlon * 180./np.pi
  vlat = ds_tgrid.vlat * 180./np.pi
  if shift_to_zero_dateline:
    vlon[vlon<0.] += 360.
  vertex_of_cell = ds_tgrid.vertex_of_cell.transpose()-1

  if lon_reg is not None:
    clon = ds_tgrid.clon * 180./np.pi
    clat = ds_tgrid.clat * 180./np.pi
    ind_reg = (clon>lon_reg[0]) & (clon<=lon_reg[1]) & (clat>lat_reg[0]) & (clat<=lat_reg[1])
    ind_reg = ind_reg.data
    vertex_of_cell = vertex_of_cell[ind_reg,:]
  else:
    ind_reg = None

  Tri = matplotlib.tri.Triangulation(vlon, vlat, triangles=vertex_of_cell)
  if do_mask_big_triangles:
    Tri, mask_bt = mask_big_triangles(vlon, vlat, vertex_of_cell, Tri, only_lon=only_lon)
  
  return ind_reg, Tri

def crop_tripolar_grid(lon_reg, lat_reg,
                       clon, clat, vertex_of_cell, edge_of_cell):
  ind_reg = np.where(   (clon>lon_reg[0]) 
                      & (clon<=lon_reg[1]) 
                      & (clat>lat_reg[0]) 
                      & (clat<=lat_reg[1]) )[0]
  clon = clon[ind_reg]
  clat = clat[ind_reg]
  vertex_of_cell = vertex_of_cell[ind_reg,:]
  edge_of_cell   = edge_of_cell[ind_reg,:]
  ind_reg = ind_reg
  return clon, clat, vertex_of_cell, edge_of_cell, ind_reg

def crop_regular_grid(lon_reg, lat_reg, Lon, Lat):
  # this does not work since Lon[ind_reg].shape = (64800, 360)
  # cropping needs probably done by each dimension
  # in this case cropping function for data is used as well
  lon = Lon[0,:]
  lat = Lat[:,0]
  indx = np.where((lon>=lon_reg[0]) & (lon<lon_reg[1]))[0]
  indy = np.where((lat>=lat_reg[0]) & (lat<lat_reg[1]))[0]
  lon = lon[indx]
  lat = lat[indy]
  #ind_reg = np.where(   (Lon>=lon_reg[0]) 
  #                    & (Lon <lon_reg[1]) 
  #                    & (Lat>=lat_reg[0]) 
  #                    & (Lat <lat_reg[1]) )[0]
  ind_reg = ((Lon>=lon_reg[0]) & (Lon<lon_reg[1]) & (Lat>=lat_reg[0]) & (Lat<lat_reg[1])).flatten()
  Lon, Lat = np.meshgrid(lon, lat)
  #Lon = Lon[ind_reg]
  #Lat = Lat[ind_reg]
  return Lon, Lat, lon, lat, ind_reg, indx, indy

"""
Routines related to time steps of data set
"""
def get_files_of_timeseries(path_data, fname):
  flist = np.array(glob.glob(path_data+fname))
  flist.sort()
  times_flist = np.zeros(flist.size, dtype='datetime64[s]')
  #for l, fpath in enumerate(flist):
  #  tstr = fpath.split('/')[-1].split('_')[-1][:-4]
  #  times_flist[l] = '%s-%s-%sT%s:%s:%s' % ( (tstr[:4], tstr[4:6], tstr[6:8], 
  #                                      tstr[9:11], tstr[11:13], tstr[13:15]))
  if flist.size==0:
    raise ValueError('::: Error: No file found matching %s!:::' % (path_data+fname))
  return times_flist, flist

def nctime2numpy(ncv):
  np_time = num2date(ncv[:], units=ncv.units, calendar=ncv.calendar
                  ).astype("datetime64[s]")
  return np_time

def nctime_to_datetime64(ncv_time, time_mode='num2date'):
  if time_mode=='num2date':
    np_time = num2date(ncv_time[:], units=ncv_time.units, calendar=ncv_time.calendar
                    ).astype("datetime64[s]")
  elif time_mode=='float2date':
    tps = ncv_time[:]
    secs_tot = np.round(86400.*(tps-np.floor(tps)))
    hours = np.floor(secs_tot/3600.)
    mins = np.floor((secs_tot-hours*3600.)/60.) 
    secs = secs_tot - hours*3600. - mins*60.
    tstrs = [0]*tps.size
    for l in range(tps.size):
      tp = tps[l]
      tstr = '%s-%s-%sT%02d:%02d:%02d' % (str(tp)[:4], str(tp)[4:6], str(tp)[6:8], hours[l], mins[l], secs[l]) 
      tstrs[l] = tstr
    np_time = np.array(tstrs, dtype='datetime64')
  else:
    raise ValueError('::: Error: Wrong time_mode %s in get_timesteps! :::' % time_mode)
  return np_time
  

def get_timesteps(flist, time_mode='num2date'):
  #f = Dataset(flist[0], 'r')
  #nt = f.variables['time'].size 
  #f.close()
  #times = np.zeros((len(flist)*nt))
  #times = np.array(['2010']*(len(flist)*nt), dtype='datetime64[s]')
  #its = np.zeros((len(flist)*nt), dtype='int')
  #flist_ts = np.zeros((len(flist)*nt), dtype='<U200')
  times = np.array([], dtype='datetime64[s]')
  its = np.array([], dtype='int')
  flist_ts = np.array([], dtype='<U200')
  for nn, fpath in enumerate(flist):
    f = Dataset(fpath, 'r')
    ncv_time = f.variables['time']
    nt = f.variables['time'].size 
    np_time = nctime_to_datetime64(ncv_time, time_mode=time_mode)
    ##if time_mode=='num2date':
    ##  np_time = num2date(ncv[:], units=ncv.units, calendar=ncv.calendar
    ##                  ).astype("datetime64[s]")
    ##elif time_mode=='float2date':
    ##  tps = ncv[:]
    ##  secs_tot = np.round(86400.*(tps-np.floor(tps)))
    ##  hours = np.floor(secs_tot/3600.)
    ##  mins = np.floor((secs_tot-hours*3600.)/60.) 
    ##  secs = secs_tot - hours*3600. - mins*60.
    ##  tstrs = [0]*tps.size
    ##  for l in range(tps.size):
    ##    tp = tps[l]
    ##    tstr = '%s-%s-%sT%02d:%02d:%02d' % (str(tp)[:4], str(tp)[4:6], str(tp)[6:8], hours[l], mins[l], secs[l]) 
    ##    tstrs[l] = tstr
    ##  np_time = np.array(tstrs, dtype='datetime64')
    ##else:
    ##  raise ValueError('::: Error: Wrong time_mode %s in get_timesteps! :::' % time_mode)
    #times[nn*nt:(nn+1)*nt] = np_time
    #flist_ts[nn*nt:(nn+1)*nt] = np.array([fpath]*nt)
    #its[nn*nt:(nn+1)*nt] = np.arange(nt)
    times    = np.concatenate((times, np_time))
    flist_ts = np.concatenate((flist_ts, np.array([fpath]*nt).astype('<U200')))
    its      = np.concatenate((its, np.arange(nt, dtype='int')))
    f.close()
  return times, flist_ts, its

def get_varnames(fpath, skip_vars=[]):
  f = Dataset(fpath, 'r')
  varnames = f.variables.keys()
  f.close()
  #varnames = [var for var in varnames if not var.startswith('clon')]
  for skip_var in skip_vars:
    varnames = [var for var in varnames if not var.startswith(skip_var)]
  return varnames

def indfind(elements, vector):                                                      
  """ return indices of elements that closest match elements in vector
  """
  # convert elements to np array                                                    
  if type(elements) is int or type(elements) is float:                              
    elements = np.array([elements])
  elif type(elements) is list:
    elements = np.array(elements)                                                   
  # find indices
  inds = [0]*elements.size                                                          
  for i in range(elements.size):                                                    
    inds[i] = np.argmin( np.abs(vector - elements[i]) )                             
  return inds

def write_dataarray_to_nc(
    fpath,
    data, coords,
    name='data', long_name='', units='',
    long_name_coords=None, units_coords=None,
    time_bnds=[],
    verbose=True,
  ):

  # --- initialize Dataset
  ds = xr.Dataset()

  # --- add data array to dataset
  ds[name] = xr.DataArray(data, dims=coords.keys(), coords=coords, attrs={'units': units, 'long_name': long_name})

  # --- add attributes to coordinates
  for nn, dims in enumerate(coords):
    if not dims.startswith('time') and units_coords is not None:
      ds[dims].attrs['units'] = units_coords[nn]
    if long_name_coords is not None:
      ds[dims].attrs['long_name'] = long_name_coords[nn]

  # --- add time bounds
  if len(time_bnds)!=0:
    ds['time_bnds'] = xr.DataArray(time_bnds)

  # --- write netcdf file
  if verbose:
    print(f'Writing data file {fpath}.')
  ds.to_netcdf(fpath)

  return ds

#def write_dataset_to_nc(
#    fpath,
#    data_list, dims_list, coords, 
#    name_list=[], long_name_list=[], units_list=[],
#    long_name_coords=None, units_coords=None,
#    time_bnds=[],
#    verbose=True,
#  ):
#
#  # --- initialize Dataset and add coords
#  ds = xr.Dataset(coords=coords)
#
#  # --- add attributes to coordinates
#  for nn, dims in enumerate(coords):
#    if not dims.startswith('time') and units_coords is not None:
#      ds[dims].attrs['units'] = units_coords[nn]
#    if long_name_coords is not None:
#      ds[dims].attrs['long_name'] = long_name_coords[nn]
#
#  # --- if no names for variables are given, make some up
#  if isempty(name_list):
#    for nn in range(len(data_list)):
#      name_list.append(f'data_{nn}')
#
#  # --- add data arrays to dataset
#  for nn, (data, coords, name) in enumerate(zip(data_list, coords_list, name_list)):
#    ds[name] = xr.DataArray(data, dims=coords.keys(), coords=coords, attrs={'units': units, 'long_name': long_name})
#
#  # --- add time bounds
#  if len(time_bnds)!=0:
#    ds['time_bnds'] = xr.DataArray(time_bnds)
#
#  # --- write netcdf file
#  if verbose:
#    print(f'Writing data file {fpath}.')
#  ds.to_netcdf(fpath)
#
#  return ds

#def nc_info(fpath):
#  if not os.path.isfile(fpath):
#    print("::: Error: file %s does not exist! :::" %(fpath))
#    sys.exit()
#  
#  ##ds = xr.open_dataset(fpath)
#  f = Dataset(fpath, 'r')
#  header =  "{code:<5}: {name:<30}: {long_name:<30}: {units:<20}: {shape:<20}".format(code='code', name='name', long_name='long_name', units='units', shape='shape')
#  print header
#  print '-'*len(header)
#  ##for var in ds.variables.keys():
#  for var in f.variables.keys():
#    ##name = ds[var].name
#    nv = f.variables[var]
#    name = "{:<30}: ".format(var[:29])
#    try:
#      ##lname = ds[var].long_name
#      lname = nv.long_name
#      lname = "{:<30}: ".format(lname[:29])
#    except:
#      lname = " "*30+": "
#    try:
#      units = nv.units
#      units = "{:<20}: ".format(units[:19])
#    except:
#      units = " "*20+": "
#    try:
#      ##code = ds[var].code
#      code = nv.code
#      code = "% 5d: "%(code)
#    except:
#      code = "     : "
#    ##shape = str(ds[var].shape)
#    shape = str(nv.shape)
#    shape = "{:<20}: ".format(shape[:19])
#    print code+name+lname+units+shape
#  f.close()
#  return Dfinf


# //////////////////////////////////////////////////////////////////////////////// 
# //////////////////////////////////////////////////////////////////////////////// 
# ---- classes and methods necessary for Jupyter data viewer

# ////////////////////////////////////////
class IP_hor_sec_rect(object):
  """
  To do:
  * similar to qp_hor_plot, see if we need both
  * try to use hplot_base
  """

  def __init__(self, 
               IcD, ax='', cax='',
               var='', clim='auto', nc=1, cmap='viridis',
               transform=None, lon_reg='auto', lat_reg='auto',
               title='auto',
               time_string='auto',
               depth_string='auto',
               edgecolor='none',
               ):
    self.ax=ax
    self.cax=cax
    self.var=var

    data = getattr(IcD, var)
    if IcD.use_tgrid:
      self.hpc = shade(IcD.Tri, 
                           data, ax=ax, cax=cax, 
                           clim=clim, cmap=cmap, transform=transform,
                           edgecolor=edgecolor,
                            )
      lon_reg = IcD.lon_reg
      lat_reg = IcD.lat_reg
    else:
      self.hpc = shade(IcD.lon, IcD.lat,
                           data, ax=ax, cax=cax, 
                           clim=clim, cmap=cmap, transform=transform,
                         ) 

      if isinstance(lon_reg, str) and lon_reg=='auto':
        lon_reg = [IcD.lon[0], IcD.lon[-1]]
      if isinstance(lat_reg, str) and lat_reg=='auto':
        lat_reg = [IcD.lat[0], IcD.lat[-1]]

    ax.set_xticks( np.linspace(lon_reg[0], lon_reg[1], 5) )
    ax.set_yticks( np.linspace(lat_reg[0], lat_reg[1], 5) )
    ax.set_xlim(*lon_reg)
    ax.set_ylim(*lat_reg)

    #ax.add_feature(cfeature.LAND, facecolor='0.7', zorder=3)
    ax.coastlines()

    if title=='auto':
      self.htitle = ax.set_title(IcD.long_name[var]+' ['+IcD.units[var]+']')
    else:
      self.htitle = ax.set_title(title)

    if time_string!='none':
      self.htstr = ax.text(0.05, 0.025, IcD.times[IcD.step_snap], 
                           transform=plt.gcf().transFigure)
    if depth_string!='none':
      self.hdstr = ax.text(0.05, 0.08, 'depth = %4.1fm'%(IcD.depth[IcD.iz]), 
                           transform=plt.gcf().transFigure)
    return
  
  def update(self, data, IcD, title='none', 
             time_string='auto', depth_string='auto'):
    if IcD.use_tgrid:
      data_nomasked_vals = data[IcD.maskTri==False]
      #print self.hpc[0].get_array.shape()
      self.hpc[0].set_array(data_nomasked_vals)
      #print self.hpc[0].get_array.shape()
      print('hello world')
    else:
      self.hpc[0].set_array(data[1:,1:].flatten())
    if title!='none':
      self.htitle.set_text(title) 
    if time_string!='none':
      self.htstr.set_text(IcD.times[IcD.step_snap])
    if depth_string!='none':
      self.hdstr.set_text('depth = %4.1fm'%(IcD.depth[IcD.iz]))
    return

##class IP_ver_sec(object):
##  def __init__(self, 
##               IcD, ax='', cax='',
##               var='', clim='auto', nc=1, cmap='viridis',
##               title='auto',
##               time_string='auto',
##               depth_string='auto',
##               edgecolor='none',
##               ):
##    self.ax=ax
##    self.cax=cax
##    self.var=var
##
##    data = getattr(IcD, var)
##    self.hpc = shade(IcD.dist_sec, IcD.depth,
##                         data, ax=ax, cax=cax, 
##                         clim=clim, cmap=cmap,
##                       ) 
##
##    ax.set_ylim(IcD.depth.max(),0)
##
##    if title=='auto':
##      self.htitle = ax.set_title(IcD.long_name[var]+' ['+IcD.units[var]+']')
##    else:
##      self.htitle = ax.set_title(title)
##
##    if time_string!='none':
##      self.htstr = ax.text(0.05, 0.025, IcD.times[IcD.step_snap], 
##                           transform=plt.gcf().transFigure)
##    #if depth_string!='none':
##    #  self.hdstr = ax.text(0.05, 0.08, 'depth = %4.1fm'%(IcD.depth[IcD.iz]), 
##    #                       transform=plt.gcf().transFigure)
##    return
##  
##  def update(self, data, IcD, title='none', 
##             time_string='auto', depth_string='auto'):
##    if IcD.use_tgrid:
##      data_nomasked_vals = data[IcD.maskTri==False]
##      #print self.hpc[0].get_array.shape()
##      self.hpc[0].set_array(data_nomasked_vals)
##      #print self.hpc[0].get_array.shape()
##      print('hello world')
##    else:
##      self.hpc[0].set_array(data[1:,1:].flatten())
##    if title!='none':
##      self.htitle.set_text(title) 
##    if time_string!='none':
##      self.htstr.set_text(IcD.times[IcD.step_snap])
##    if depth_string!='none':
##      self.hdstr.set_text('depth = %4.1fm'%(IcD.depth[IcD.iz]))
##    return

#def asses_memory(mode=0):
#    vdict = globals()
#    Dmem = dict()
#    tot_mem = 0.
#    for el in vdict:
#        if el.startswith('_'):
#            continue
#        try:
#            mem = vdict[el].nbytes/1e9
#            Dmem[el] = [mem]
#            tot_mem += mem
#    #         print(f'Memory of {el}: {vdict[el].nbytes/1e9}')
#        except:
#            pass
#    Dmem['tot_mem'] = [tot_mem]
#    df = pd.DataFrame(Dmem, index=['memory [GB]']).transpose()
#    if mode==1:
#        print(df)
#    if mode==2:
#        print(f'total memory [GB]: {tot_mem}')
#    return df

def info(array):
  """ Prints some information about an xarray. (not well tested yet).

  To do:
    * decide whether numpy, or array
    * decide whether xarray contains dask or numpy array
  """
  print(f'| dims = {array.dims} | shape = {array.shape} | memory = {array.nbytes/1024**3:.3f}GB | chunks = {array.chunks} |')
  return

def stats(array):
  """ Prints some basic information about an array. (not well tested yet).

  To do:
    * decide whether numpy, or xarray
    * decide whether xarray contains dask or numpy array
  """
  print(f'| min = {array.min().compute().data} | max = {array.max().compute().data} | mean = {array.mean().compute().data} |')
  print(f'| #NaNs = {xr.ufuncs.isnan(array).sum().compute().data} | #Zeros = {(array==0).sum().compute().data} |')
  return

def patch_plot_derive_bnds(ds_tgrid, lon_reg=[-180, 180], lat_reg=[-90, 90]):
  # --- load tgrid
  vertex_of_cell = ds_tgrid['vertex_of_cell'][:].transpose()-1
  clon_bnds = ds_tgrid.vlon[vertex_of_cell] * 180./np.pi
  clat_bnds = ds_tgrid.vlat[vertex_of_cell] * 180./np.pi

  cells_of_vertex = ds_tgrid['cells_of_vertex'][:].transpose()-1
  vlon_bnds = ds_tgrid.clon[cells_of_vertex] * 180./np.pi
  vlat_bnds = ds_tgrid.clat[cells_of_vertex] * 180./np.pi

  vlon_bnds = vlon_bnds.to_masked_array()
  vlat_bnds = vlat_bnds.to_masked_array()
  cells_of_vertex = cells_of_vertex.to_masked_array()
  vlon_bnds[cells_of_vertex==-1] = np.ma.masked
  vlat_bnds[cells_of_vertex==-1] = np.ma.masked

  clon = ds_tgrid.clon.data * 180./np.pi
  clat = ds_tgrid.clat.data * 180./np.pi
  vlon = ds_tgrid.vlon.data * 180./np.pi
  vlat = ds_tgrid.vlat.data * 180./np.pi

  # --- sort edges of dual grid
  x = (vlon_bnds-vlon[:,np.newaxis])
  y = (vlat_bnds-vlat[:,np.newaxis])

  ex, ey = 1., 0.

  ab_scalar = (x*ex+y*ey)/np.sqrt(x**2+y**2)
  ab_cross  = (x*ey-y*ex)/np.sqrt(x**2+y**2)

  sort_crit = np.ma.arccos(ab_scalar) * np.sign(ab_cross) * 180./np.pi
  isort = np.ma.argsort(sort_crit, axis=1)

  sorted_sort_crit = np.take_along_axis(sort_crit, isort, axis=1)
  cells_of_vertex_sorted = np.take_along_axis(cells_of_vertex, isort, axis=1)
  vlon_bnds_sorted = np.take_along_axis(vlon_bnds, isort, axis=1)
  vlat_bnds_sorted = np.take_along_axis(vlat_bnds, isort, axis=1)
  
  vlon_bnds_sorted = xr.DataArray(vlon_bnds_sorted, dims=['vertex', 'nc'])
  vlat_bnds_sorted = xr.DataArray(vlat_bnds_sorted, dims=['vertex', 'nc'])
  
  # --- cut region
  ireg_v = (vlon>lon_reg[0]) & (vlon<=lon_reg[1]) & (vlat>lat_reg[0]) & (vlat<=lat_reg[1])
  ireg_c = (clon>lon_reg[0]) & (clon<=lon_reg[1]) & (clat>lat_reg[0]) & (clat<=lat_reg[1])

  vlon_bnds_sorted = vlon_bnds_sorted[ireg_v]
  vlat_bnds_sorted = vlat_bnds_sorted[ireg_v]
  cells_of_vertex_sorted = cells_of_vertex_sorted[ireg_v]
  clon_bnds = clon_bnds[ireg_c]
  clat_bnds = clat_bnds[ireg_c]

  return clon_bnds, clat_bnds, vlon_bnds_sorted, vlat_bnds_sorted, cells_of_vertex_sorted

def calc_grid_area_rectgrid(lon, lat):
  # FIXME: Optimize by getting rid of loops
  radius = 6371000.
  dlon = np.radians(lon[2] - lon[1])
  area = np.zeros((lat.size,lon.size))
  for j in np.arange(1,np.size(lat)-1):
    lat1 = (lat[j-1] +  lat[j]  )*0.5
    lat2 = (lat[j]   +  lat[j+1])*0.5
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    for i in np.arange(1,np.size(lon)-1):
      # A = R^2 |sin(lat1)-sin(lat2)| |lon1-lon2| where R is earth radius (6378 kms)
      area[j,i] = np.square(radius)*(np.sin(lat2)-np.sin(lat1))*dlon
  # area fraction w.r.t. Earth's surface area
  area_e = 4. * np.pi * np.square(radius)
  area = area / area_e
  return area

def rename_dimension_names(ds, rename_dict=None, verbose=False):
    """ Rename dimension names of ICON output files to match name convention in grid files.
    
    ds: Xarray dataset which should be renamed
    rename_dict: Dictionary. which contains renaming instruction. If left None, default convention is used.
    verbose: Bool. If set to true rename_dict is printed.
    """
    if rename_dict is None:
        # rename horizontal dimensions
        rename_dict = dict()
        try:
            rename_dict[ds.clon.dims[0]] = 'cell'
        except:
            pass
        try:
            rename_dict[ds.elon.dims[0]] = 'edge'
        except:
            pass
        try:
            rename_dict[ds.vlon.dims[0]] = 'vertex'
        except:
            pass
        
        # rename vertical dimensions
        # !!! Renaming vertical dimensions is problematic since it cannot capture all potential cases. !!!

        # rename ocean depth dimension
        depth_dims = [(dim) for dim in ds.dims if dim.startswith('depth')]
        if len(depth_dims)==0:
            pass
        elif len(depth_dims)==1:
            # assume dimension pointing to center if only one depth dim is in ds 
            # (this case is more likely but other cases are possible as well)
            rename_dict[depth_dims[0]] = 'depthc'
        elif ds[depth_dims[0]].size<ds[depth_dims[1]].size:
            # assume that interface dimension is (by 1) larger than cell dimension
            # (this should always be the case)
            rename_dict[depth_dims[0]] = 'depthc'
            rename_dict[depth_dims[1]] = 'depthi'
        else:
            rename_dict[depth_dims[0]] = 'depthi'
            rename_dict[depth_dims[1]] = 'depthc'
            
        # rename atm height dimension
        height_dims = [(dim) for dim in ds.dims if dim.startswith('height')]
        if len(height_dims)==0:
            pass
        elif len(height_dims)==1:
            rename_dict[height_dims[0]] = 'heightc'
        elif ds[height_dims[0]].size<ds[height_dims[1]].size: # this will break if ds does not contain height, height_2... not sure whether it does
            rename_dict[height_dims[0]] = 'heightc'
            rename_dict[height_dims[1]] = 'heighti'
        else:
            rename_dict[height_dims[0]] = 'heighti'
            rename_dict[height_dims[1]] = 'heightc'

    if verbose:
        print(rename_dict)
        
    # finally rename dataset
    ds = ds.rename(rename_dict)
    return ds
