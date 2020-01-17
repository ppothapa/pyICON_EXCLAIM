import sys, glob, os
import datetime
import numpy as np
from netCDF4 import Dataset, num2date
from scipy import interpolate
from scipy.spatial import cKDTree
# --- plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
#import my_toolbox as my
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
import json
from ipdb import set_trace as mybreak  
from importlib import reload

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
  load_tripolar_grid
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
#                  inds=None, radius_of_influence=100e3):
#  """
#  """
#  data_interpolated = apply_ckdtree(data, distances=distances, inds=inds, 
#                                    radius_of_influence=radius_of_influence)
#  data_interpolated = data_interpolated.reshape(shape)
#  return data_interpolated
#
#def icon_to_section(data, distances=None, \
#                  inds=None, radius_of_influence=100e3):
#  """
#  """
#  data_interpolated = apply_ckdtree(data, distances=distances, inds=inds, 
#                                    radius_of_influence=radius_of_influence)
#  return data_interpolated

def apply_ckdtree(data, fpath_ckdtree, coordinates='clat clon', radius_of_influence=100e3):
  """
  * credits
    function modified from pyfesom (Nikolay Koldunov)
  """
  ddnpz = np.load(fpath_ckdtree)
  if coordinates=='clat clon':
    distances = ddnpz['dckdtree_c']
    inds = ddnpz['ickdtree_c'] 
  elif coordinates=='elat elon':
    distances = ddnpz['dckdtree_e']
    inds = ddnpz['ickdtree_e'] 
  elif coordinates=='vlat vlon':
    distances = ddnpz['dckdtree_v']
    inds = ddnpz['ickdtree_v'] 
  else:
    raise ValueError('::: Error: Unsupported coordinates: %s! ::: ' % (coordinates))

  if distances.ndim == 1:
    #distances_ma = np.ma.masked_greater(distances, radius_of_influence)
    if data.ndim==1:
      data_interpolated = data[inds]
      data_interpolated[distances>=radius_of_influence] = np.nan
      data_interpolated = np.ma.masked_invalid(data_interpolated)
    elif data.ndim==2:
      data_interpolated = data[:,inds]
      data_interpolated[:,distances>=radius_of_influence] = np.nan
      data_interpolated = np.ma.masked_invalid(data_interpolated)
  else:
    #raise ValueError("::: distances.ndim>1 is not properly supported yet. :::")
    distances_ma = np.ma.masked_greater(distances, radius_of_influence)
    
    w = 1.0 / distances_ma**2
    data_interpolated = np.ma.sum(w * data[inds], axis=1) / np.ma.sum(w, axis=1)
    data_interpolated = np.ma.masked_invalid(data_interpolated)
  return data_interpolated

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
    data = data3d[k,:]
    #print('k = %d/%d'%(k,nz))
    # --- mask land points
    data[data==0] = np.ma.masked
    # --- mask not-this-basin points
    data[mask_basin==False] = np.ma.masked
    # --- go to normal np.array (not np.ma object)
    data = data.filled(0.)
    # --- interpolate to rectangular grid
    datai = apply_ckdtree(data, fpath_ckdtree, coordinates=coordinates)
    datai = datai.reshape(shape)
    # --- go back to masked array
    datai = np.ma.array(datai, mask=datai==0.)
    # --- do zonal average
    data_zave[k,:] = datai.mean(axis=1)
  return lat_sec, data_zave

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
                 ):
  """
  """
  if tgname=='':
    Drgrid = identify_grid(path_tgrid, path_tgrid+fname_tgrid) 
    tgname = Drgrid['name']
  lon1str, lat1str = lonlat2str(lon_reg[0], lat_reg[0])
  lon2str, lat2str = lonlat2str(lon_reg[1], lat_reg[1])

  if n_nearest_neighbours==1:
    fname = '%s_res%3.2f_%s-%s_%s-%s.npz'%(tgname, res, lon1str, lon2str, lat1str, lat2str) 
  else:
    fname = '%s_res%3.2f_%dnn_%s-%s_%s-%s.npz'%(tgname, res, n_nearest_neighbours, lon1str, lon2str, lat1str, lat2str) 
  fpath_ckdtree = path_ckdtree+fname

  # --- load triangular grid
  f = Dataset(path_tgrid+fname_tgrid, 'r')
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

  # --- make rectangular grid 
  lon = np.arange(lon_reg[0],lon_reg[1],res)
  lat = np.arange(lat_reg[0],lat_reg[1],res)
  Lon, Lat = np.meshgrid(lon, lat)

  # --- ckdtree for cells, edges and vertices
  if load_cgrid:
    dckdtree_c, ickdtree_c = calc_ckdtree(lon_i=clon, lat_i=clat,
                                          lon_o=Lon.flatten(), lat_o = Lat.flatten(),
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          )
  if load_egrid:
    dckdtree_e, ickdtree_e = calc_ckdtree(lon_i=elon, lat_i=elat,
                                          lon_o=Lon.flatten(), lat_o = Lat.flatten(),
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          )
  if load_vgrid:
    dckdtree_v, ickdtree_v = calc_ckdtree(lon_i=vlon, lat_i=vlat,
                                          lon_o=Lon.flatten(), lat_o = Lat.flatten(),
                                          n_nearest_neighbours=n_nearest_neighbours,
                                          )

  # --- save dict
  Dsave = dict()
  if load_cgrid: 
    Dsave['dckdtree_c'] = dckdtree_c
    Dsave['ickdtree_c'] = ickdtree_c
  if load_egrid: 
    Dsave['dckdtree_e'] = dckdtree_e
    Dsave['ickdtree_e'] = ickdtree_e
  if load_vgrid: 
    Dsave['dckdtree_v'] = dckdtree_v
    Dsave['ickdtree_v'] = ickdtree_v

  # --- save grid
  print('Saving grid file: %s' % (fpath_ckdtree))
  np.savez(fpath_ckdtree,
            lon=lon,
            lat=lat,
            sname=sname,
            gname=gname,
            tgname='test',
            **Dsave,
            #dckdtree_c=dckdtree_c,
            #ickdtree_c=ickdtree_c,
            #dckdtree_e=dckdtree_e,
            #ickdtree_e=ickdtree_e,
            #dckdtree_v=dckdtree_v,
            #ickdtree_v=ickdtree_v,
           )
  return

def ckdtree_section(p1, p2, npoints=101, 
                 fname_tgrid='',
                 path_tgrid='',
                 path_ckdtree='',
                 sname='auto',
                 gname='',
                 tgname='',
                 ):
  """
  """
  if tgname=='':
    Drgrid = identify_grid(path_tgrid, path_tgrid+fname_tgrid) 
    tgname = Drgrid['name']
  lon1str, lat1str = lonlat2str(p1[0], p1[1])
  lon2str, lat2str = lonlat2str(p2[0], p2[1])

  fname = '%s_nps%d_%s%s_%s%s.npz'%(tgname, npoints, lon1str, lat1str, lon2str, lat2str) 
  fpath_ckdtree = path_ckdtree+fname

  # --- load triangular grid
  f = Dataset(path_tgrid+fname_tgrid, 'r')
  clon = f.variables['clon'][:] * 180./np.pi
  clat = f.variables['clat'][:] * 180./np.pi
  elon = f.variables['elon'][:] * 180./np.pi
  elat = f.variables['elat'][:] * 180./np.pi
  vlon = f.variables['vlon'][:] * 180./np.pi
  vlat = f.variables['vlat'][:] * 180./np.pi
  f.close()

  if sname=='auto':
    sname = fpath_ckdtree.split('/')[-1][:-4]

  # --- derive section points
  lon_sec, lat_sec, dist_sec = derive_section_points(p1, p2, npoints)

  # --- ckdtree for cells, edges and vertices
  dckdtree_c, ickdtree_c = calc_ckdtree(lon_i=clon, lat_i=clat,
                                        lon_o=lon_sec, lat_o=lat_sec)
  dckdtree_e, ickdtree_e = calc_ckdtree(lon_i=elon, lat_i=elat,
                                        lon_o=lon_sec, lat_o=lat_sec)
  dckdtree_v, ickdtree_v = calc_ckdtree(lon_i=vlon, lat_i=vlat,
                                        lon_o=lon_sec, lat_o=lat_sec)

  # --- save grid
  print('Saving grid file: %s' % (fpath_ckdtree))
  np.savez(fpath_ckdtree,
            dckdtree_c=dckdtree_c,
            ickdtree_c=ickdtree_c,
            dckdtree_e=dckdtree_e,
            ickdtree_e=ickdtree_e,
            dckdtree_v=dckdtree_v,
            ickdtree_v=ickdtree_v,
            lon_sec=lon_sec,
            lat_sec=lat_sec,
            dist_sec=dist_sec,
            sname=sname,
            gname=gname,
           )
  return dckdtree_c, ickdtree_c, lon_sec, lat_sec, dist_sec

def calc_ckdtree(lon_i, lat_i, lon_o, lat_o, n_nearest_neighbours=1):
  """
  """
  # --- initialize timing
#  tims = np.array([0])
#  tims = timing(tims)
  # --- do ckdtree
#  tims = timing(tims, 'CKD: define reg grid')
  lzip_i = list(zip(lon_i, lat_i))
#  tims = timing(tims, 'CKD: zip orig grid')
  tree = cKDTree(lzip_i)
#  tims = timing(tims, 'CKD: CKDgrid')
  lzip_o = list(zip(lon_o, lat_o))
#  tims = timing(tims, 'CKD: zip reg grid')
  dckdtree, ickdtree = tree.query(lzip_o , k=n_nearest_neighbours, n_jobs=1)
#  tims = timing(tims, 'CKD: tree query')
  return dckdtree, ickdtree

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

def derive_section_points(p1, p2, npoints=101,):
  # --- derive section points
  if p1[0]==p2[0]:
    lon_sec = p1[0]*np.ones((npoints)) 
    lat_sec = np.linspace(p1[1],p2[1],npoints)
  else:
    lon_sec = np.linspace(p1[0],p2[0],npoints)
    lat_sec = (p2[1]-p1[1])/(p2[0]-p1[0])*(xsec-p1[0])+p1[1]
  dist_sec = haversine_dist(lon_sec[0], lat_sec[0], lon_sec, lat_sec)
  return lon_sec, lat_sec, dist_sec

def load_hsnap(fpath, var, it=0, iz=0, fpath_ckdtree=''):
  f = Dataset(fpath, 'r')
  print("Loading %s from %s" % (var, fpath))
  if f.variables[var].ndim==2:
    data = f.variables[var][it,:]
  else:
    data = f.variables[var][it,iz,:]
  f.close()

  data[data==0.] = np.ma.masked
  return data

def timing(ts, string=''):
  if ts[0]==0:
    ts = np.array([datetime.datetime.now()])
  else:
    ts = np.append(ts, [datetime.datetime.now()])
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

def identify_grid(path_grid, fpath_data):
  """ Identifies ICON grid in depending on clon.size in fpath_data.
  
  r2b4: 160km:    15117: OceanOnly_Icos_0158km_etopo40.nc
  r2b6:  40km:   327680: OCEANINP_pre04_LndnoLak_039km_editSLOHH2017_G.nc
  r2b8:  10km:  3729001: OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min.nc
  r2b9:   5km: 14886338: OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min.nc
  r2b9a:  5km: 20971520: /pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc
  """
  
  Dgrid_list = dict()
  
  grid_name = 'r2b4'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '160km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Icos_0158km_etopo40'
  Dgrid_list[grid_name]['size'] = 15117
  Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  
  grid_name = 'r2b6'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '40km'
  Dgrid_list[grid_name]['long_name'] = 'OCEANINP_pre04_LndnoLak_039km_editSLOHH2017_G'
  Dgrid_list[grid_name]['size'] = 327680
  Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  
  grid_name = 'r2b8'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '10km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_Global_IcosSymmetric_0010km_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 3729001
  Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'
  
  grid_name = 'r2b9'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '5km'
  Dgrid_list[grid_name]['long_name'] = 'OceanOnly_IcosSymmetric_4932m_rotatedZ37d_modified_srtm30_1min'
  Dgrid_list[grid_name]['size'] = 14886338
  Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '/' + Dgrid_list[grid_name]['long_name'] + '.nc'

  grid_name = 'r2b9a'; Dgrid_list[grid_name] = dict()
  Dgrid_list[grid_name]['name'] = grid_name
  Dgrid_list[grid_name]['res'] = '5km'
  Dgrid_list[grid_name]['long_name'] = 'icon_grid_0015_R02B09_G'
  Dgrid_list[grid_name]['size'] = 20971520
  Dgrid_list[grid_name]['fpath_grid'] = path_grid + Dgrid_list[grid_name]['long_name'] + '.nc'
  
  f = Dataset(fpath_data, 'r')
  gsize = f.variables['clon'].size
  f.close()
  for grid_name in Dgrid_list.keys():
    if gsize == Dgrid_list[grid_name]['size']:
      Dgrid = Dgrid_list[grid_name]
      break
  fpath_grid = '/pool/data/ICON/oes/input/r0002/' + Dgrid['long_name'] +'/' + Dgrid['long_name'] + '.nc'
  return Dgrid

def load_tripolar_grid(fpath_grid):
  """ Load longitude and latitude of cell centers, edges and vertices and vertex_of_cell and edge_of_cell from fpath_grid.'
  """
  f = Dataset(fpath_grid, 'r')
  clon = f.variables['clon'][:] * 180./np.pi
  clat = f.variables['clat'][:] * 180./np.pi
  vlon = f.variables['vlon'][:] * 180./np.pi
  vlat = f.variables['vlat'][:] * 180./np.pi
  elon = f.variables['elon'][:] * 180./np.pi
  elat = f.variables['elat'][:] * 180./np.pi
  vertex_of_cell = f.variables['vertex_of_cell'][:]-1
  vertex_of_cell = vertex_of_cell.transpose()
  edge_of_cell = f.variables['edge_of_cell'][:]-1
  edge_of_cell = edge_of_cell.transpose()
  f.close()
  return clon, clat, vlon, vlat, elon, elat, vertex_of_cell, edge_of_cell

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
  print(Lon.shape)
  ind_reg = np.where(   (Lon>=lon_reg[0]) 
                      & (Lon<=lon_reg[1]) 
                      & (Lat>=lat_reg[0]) 
                      & (Lat<=lat_reg[1]) )[0]
  Lon = Lon[ind_reg]
  Lat = Lat[ind_reg]
  lon = Lon[0,:] 
  lat = Lat[:,0] 
  ind_reg = ind_reg
  return Lon, Lat, lon, lat, ind_reg

def get_files_of_timeseries(path_data, search_str):
  flist = np.array(glob.glob(path_data+search_str))
  flist.sort()
  times_flist = np.zeros(flist.size, dtype='datetime64[s]')
  #for l, fpath in enumerate(flist):
  #  tstr = fpath.split('/')[-1].split('_')[-1][:-4]
  #  times_flist[l] = '%s-%s-%sT%s:%s:%s' % ( (tstr[:4], tstr[4:6], tstr[6:8], 
  #                                      tstr[9:11], tstr[11:13], tstr[13:15]))
  if flist.size==0:
    raise ValueError('::: Error: No file found matching %s!:::' % (path_data+search_str))
  return times_flist, flist

def get_varnames(fpath, skip_vars=[]):
  f = Dataset(fpath, 'r')
  varnames = f.variables.keys()
  f.close()
  #varnames = [var for var in varnames if not var.startswith('clon')]
  for skip_var in skip_vars:
    varnames = [var for var in varnames if not var.startswith(skip_var)]
  return varnames

def get_timesteps(flist, mode='num2date'):
  f = Dataset(flist[0], 'r')
  nt = f.variables['time'].size 
  f.close()
  #times = np.zeros((len(flist)*nt))
  times = np.array(['2010']*(len(flist)*nt), dtype='datetime64[s]')
  its = np.zeros((len(flist)*nt), dtype='int')
  flist_ts = np.zeros((len(flist)*nt), dtype='<U200')
  for nn, fpath in enumerate(flist):
    f = Dataset(fpath, 'r')
    ncv = f.variables['time']
    if mode=='num2date':
      np_time = num2date(ncv[:], units=ncv.units, calendar=ncv.calendar
                      ).astype("datetime64[s]")
    elif mode=='float2date':
      tps = ncv[:]
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
      raise ValueError('::: Error: Wrong mode %s in get_timesteps! :::' % mode)
    times[nn*nt:(nn+1)*nt] = np_time
    f.close()
    flist_ts[nn*nt:(nn+1)*nt] = np.array([fpath]*nt)
    its[nn*nt:(nn+1)*nt] = np.arange(nt)
  return times, flist_ts, its

def hplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               projection='none', use_tgrid='auto',
               logplot=False,
               sasp=0.5,
               fig_size_fac=2.,
              ):
  """
  IaV variable needs the following attributes
    * name
    * long_name
    * units
    * data
    if use_tgrid==True
    * Tri
    else 
    * lon
    * lat

  returns:
    * ax
    * cax
    * mappable
  """
  Dstr = dict()

  # --- plotting on original tgrid or interpolated rectgrid
  if isinstance(use_tgrid, str) and use_tgrid=='auto':
    use_tgrid = IcD.use_tgrid

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  if cmap.startswith('cmo'):
    cmap = cmap.split('.')[-1]
    cmap = getattr(cmocean.cm, cmap)
  else:
    cmap = getattr(plt.cm, cmap)

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
  else:
    ccrs_proj = getattr(ccrs, projection)()

  # --- make axes and colorbar (taken from shade)
  if ax == 'auto':
      #fig, ax = plt.subplots(subplot_kw={'projection': ccrs_proj}) 
    hca, hcb = arrange_axes(1,1, plot_cb=True, sasp=sasp, fig_size_fac=fig_size_fac,
                                 projection=ccrs_proj,
                                )
    ax = hca[0]
    cax = hcb[0]

  # --- do plotting
  if use_tgrid:
    hm = trishade(IcD.Tri, IaV.data, 
                      ax=ax, cax=cax, clim=clim, cincr=cincr, cmap=cmap,
                      transform=ccrs_proj,
                      logplot=logplot,
                 )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.clon.min(), IcD.clon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.clat.min(), IcD.clat.max()]
  else:
    hm = shade(IcD.lon, IcD.lat, IaV.data,
                      ax=ax, cax=cax, clim=clim, cincr=cincr, cmap=cmap,
                      transform=ccrs_proj,
                      logplot=logplot,
              )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.lon.min(), IcD.lon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.lat.min(), IcD.lat.max()]

  mappable = hm[0]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  if projection!='none':
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='0.9')
    ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7), crs=ccrs_proj)
    ax.set_yticks(np.linspace(np.round(ylim[0]),np.round(ylim[1]),7), crs=ccrs_proj)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #ax.stock_img()
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')
  return ax, cax, mappable, Dstr

def vplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               xvar='lat',
               log2vax=False,
               logplot=False,
               sasp=0.5,
               fig_size_fac=2.0,
              ):
  """
  IaV variable needs the following attributes
    * name
    * long_name
    * units
    * data
    * lon_sec, lat_sec, dist_sec

  returns:
    * ax
    * cax
    * mappable
  """
  Dstr = dict()

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  if cmap.startswith('cmo'):
    cmap = cmap.split('.')[-1]
    cmap = getattr(cmocean.cm, cmap)
  else:
    cmap = getattr(plt.cm, cmap)

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- make axes and colorbar (taken from shade)
  if ax == 'auto':
    hca, hcb = arrange_axes(1,1, plot_cb=True, sasp=sasp, fig_size_fac=fig_size_fac,
                           )
    ax = hca[0]
    cax = hcb[0]

  nz = IaV.data.shape[0]

  # --- do plotting
  if xvar=='lon':
    x = IaV.lon_sec
    xstr = 'longitude'
  elif xvar=='lat':
    x = IaV.lat_sec
    xstr = 'latitude'
  elif xvar=='dist':
    x = IaC.dist_sec/1e3
    xstr = 'distance [km]'
  if nz==IcD.depthc.size:
    depth = IcD.depthc
  else: 
    depth = IcD.depthi
  if log2vax:
    depth = np.log(depth)/np.log(2) 
  ylabel = 'depth [m]'

  hm = shade(x, depth, IaV.data,
                    ax=ax, cax=cax, clim=clim, cmap=cmap, cincr=cincr,
                    logplot=logplot,
            )
  if isinstance(xlim, str) and (xlim=='auto'):
    xlim = [x.min(), x.max()]
  if isinstance(ylim, str) and (ylim=='auto'):
    ylim = [depth.max(), depth.min()]

  mappable = hm[0]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xstr)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  if log2vax:
    ax.set_yticklabels(2**ax.get_yticks())
  ax.set_facecolor('0.8')
  ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7))
  ax.set_yticks(np.arange(0,5000,1000.))
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')

  return ax, cax, mappable, Dstr


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
class IconVariable(object):
  def __init__(self, name, units='', long_name='', 
                     coordinates='clat clon', fpath_ckdtree='',
                     is3d=None, isinterpolated=False,
               ):
    self.name = name
    self.units = units
    self.long_name = long_name
    self.is3d = is3d
    self.coordinates = coordinates
    self.isinterpolated = isinterpolated
    self.fpath_ckdtree = fpath_ckdtree
    return

  def load_hsnap(self, fpath, it=0, iz=0, step_snap=0, fpath_ckdtree=''):
    self.step_snap = step_snap
    self.it = it
    self.iz = iz
    self.fpath = fpath

    self.data = load_hsnap(fpath, self.name, it=it, iz=iz, fpath_ckdtree=fpath_ckdtree)
    self.mask = self.data.mask

    if fpath_ckdtree=='':
      self.isinterpolated = False
    else:
      self.interp_to_rectgrid(fpath_ckdtree)
      self.isinterpolated = True
    return
  
  def load_vsnap(self, fpath, fpath_ckdtree, it=0, step_snap=0):
    self.step_snap = step_snap
    self.it = it
    self.fpath = fpath
    # --- load ckdtree
    ddnpz = np.load(fpath_ckdtree)
    #dckdtree = ddnpz['dckdtree']
    #ickdtree = ddnpz['ickdtree'] 
    self.lon_sec = ddnpz['lon_sec'] 
    self.lat_sec = ddnpz['lat_sec'] 
    self.dist_sec = ddnpz['dist_sec'] 

    f = Dataset(fpath, 'r')
    var = self.name
    print("Loading %s from %s" % (var, fpath))
    if f.variables[var].ndim==2:
      raise ValueError('::: Warning: Cannot do section of 2D variable %s! :::'%var)
    nz = f.variables[var].shape[1]
    data = np.ma.zeros((nz,self.dist_sec.size))
    for k in range(nz):
      #print('k = %d/%d'%(k,nz))
      data_hsec = f.variables[var][it,k,:]
      data[k,:] = apply_ckdtree(data_hsec, fpath_ckdtree, coordinates=self.coordinates)
    f.close()
    self.data = data

    self.mask = self.data==0.
    self.data[self.mask] = np.ma.masked
    self.nz = nz
    return

  def load_moc(self, fpath, it=0, step_snap=0):
    self.step_snap = step_snap
    self.it = it
    self.fpath = fpath

    var = self.name
    print("Loading %s from %s" % (var, fpath))

    f = Dataset(fpath, 'r')
    self.nz = f.variables[var].shape[1]
    self.data = f.variables[var][it,:,:,0]
    self.lat_sec = f.variables['lat'][:]
    self.depth = f.variables['depth'][:]
    f.close()

    self.mask = self.data==0.
    self.data[self.mask] = np.ma.masked
    return

  def interp_to_rectgrid(self, fpath_ckdtree):
    if self.isinterpolated:
      raise ValueError('::: Variable %s is already interpolated. :::'%self.name)

    ddnpz = np.load(fpath_ckdtree)
    #dckdtree = ddnpz['dckdtree']
    #ickdtree = ddnpz['ickdtree'] 
    self.lon = ddnpz['lon'] 
    self.lat = ddnpz['lat'] 
    self.data = apply_ckdtree(self.data, fpath_ckdtree, coordinates=self.coordinates)
    self.data = self.data.reshape([self.lat.size, self.lon.size])
    self.data[self.data==0.] = np.ma.masked
    return

#### //////////////////////////////////////////////////////////////////////////////// 
###class IconDataFile(object):
###  """
###  Used by QuickPlots
###
###  To do:
###    * similar to IconData see if we need both
###  """
###  def __init__(self, 
###               fpath_data,
###               path_grid='/pool/data/ICON/oes/input/r0002/',
###              ):
###    self.path_grid = path_grid
###    self.fpath_data = fpath_data
###    return
###
###
###  def identify_grid(self):
###    self.Dgrid = identify_grid(path_grid=self.path_grid, fpath_data=self.fpath_data)
###    return
###  
###  def load_tripolar_grid(self):
###    (self.clon, self.clat, self.vlon, self.vlat,
###     self.elon, self.elat, self.vertex_of_cell,
###     self.edge_of_cell ) = load_tripolar_grid(fpath_grid=self.Dgrid['fpath_grid'])
###    return
###  
###  def crop_grid(self, lon_reg, lat_reg, grid='orig'):
###    """ Crop all cell related variables (data, clon, clat, vertex_of_cell, edge_of_cell to regin defined by lon_reg and lat_reg.
###    """
###    if grid=='orig':
###      (self.clon, self.clat,
###       self.vertex_of_cell, self.edge_of_cell,
###       self.ind_reg ) = crop_tripolar_grid(lon_reg, lat_reg,
###                                           self.clon, self.clat, 
###                                           self.vertex_of_cell,
###                                           self.edge_of_cell)
###    else:
###      (self.Lon, self.Lat, self.lon, self.lat, 
###       self.ind_reg ) = crop_regular_grid(lon_reg, lat_reg, self.Lon, self.Lat)
###
###  def mask_big_triangles(self, do_mask_zeros=True):
###    mask_grid_c = (
###          (self.vlon[self.vertex_of_cell[:,0]] - self.vlon[self.vertex_of_cell[:,1]] )**2
###        + (self.vlon[self.vertex_of_cell[:,0]] - self.vlon[self.vertex_of_cell[:,2]] )**2 
###        + (self.vlat[self.vertex_of_cell[:,0]] - self.vlat[self.vertex_of_cell[:,1]] )**2
###        + (self.vlat[self.vertex_of_cell[:,0]] - self.vlat[self.vertex_of_cell[:,2]] )**2 
###                  ) > 2.*180./np.pi
###    #ipdb.set_trace()
###    if do_mask_zeros:
###      mask_grid_c += self.data==0
###    self.Tri.set_mask(mask_grid_c)
###    return

# //////////////////////////////////////////////////////////////////////////////// 
# ---- classes and methods necessary for Jupyter data viewer
class IconData(object):
  """
  Used by Jupyter
  """
  def __init__(self, 
               # data
               search_str="",
               path_data      = "",
               # original grid
               path_tgrid     = "auto",
               fpath_tgrid    = "auto",
               fpath_fx       = "auto",
               # interpolation
               path_ckdtree   = "",
               path_rgrid     = "auto", # not needed if conventions are followed
               path_sections  = "auto", # not needed if convections are followed
               rgrid_name     = "",
               section_name   = "",
               run            = "auto",
               lon_reg=[-180, 180],
               lat_reg=[-90, 90],
               do_triangulation=True,
              ):


    if rgrid_name=='orig':
      use_tgrid = True
      rgrid_name = ""
    else:
      use_tgrid = False

    # --- paths and file names
    self.path_data     = path_data
    self.path_ckdtree  = path_ckdtree
    if path_rgrid=="auto":
      self.path_rgrid = self.path_ckdtree + 'rectgrids/'
    else:
      self.path_rgrid = path_rgrid
    if path_sections=="auto":
      self.path_sections = self.path_ckdtree + 'sections/'
    else:
      self.path_sections = path_sections
    if run=='auto':
      self.run = self.path_data.split('/')[-2]
    else: 
      self.run = run
    if fpath_fx=='auto':
      self.fpath_fx = self.path_data + self.run + '_fx.nc'
    else:
      self.fpath_fx = fpath_fx
    self.Dgrid = identify_grid(path_grid='', fpath_data=self.fpath_fx)
    if path_tgrid=='auto':
      self.path_tgrid    = path_data
      self.fpath_tgrid   = self.path_data + self.Dgrid['long_name'] + '.nc'
    else:
      self.fpath_tgrid   = fpath_tgrid
      self.path_tgrid    = path_tgrid
    self.Dgrid['fpath_grid'] = self.fpath_tgrid

    for fp in [self.path_data, self.path_ckdtree, 
               self.fpath_tgrid, self.fpath_fx]:
      if not os.path.exists(fp):
        raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

    # --- global variables
    self.interpolate = True
    self.units=dict()
    self.long_name=dict()
    self.data=dict()

    self.lon_reg = lon_reg
    self.lat_reg = lat_reg
    self.use_tgrid = use_tgrid
    self.search_str = search_str

    # --- find ckdtrees fitting for this data set
    sec_fpaths = np.array(
      glob.glob(path_ckdtree+'sections/'+self.Dgrid['name']+'*.npz'))
    sec_names = np.zeros(sec_fpaths.size, '<U200')
    self.sec_fpath_dict = dict()
    for nn, fpath_ckdtree in enumerate(sec_fpaths): 
      ddnpz = np.load(fpath_ckdtree)
      sec_names[nn] = ddnpz['sname']
      self.sec_fpath_dict[sec_names[nn]] = fpath_ckdtree
    self.sec_fpaths = sec_fpaths
    self.sec_names = sec_names

    rgrid_fpaths = np.array(
      glob.glob(path_ckdtree+'rectgrids/'+self.Dgrid['name']+'*.npz'))
    rgrid_names = np.zeros(rgrid_fpaths.size, '<U200')
    self.rgrid_fpath_dict = dict()
    for nn, fpath_ckdtree in enumerate(rgrid_fpaths): 
      ddnpz = np.load(fpath_ckdtree)
      rgrid_names[nn] = ddnpz['sname']
      self.rgrid_fpath_dict[rgrid_names[nn]] = fpath_ckdtree
    self.rgrid_fpaths = rgrid_fpaths
    self.rgrid_names = rgrid_names

    if rgrid_names.size==0:
      raise ValueError('::: Error: Could not find any rgrid-npz-file in %s. :::' 
                        % (path_ckdtree+'rectgrids/'))

    # --- choose rgrid and section
    self.set_rgrid(rgrid_name)
    self.set_section(section_name)

    # --- load grid
    self.load_tgrid()
    self.load_rgrid()
    self.load_vgrid()
    #self.crop_grid(lon_reg=self.lon_reg, lat_reg=self.lat_reg)
    # --- triangulation
    if do_triangulation:
      self.Tri = matplotlib.tri.Triangulation(self.vlon, self.vlat, 
                                              triangles=self.vertex_of_cell)
      self.mask_big_triangles()

    # --- list of variables and time steps / files
    self.get_files_of_timeseries()
    self.get_varnames(self.flist[0])
    self.associate_variables(fpath_data=self.flist[0], skip_vars=[])
    self.get_timesteps()
    return

  def get_files_of_timeseries(self):
    self.times_flist, self.flist = get_files_of_timeseries(self.path_data, self.search_str)
    return 
  
  def get_timesteps(self):
    self.times, self.flist_ts, self.its = get_timesteps(self.flist)
    return
  
  def get_varnames(self, fpath, skip_vars=[]):
    skip_vars = ['clon', 'clat', 'elon', 'elat', 'time', 'depth', 'lev']
    varnames = get_varnames(fpath, skip_vars)
    self.varnames = varnames
    return

  def associate_variables(self, fpath_data, skip_vars=[]):
    fi = Dataset(fpath_data, 'r')
    self.vars = dict()
    for var in self.varnames:
      try:
        units = fi.variables[var].units
      except:
        units = ''
      try:
        long_name = fi.variables[var].long_name
      except:
        long_name = ''
      try:
        coordinates = fi.variables[var].coordinates
      except:
        coordinates = ''
      shape = fi.variables[var].shape
      if (self.nz in shape) or ((self.nz+1) in shape):
        is3d = True
      else:
        is3d = False 
      #print(var, fi.variables[var].shape, is3d)
      IV = IconVariable(var, units=units, long_name=long_name, is3d=is3d, coordinates=coordinates)
      #print('%s: units = %s, long_name = %s'%(IV.name,IV.units,IV.long_name))
      self.vars[var] = IV
      #setattr(self, var, IV)
    fi.close()
    return

  def set_rgrid(self, rgrid_name):
    if rgrid_name=="":
      # take first of list
      self.rgrid_fpath = self.rgrid_fpaths[0]
      self.rgrid_name  = self.rgrid_names[0]
    else:
      if rgrid_name in self.rgrid_names:
        self.rgrid_fpath = self.rgrid_fpaths[
          np.where(self.rgrid_names==rgrid_name)[0][0] ]
        self.rgrid_name  = rgrid_name
      else: 
        self.rgrid_fpath = self.rgrid_fpaths[0]
        self.rgrid_name  = self.rgrid_names[0]
        print('::: Error: %s could not be found. :::' 
              % (rgrid_name))
        print('You could have chosen one from:')
        print(self.rgrid_names)
        raise ValueError('::: Stopping! :::')
    return

  def set_section(self, sec_name):
    if sec_name=="":
      # take first of list
      self.sec_fpath = self.sec_fpaths[0]
      self.sec_name  = self.sec_names[0]
    else:
      if sec_name in self.sec_names:
        self.sec_fpath = self.sec_fpaths[
          np.where(self.sec_names==sec_name)[0][0] ]
        self.sec_name  = sec_name
      else: 
        self.sec_fpath = self.sec_fpaths[0]
        self.sec_name  = self.sec_names[0]
        print('::: Error: %s could not be found. :::' 
              % (sec_name))
        print('You could have chosen one from:')
        print(self.sec_names)
        raise ValueError('::: Stopping! :::')
    return
  
  def show_grid_info(self):
    print('------------------------------------------------------------')
    fpaths = glob.glob(self.path_rgrid+self.Dgrid['name']+'*.npz')
    print('regular grid files:')
    print(self.path_rgrid)
    for fp in fpaths:
      ddnpz = np.load(fp)
      info = ('{:40s} {:20s}').format(fp.split('/')[-1]+':', ddnpz['sname'])
      print(info)
    
    print('------------------------------------------------------------')
    fpaths = glob.glob(self.path_sections+self.Dgrid['name']+'*.npz')
    print('section files:')
    print(self.path_sections)
    for fp in fpaths:
      ddnpz = np.load(fp)
      info = ('{:40s} {:20s}').format(fp.split('/')[-1]+':', ddnpz['sname'])
      print(info)
    
    print('------------------------------------------------------------') 
    return

  def load_vgrid(self, lon_reg='all', lat_reg='all'):
    # --- vertical levels
    f = Dataset(self.fpath_fx, 'r')
    #self.clon = f.variables['clon'][:] * 180./np.pi
    #self.clat = f.variables['clat'][:] * 180./np.pi
    self.depthc = f.variables['depth'][:]
    self.depthi = f.variables['depth_2'][:]

    # --- the variables prism_thick_flat_sfc_c seem to be corrupted in fx file
    self.prism_thick_flat_sfc_c = f.variables['prism_thick_flat_sfc_c'][:]
    self.constantPrismCenters_Zdistance = f.variables['constantPrismCenters_Zdistance'][:]
    self.dzw = f.variables['prism_thick_flat_sfc_c'][:]
    self.dzt = f.variables['constantPrismCenters_Zdistance'][:]
    self.nz = self.depthc.size
    #for var in f.variables.keys():
    #  print(var)
    #  print(f.variables[var][:].max())
    #mybreak()
    f.close()
    return

    # --- tripolar grid
    #self.load_tripolar_grid()

  def load_rgrid(self, lon_reg='all', lat_reg='all'):
    # --- rectangular grid
    ddnpz = np.load(self.rgrid_fpath)
    #self.dckdtree = ddnpz['dckdtree']
    #self.ickdtree = ddnpz['ickdtree']
    self.lon = ddnpz['lon']
    self.lat = ddnpz['lat']
    self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)

      ## --- triangle grid
      #self.ind_reg = np.where( 
      #    (self.clon >  self.lon_reg[0]) 
      #  & (self.clon <= self.lon_reg[1]) 
      #  & (self.clat >  self.lat_reg[0]) 
      #  & (self.clat <= self.lat_reg[1]) )[0]
      #self.clon = self.clon[self.ind_reg]
      #self.clat = self.clat[self.ind_reg]

      ## triangulation
      #ntr = self.clon.size
      #f = Dataset(self.fpath_fx, 'r')
      #clon_bnds = f.variables['clon_bnds'][:] * 180./np.pi
      #clat_bnds = f.variables['clat_bnds'][:] * 180./np.pi
      #clon_bnds = clon_bnds[self.ind_reg,:]
      #clat_bnds = clat_bnds[self.ind_reg,:]
      #f.close()

      #clon_bnds_rs = clon_bnds.reshape(ntr*3)
      #clat_bnds_rs = clat_bnds.reshape(ntr*3)
      #triangles = np.arange(ntr*3).reshape(ntr,3)
      #self.Tri = matplotlib.tri.Triangulation(
      #  clon_bnds_rs, clat_bnds_rs, triangles=triangles)

      #mask_grid = (   (clon_bnds[:,0]-clon_bnds[:,1])**2 
      #              + (clon_bnds[:,0]-clon_bnds[:,2])**2 
      #              + (clat_bnds[:,0]-clat_bnds[:,1])**2 
      #              + (clat_bnds[:,0]-clat_bnds[:,2])**2 
      #            ) > 2.*180./np.pi
      #self. maskTri = mask_grid
      #self.Tri.set_mask(self.maskTri)
    return

#  def load_hsnap(self, varnames, step_snap=0, iz=0):
#    self.step_snap = step_snap
#    it = self.its[step_snap]
#    self.it = it
#    self.iz = iz
#    fpath = self.flist_ts[step_snap]
#    #print("Using data set %s" % fpath)
#    f = Dataset(fpath, 'r')
#    for var in varnames:
#      print("Loading %s" % (var))
#      if f.variables[var].ndim==2:
#        data = f.variables[var][it,:]
#      else:
#        data = f.variables[var][it,iz,:]
#      self.long_name[var] = f.variables[var].long_name
#      self.units[var] = f.variables[var].units
#      self.data[var] = var
#
#      #if self.interpolate:
#      if self.use_tgrid:
#        data = data[self.ind_reg] 
#      else:
#        data = icon_to_regular_grid(data, self.Lon.shape, 
#                            distances=self.dckdtree, inds=self.ickdtree)
#
#      # add data to IconData object
#      data[data==0.] = np.ma.masked
#      setattr(self, var, data)
#    f.close()
#    return

#  def load_vsnap(self, varnames, fpath_ckdtree, step_snap=0,):
#    self.step_snap = step_snap
#    it = self.its[step_snap]
#    self.it = it
#    #self.iz = iz
#    fpath = self.flist_ts[step_snap]
#    print("Using data set %s" % fpath)
#
#    ddnpz = np.load(fpath_ckdtree)
#    #dckdtree = ddnpz['dckdtree']
#    #ickdtree = ddnpz['ickdtree'] 
#    self.lon_sec = ddnpz['lon_sec'] 
#    self.lat_sec = ddnpz['lat_sec'] 
#    self.dist_sec  = ddnpz['dist_sec'] 
#
#    f = Dataset(fpath, 'r')
#    for var in varnames:
#      print("Loading %s" % (var))
#      if f.variables[var].ndim==2:
#        print('::: Warning: Cannot do section of 2D variable %s! :::'%var)
#      else:
#        nz = f.variables[var].shape[1]
#        data_sec = np.ma.zeros((nz,self.dist_sec.size))
#        for k in range(nz):
#          #print('k = %d/%d'%(k,nz))
#          data = f.variables[var][it,k,:]
#          data_sec[k,:] = apply_ckdtree(data, fpath_ckdtree)
#
#        self.long_name[var] = f.variables[var].long_name
#        self.units[var] = f.variables[var].units
#        self.data[var] = var
#
#        # add data to IconData object
#        data_sec[data_sec==0.] = np.ma.masked
#        setattr(self, var, data_sec)
#    f.close()
#    return

  def load_tgrid(self):
    (self.clon, self.clat, self.vlon, self.vlat,
     self.elon, self.elat, self.vertex_of_cell,
     self.edge_of_cell ) = load_tripolar_grid(self.fpath_tgrid)
    f = Dataset(self.fpath_tgrid, 'r')
    self.cell_area = f.variables['cell_area'][:]
    f.close()
    return

  
  def crop_grid(self, lon_reg, lat_reg):
    """ Crop all cell related variables (data, clon, clat, vertex_of_cell, edge_of_cell to regin defined by lon_reg and lat_reg.
    """
    # --- crop tripolar grid
    (self.clon, self.clat,
     self.vertex_of_cell, self.edge_of_cell,
     self.ind_reg_tri ) = crop_tripolar_grid(lon_reg, lat_reg,
                                         self.clon, self.clat, 
                                         self.vertex_of_cell,
                                         self.edge_of_cell)
    # --- crop rectangular grid
    (self.Lon, self.Lat, self.lon, self.lat, 
     self.ind_reg_rec ) = crop_regular_grid(lon_reg, lat_reg, self.Lon, self.Lat)
    return

  def mask_big_triangles(self):
    mask_bt = (
      (self.vlon[self.vertex_of_cell[:,0]] - self.vlon[self.vertex_of_cell[:,1]])**2
    + (self.vlon[self.vertex_of_cell[:,0]] - self.vlon[self.vertex_of_cell[:,2]])**2
    + (self.vlat[self.vertex_of_cell[:,0]] - self.vlat[self.vertex_of_cell[:,1]])**2
    + (self.vlat[self.vertex_of_cell[:,0]] - self.vlat[self.vertex_of_cell[:,2]])**2
               ) > 2.*180./np.pi
    self.Tri.set_mask(mask_bt)
    self.maskTri = mask_bt
    return

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
      self.hpc = trishade(IcD.Tri, 
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

# ================================================================================ 
# Quick Plots
# ================================================================================ 

# --------------------------------------------------------------------------------
# Horizontal plots
# --------------------------------------------------------------------------------
def qp_hplot(fpath, var, IcD='none', depth=-1e33, iz=0, it=0,
              rgrid_name="orig",
              path_ckdtree="",
              clim='auto', cincr=-1., cmap='auto',
              xlim=[-180,180], ylim=[-90,90], projection='none',
              sasp=0.543,
              title='auto', xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              ):


  for fp in [fpath]:
    if not os.path.exists(fp):
      raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  # get fname and path_data from fpath
  fname = fpath.split('/')[-1]
  path_data = ''
  for el in fpath.split('/')[1:-1]:
    path_data += '/'
    path_data += el
  path_data += '/'

  # --- set-up grid and region if not given to function
  if isinstance(IcD,str) and IcD=='none':
    IcD = IconData(
                   search_str   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   rgrid_name   = rgrid_name
                  )
  else:
    print('Using given IcD!')

  if depth!=-1e33:
    iz = np.argmin((IcD.depthc-depth)**2)
  IaV = IcD.vars[var]
  step_snap = it

  # synchronize with Jupyter update_fig
  # --- load data 
  IaV.load_hsnap(fpath=IcD.flist_ts[step_snap], 
                      it=IcD.its[step_snap], 
                      iz=iz,
                      step_snap = step_snap
                     ) 
  # --- interpolate data 
  if not IcD.use_tgrid:
    IaV.interp_to_rectgrid(fpath_ckdtree=IcD.rgrid_fpath)
  # --- crop data

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
  else:
    ccrs_proj = getattr(ccrs, projection)()

  # --- do plotting
  (ax, cax, 
   mappable,
   Dstr
  ) = hplot_base(
              IcD, IaV, 
              ax=ax, cax=cax,
              clim=clim, cmap=cmap, cincr=cincr,
              xlim=xlim, ylim=ylim,
              title='auto', 
              projection=projection,
              #use_tgrid=IcD.use_tgrid,
              logplot=logplot,
              sasp=sasp,
             )


  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

def qp_vplot(fpath, var, IcD='none', it=0,
              sec_name="specify_sec_name",
              path_ckdtree="",
              var_fac=1.,
              clim='auto', cincr=-1., cmap='auto',
              xlim=[-180,180], ylim=[-90,90], projection='none',
              sasp=0.543,
              title='auto', xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              log2vax=False,
              mode_load='normal',
              ):


  for fp in [fpath]:
    if not os.path.exists(fp):
      raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  # get fname and path_data from fpath
  fname = fpath.split('/')[-1]
  path_data = ''
  for el in fpath.split('/')[1:-1]:
    path_data += '/'
    path_data += el
  path_data += '/'

  # --- load data set
  if isinstance(IcD,str) and IcD=='none':
    IcD = IconData(
                   search_str   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   #rgrid_name   = rgrid_name
                  )
  else:
    print('Using given IcD!')

  IaV = IcD.vars[var]
  step_snap = it

  # --- load data
  # FIXME: MOC and ZAVE cases could go into load_vsnap
  if sec_name.endswith('moc'):
    IaV.load_moc(
                   fpath=IcD.flist_ts[step_snap], 
                   it=IcD.its[step_snap], 
                   step_snap = step_snap
                  ) 
  elif sec_name.startswith('zave'):
    basin      = sec_name.split(':')[1]
    rgrid_name = sec_name.split(':')[2]
    IaV.lat_sec, IaV.data = zonal_average(
                                   fpath_data=IcD.flist_ts[step_snap], 
                                   var=var, basin=basin, it=it,
                                   fpath_fx=IcD.fpath_fx, 
                                   fpath_ckdtree=IcD.rgrid_fpaths[
                                     np.where(IcD.rgrid_names==rgrid_name)[0][0]]
                                         )
  else:
    sec_fpath = IcD.sec_fpaths[np.where(IcD.sec_names==sec_name)[0][0] ]
    IaV.load_vsnap(
                   fpath=IcD.flist_ts[step_snap], 
                   fpath_ckdtree=sec_fpath,
                   it=IcD.its[step_snap], 
                   step_snap = step_snap
                  ) 

  IaV.data *= var_fac

  # --- do plotting
  (ax, cax, 
   mappable,
   Dstr
  ) = vplot_base(
                 IcD, IaV, 
                 ax=ax, cax=cax,
                 clim=clim, cmap=cmap, cincr=cincr,
                 title='auto', 
                 log2vax=log2vax,
                 logplot=logplot,
                )


  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

##def qp_hor_plot(fpath, var, IC='none', iz=0, it=0,
##              grid='orig', 
##              path_rgrid="",
##              clim='auto', cincr='auto', cmap='auto',
##              xlim=[-180,180], ylim=[-90,90], projection='none',
##              title='auto', xlabel='', ylabel='',
##              verbose=1,
##              ax='auto', cax=1,
##              ):
##
##
##  # --- load data
##  fi = Dataset(fpath, 'r')
##  data = fi.variables[var][it,iz,:]
##  if verbose>0:
##    print('Plotting variable: %s: %s' % (var, IC.long_name)) 
##
##  # --- set-up grid and region if not given to function
##  if isinstance(IC,str) and clim=='none':
##    pass
##  else:
##    IC = IconDataFile(fpath, path_grid='/pool/data/ICON/oes/input/r0002/')
##    IC.identify_grid()
##    IC.load_tripolar_grid()
##    IC.data = data
##    if grid=='orig':
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      IC.Tri = matplotlib.tri.Triangulation(IC.vlon, IC.vlat, 
##                                            triangles=IC.vertex_of_cell)
##      IC.mask_big_triangles()
##      use_tgrid = True
##    else: 
##      # --- rectangular grid
##      if not os.path.exists(path_rgrid+grid):
##        raise ValueError('::: Error: Cannot find grid file %s! :::' % 
##          (path_rgrid+grid))
##      ddnpz = np.load(path_rgrid+grid)
##      IC.lon, IC.lat = ddnpz['lon'], ddnpz['lat']
##      IC.Lon, IC.Lat = np.meshgrid(IC.lon, IC.lat)
##      IC.data = icon_to_regular_grid(IC.data, IC.Lon.shape, 
##                          distances=ddnpz['dckdtree'], inds=ddnpz['ickdtree'])
##      IC.data[IC.data==0] = np.ma.masked
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      use_tgrid = False
##  IC.data = IC.data[IC.ind_reg]
##
##  IC.long_name = fi.variables[var].long_name
##  IC.units = fi.variables[var].units
##  IC.name = var
##
##  ax, cax, mappable = hplot_base(IC, var, clim=clim, title=title, 
##    projection=projection, use_tgrid=use_tgrid)
##
##  fi.close()
##
##  # --- output
##  FigInf = dict()
##  FigInf['fpath'] = fpath
##  FigInf['long_name'] = long_name
##  FigInf['IC'] = IC
##  #ipdb.set_trace()
##  return FigInf

# ================================================================================ 
# ================================================================================ 
class QuickPlotWebsite(object):
  """ Creates a website where the quick plots can be shown.

Minimal example:

# --- setup
qp = QuickPlotWebsite(
  title='pyicon Quick Plot', 
  author='Nils Brueggemann', 
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  fpath_css='./mycss2.css'
  )

# --- content
for i in range(1,11):
  qp.add_section('Section %d' % i)
  for j in range(1,4):
    qp.add_subsection('Subsection %d.%d' % (i,j))
    qp.add_paragraph(('').join(['asdf %d'%(i)]*10))
    qp.add_fig('./pics/','fig_01.png')
qp.write_to_file()
  """

  def __init__(self, title='Quick Plot', author='', date='', 
               info='', path_data='',
               fpath_css='', fpath_html='./qp_index.html'):
    self.author = author 
    self.title = title
    self.date = date
    self.info = info
    self.path_data = path_data
    self.fpath_css = fpath_css
    self.fpath_html = fpath_html

    self.first_add_section_call = True

    self.main = ""
    self.toc = ""

    self.header = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">

<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta http-equiv="Content-Style-Type" content="text/css" />
  <meta name="generator" content="pyicon" />
  <meta name="author" content="{author}" />
  <title>{title}</title>
  <style type="text/css">code{{white-space: pre;}}</style>
  <link rel="stylesheet" href="{fpath_css}" type="text/css" />
</head>

<body>

<div id="header">
<h1 class="title">{title}</h1>
<p> {author} | {date} | {path_data} </p>
<p> {info} </>
</div>

""".format(author=self.author, title=self.title, 
           date=self.date, 
           path_data=self.path_data,
           info=self.info,
           fpath_css=self.fpath_css, 
          )

#<div id="header">
#<h1 class="title">{title}</h1>
#<h2 class="author">{author}</h2>
#<h3 class="date">{date}</h3>
#</div>

    self.footer = """
</body>
</html>
"""
  
  def add_section(self, title='Section'):
    # --- add to main
    href = title.replace(' ', '-')
    self.main += '\n'
    self.main += "<h1 id=\"{href}\">{title}</h1>\n".format(title=title, href=href)
    # --- add to toc
    if self.first_add_section_call:
      self.first_add_section_call = False
      self.toc += """
<div id="TOC">
<ul>
"""
    else:
      self.toc += '</ul></li> \n'
    self.toc += '<li><a href="#{href}">{title}</a><ul>\n'.format(title=title, href=href)
    return

  def add_subsection(self, title='Subsection'):
    # --- add to main
    href = title.replace(' ', '-')
    #self.main += '\n'
    self.main += "  <h2 id=\"{href}\">{title}</h2>\n".format(title=title, href=href)
    # --- add to toc
    self.toc += '<li><a href="#{href}">{title}</a></li>\n'.format(title=title, href=href)
    return

  def add_paragraph(self, text=''):
    self.main += '    <p>'
    self.main += text
    self.main += '    </p>'
    self.main += '\n'
    return
   
  def add_fig(self, fpath, width="1000"):
    self.main += '    <div class="figure"> <img src="{fpath}" width="{width}" /> </div>'.format(fpath=fpath, width=width)
    self.main += '\n'
    return
  
  def close_toc(self):
    # --- close toc
    self.toc += """</ul></li>
</ul>
</div>

"""
    return

  def write_to_file(self):
    # --- close toc
    self.close_toc()

    # --- write to output file
    f = open(self.fpath_html, 'w')
    f.write(self.header)
    f.write(self.toc)
    f.write(self.main)
    f.write(self.footer)
    f.close()
    return

# ================================================================================ 
# ================================================================================ 

def shade(x, y, datai,
            ax='auto', cax=0,
            cmap='auto',
            cincr=-1.,
            norm=None,
            rasterized=True,
            clim=[None, None],
            extend='both',
            conts=None,
            nclev='auto',
            cint='auto',
            contcolor='k',
            contthick=0.,
            contfs=None,
            contlw=1.,
            use_pcol=True,
            cbticks='auto',
            adjust_axlims=True,
            bmp=None,
            transform=None,
            logplot=False,
            cbtitle='',
            edgecolor='none',
         ):
  """ Makes a nice pcolor(mesh) plot.

last change:
----------
2016-08-23
  """
  # mask 0 and negative values in case of log plot
  data = 1.*datai
  if logplot and isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.ma.masked
    data = np.ma.log10(data) 
  elif logplot and not isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.nan
    data = np.log10(data) 

  #clims
  if isinstance(clim, str) and clim=='auto':
    clim = [None, None]
  elif isinstance(clim, str) and clim=='sym':
    clim = np.abs(data).max()
  clim=np.array(clim)
  if clim.size==1:
    clim = np.array([-1, 1])*clim
  if clim[0] is None:
    clim[0] = data.min()
  if clim[1] is None:
    clim[1] = data.max()

  if (clim[0]==-clim[1]) and cmap=='auto':
    cmap = 'RdBu_r'
  elif cmap=='auto':
    #cmap = 'viridis'
    cmap = 'RdYlBu_r'
  if isinstance(cmap, str):
    cmap = getattr(plt.cm, cmap)

  if cincr>0.:
    levs = np.arange(clim[0], clim[1]+cincr, cincr)
    norm = matplotlib.colors.BoundaryNorm(boundaries=levs, ncolors=cmap.N)
  else:
    norm = None

  # calculate contour x/y and contour levels if needed
  if conts is None:
    use_cont = False
  elif isinstance(conts,str) and conts=='auto':
    use_cont = True
    if isinstance(nclev,str) and nclev=='auto':
      conts = np.linspace(clim[0], clim[1], 11)
    else:
      conts = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      conts = np.arange(clim[0], clim[1]+cint, cint)
  else:
    use_cont = True
    conts = np.array(conts)

  if contfs is None:
    use_contf=False
  elif isinstance(contfs, str) and contfs=='auto':
    use_contf=True
    use_pcol=False
    if isinstance(nclev,str) and nclev=='auto':
      contfs = np.linspace(clim[0], clim[1], 11)
    else:
      contfs = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      contfs = np.arange(clim[0], clim[1]+cint, cint)
  elif isinstance(contfs, str) and contfs!='auto':
    use_contf=True
    use_pcol=False
    contfs = np.linspace(clim[0], clim[1], int(contfs))
  else:
    use_contf=True
    use_pcol=False
    contfs = np.array(contfs)

  ccrsdict = dict()
  if transform is not None:
    ccrsdict = dict(transform=transform)
    #adjust_axlims = False
    adjust_axlims = True
  
  # make axes if necessary
  if ax == 'auto':
    ax = plt.gca()

  # make x and y 2D
  if x.ndim==1:
    x, y = np.meshgrid(x, y)

  # convert to Basemap maps coordinates
  if bmp is not None:
    x, y = bmp(x, y)
    
  # bring x and y to correct shape for contour
  if (use_cont) or (use_contf):
    if x.shape[1] != data.shape[1]:
      xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
      yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
    else:
      xc = 1.*x
      yc = 1.*y
    
  hs = []
  # pcolor plot
  if use_pcol:
    hm = ax.pcolormesh(x, y, data, 
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        rasterized=rasterized,
                        edgecolor=edgecolor,
                        **ccrsdict
                      )
    hs.append(hm)
  # contourf plot
  elif use_contf:
    hm = ax.contourf(xc, yc, data, contfs,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        extend='both',
                        **ccrsdict
                      )
    # this prevents white lines if fig is saved as pdf
    for cl in hm.collections: 
      cl.set_edgecolor("face")
      cl.set_rasterized(True)
    # add handle to hanlde list
    hs.append(hm)
    # rasterize
    if rasterized:
      zorder = -5
      ax.set_rasterization_zorder(zorder)
      for cl in hm.collections:
        cl.set_zorder(zorder - 1)
        cl.set_rasterized(True)
  else:
    hm = None

  # extra contours
  if use_cont:
    hc = ax.contour(xc, yc, data, conts, colors=contcolor, linewidths=contlw, **ccrsdict)
    try:
      i0 = np.where(hc.levels==contthick)[0][0]
      #hc.collections[i0].set_linewidth(1.5)
      hc.collections[i0].set_linewidth(2.5*contlw)
    except:
      #print "::: Warning: Could not make contour contthick=%g thick. :::" % (contthick)
      pass
    hs.append(hc)

  # colorbar
  if ((cax is not None) and (cax!=0)) and (hm is not None): 
    if cax == 1:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      div = make_axes_locatable(ax)
      cax = div.append_axes("right", size="10%", pad=0.1)
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    # this prevents white lines if fig is saved as pdf
    cb.solids.set_edgecolor("face")
    hs.append(cb)

    # colobar ticks
    if isinstance(cbticks, str) and cbticks=='auto':
      cb.formatter.set_powerlimits((-3, 2))
      tick_locator = ticker.MaxNLocator(nbins=8)
      cb.locator = tick_locator
      cb.update_ticks()
    else:
      if norm is not None:
        cb.set_ticks(norm.boundaries[::2]) 
      cb.formatter.set_powerlimits((-1, 1))
      #cb.formatter.set_scientific(False)
      cb.update_ticks()

  if isinstance(cax, int) and cax==0:
    pass
  else:
    cax.set_title(cbtitle)

  # labels and ticks
  if adjust_axlims:
    ax.locator_params(nbins=5)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
  return hs 

# ================================================================================ 
def trishade(Tri, data,
            ax='auto', cax=0,
            cmap='auto',
            cincr=-1.,
            norm=None,
            rasterized=True,
            clim=[None, None],
            extend='both',
            edgecolor='none',
            conts=None,
            nclev='auto',
            cint='auto',
            contcolor='k',
            contthick=0.,
            contfs=None,
            contlw=1.,
            use_pcol=True,
            adjust_axlims=True,
            bmp=None,
            transform=None,
            logplot=False,
         ):
  """ Makes a nice tripcolor plot.

last change:
----------
2018-03-08
  """
  # mask 0 and negative values in case of log plot
  if logplot and isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.ma.masked
    data = np.ma.log10(data) 
  elif logplot and not isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.nan
    data = np.log10(data) 

  #clims
  if isinstance(clim, str) and clim=='auto':
    clim = [None, None]
  elif isinstance(clim, str) and clim=='sym':
    clim = np.abs(data).max()
  clim=np.array(clim)
  if clim.size==1:
    clim = np.array([-1, 1])*clim
  if clim[0] is None:
    clim[0] = data.min()
  if clim[1] is None:
    clim[1] = data.max()

  if (clim[0]==-clim[1]) and cmap=='auto':
    cmap = 'RdBu_r'
  elif cmap=='auto':
    #cmap = 'viridis'
    cmap = 'RdYlBu_r'
  if isinstance(cmap, str):
    cmap = getattr(plt.cm, cmap)

  if cincr>0.:
    levs = np.arange(clim[0], clim[1]+cincr, cincr)
    norm = matplotlib.colors.BoundaryNorm(boundaries=levs, ncolors=cmap.N)
  else:
    norm = None

  # calculate contour x/y and contour levels if needed
  if conts is None:
    use_cont = False
  elif isinstance(conts,str) and conts=='auto':
    use_cont = True
    if isinstance(nclev,str) and nclev=='auto':
      conts = np.linspace(clim[0], clim[1], 11)
    else:
      conts = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      conts = np.arange(clim[0], clim[1]+cint, cint)
  else:
    use_cont = True
    conts = np.array(conts)

  if contfs is None:
    use_contf=False
  elif isinstance(contfs, str) and contfs=='auto':
    use_contf=True
    use_pcol=False
    if isinstance(nclev,str) and nclev=='auto':
      contfs = np.linspace(clim[0], clim[1], 11)
    else:
      contfs = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      contfs = np.arange(clim[0], clim[1]+cint, cint)
  elif isinstance(contfs, str) and contfs!='auto':
    use_contf=True
    use_pcol=False
    contfs = np.linspace(clim[0], clim[1], int(contfs))
  else:
    use_contf=True
    use_pcol=False
    contfs = np.array(contfs)

  ccrsdict = dict()
  if transform is not None:
    ccrsdict = dict(transform=transform)
    #adjust_axlims = False
    adjust_axlims = True
  
  # make axes if necessary
  if ax == 'auto':
    ax = plt.gca()

  #### make x and y 2D
  ###if x.ndim==1:
  ###  x, y = np.meshgrid(x, y)

  #### convert to Basemap maps coordinates
  ###if bmp is not None:
  ###  x, y = bmp(x, y)
  ###  
  #### bring x and y to correct shape for contour
  ###if (use_cont) or (use_contf):
  ###  if x.shape[1] != data.shape[1]:
  ###    xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
  ###    yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
  ###  else:
  ###    xc = 1.*x
  ###    yc = 1.*y
    
  hs = []
  # pcolor plot
  if use_pcol:

    hm = ax.tripcolor(Tri, data, 
                        edgecolor=edgecolor,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        rasterized=rasterized,
                        **ccrsdict
                      )
    hs.append(hm)
  # contourf plot
  elif use_contf:
    hm = ax.contourf(xc, yc, data, contfs,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        extend='both',
                        **ccrsdict
                      )
    # this prevents white lines if fig is saved as pdf
    for cl in hm.collections: 
      cl.set_edgecolor("face")
    # add handle to hanlde list
    hs.append(hm)
  else:
    hm = None

  # extra contours
  if use_cont:
    hc = ax.contour(xc, yc, data, conts, colors=contcolor, linewidths=contlw, **ccrsdict)
    try:
      i0 = np.where(conts==contthick)[0][0]
      #hc.collections[i0].set_linewidth(1.5)
      hc.collections[i0].set_linewidth(2.5*contlw)
    except:
      #print "::: Warning: Could not make contour contthick=%g thick. :::" % (contthick)
      pass
    hs.append(hc)

  # colorbar
  if ((cax is not None) and (cax!=0)) and (hm is not None): 
    if cax == 1:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      div = make_axes_locatable(ax)
      cax = div.append_axes("right", size="10%", pad=0.1)
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    # this prevents white lines if fig is saved as pdf
    cb.solids.set_edgecolor("face")
    hs.append(cb)

    if norm is None:
      # colobar ticks
      cb.formatter.set_powerlimits((-3, 2))
      tick_locator = ticker.MaxNLocator(nbins=8)
      cb.locator = tick_locator
      cb.update_ticks()
    else:
      cb.set_ticks(norm.boundaries[::2]) 

  # labels and ticks
  if adjust_axlims:
    ax.locator_params(nbins=5)
    ax.set_xlim(Tri.x.min(), Tri.x.max())
    ax.set_ylim(Tri.y.min(), Tri.y.max())
  return hs 

# ================================================================================ 
def arrange_axes( nx,ny,
                  # height of and aspect ratio of subplot
                  asy  = 3.5,
                  sasp = 1.0,
                  # plot colorbar
                  plot_cb = False,
                  # have common x or y axes
                  sharex = True, sharey = True,
                  xlabel = "",   ylabel = "",
                  # additional space left right and above and below axes
                  oxl = 0.1, oxr = 0.0,
                  oyb = 0.0, oyt = 0.0,
                  # factor that increases distance between axes
                  axfac_x = 1., axfac_y = 1.,
                  # kw for axes labels [(a), (b), etc.]
                  axlab_kw = dict(),
                  # figure size and aspect ratio
                  fig_size     = 'auto',
                  fig_asp      = 'auto',
                  fig_size_fac = 1.,
                  # figure title
                  fig_title = None,
                  projection = None,
                  ):
  """
last change:
----------
2015-07-22
 """ 

  # all lengths are in cm
  cm2inch = 0.3937        # to convert cm into inch

  # horizontal standard spaces
  alx = 1.0
  asx = asy / sasp
  adx = 0.5    
  cdx = 0.2
  clx = 0.8
  csx = 0.32 
  
  # vertical standard spaces
  aly = 0.8
  asy = asy
  ady = 0.2  
  aty = 0.6
  fty = 1.               # extra space for figure title (set to zero if fig_title = None)

  # apply manual changes to spaces
  adx = adx * axfac_x 
  ady = ady * axfac_y 
  #cdx = cdx * axfac_x   # this is a fix I do not understand why adxv is set to cdx if icbspace==True
  clx = clx * axfac_x

  if fig_title==None:
    fty = 0.

  # make vector of plot_cb if it has been true or false before
  # plot_cb can have values [{1}, 0] 
  # with meanings:
  #   1: plot cb; 
  #   0: do not plot cb
  if isinstance(plot_cb, bool) and (plot_cb==True):
    plot_cb = np.ones((nx,ny))  
    nohcb = False
  elif isinstance(plot_cb, bool) and (plot_cb==False):
    plot_cb = np.zeros((nx,ny))
    nohcb = True
  else:
    plot_cb = np.array(plot_cb)
    if plot_cb.size!=nx*ny:
      raise ValueError('Vector plot_cb has wrong length!')
    if plot_cb.shape[0]==nx*ny:
      plot_cb = plot_cb.reshape(ny,nx).transpose()
    elif plot_cb.shape[0]==ny:
      plot_cb = plot_cb.transpose()
    nohcb = False

  if not isinstance(projection, list):
    projection = [projection]*nx*ny

  # make spaces vectors
  # horizontal
  alxv = np.array([alx]*(nx))
  asxv = np.array([asx]*(nx))
  adxv = np.array([adx]*(nx))
  clxv = np.array([clx]*(nx))
  csxv = np.array([csx]*(nx))

  icbspace = plot_cb.sum(axis=1)>0
  csxv[icbspace==False] = 0.0
  clxv[icbspace==False] = 0.0
  adxv[icbspace==True ] = cdx
  if sharey:
    alxv[1:] = 0.0  

  # vertical
  alyv = np.array([aly]*(ny))
  asyv = np.array([asy]*(ny))
  adyv = np.array([ady]*(ny))
  atyv = np.array([aty]*(ny))

  if sharex:
    alyv[:-1] = 0.0

  # calculate figure size
  fw_auto = ( oxl + (alxv+asxv+adxv+csxv+clxv).sum() + oxr       )
  fh_auto = ( oyb + (alyv+asyv+adyv+atyv).sum()      + oyt + fty )
  if fig_size == 'auto':
    fw = fw_auto 
    fh = fh_auto 
  elif fig_size == 'dina4pt':
    fw = 21.0
    fh = 29.7
  elif fig_size == 'dina4ls':
    fw = 29.7
    fh = 21.0
  elif fig_size == 'jpo':
    fw = 15.5
    if fig_asp == 'auto':
      fh = fh_auto
    else:
      fh = fw*fig_asp
  elif isinstance( fig_size, (int,float) ):
    fw = fig_size
    if fig_asp == 'auto':
      fh = fh_auto
    else:
      fh = fw*fig_asp

  # make figure
  fasp = fh/fw
  hcf = plt.figure(figsize=(fw*cm2inch*fig_size_fac, fh*cm2inch*fig_size_fac))

  if not fig_title == None:
    hcf.suptitle(fig_title)

  # handle for axes
  hca = [0]*(nx*ny) 
  hcb = [0]*(nx*ny)

  kk = -1
  for jj in range(ny):
    for ii in range(nx):
      kk += 1

      # set axes x offspring
      if ii == 0:
        oxa = oxl + alxv[ii]
      else:
        oxa = oxa + alxv[ii] + (asxv+adxv+csxv+clxv)[ii-1]

      # set axes y offsping
      #if jj == 0 and ii == 0:
      #  oya = oyb + alyv[jj]
      #elif jj != 0 and ii == 0:
      #  oya = oya + alyv[jj] + (asyv+adyv+atyv)[jj-1]

      if jj == 0 and ii == 0:
        oya = fh - oyt - fty - (atyv+asyv)[jj]
      elif jj != 0 and ii == 0:
        oya =      oya - alyv[jj-1] - (adyv+atyv+asyv)[jj]

      # set colorbar x offspring
      oxc = oxa + (asxv+adxv)[ii]

      # calculated rectangles for axes and colorbar
      rect   = np.array([oxa, oya/fasp, asxv[ii], asyv[jj]/fasp])/fw
      rectcb = np.array([oxc, oya/fasp, csxv[ii], asyv[jj]/fasp])/fw
      
      # plot axes
      if projection[kk] is None:
        hca[kk] = plt.axes(rect, xlabel=xlabel, ylabel=ylabel)
      else:
        hca[kk] = plt.axes(rect, xlabel=xlabel, ylabel=ylabel, projection=projection[kk])

      # delet labels for shared axes
      if sharex and jj!=ny-1:
        hca[kk].ticklabel_format(axis='x',style='plain',useOffset=False)
        hca[kk].tick_params(labelbottom='off')
        hca[kk].set_xlabel('')

      if sharey and ii!=0:
        hca[kk].ticklabel_format(axis='y',style='plain',useOffset=False)
        hca[kk].tick_params(labelleft='off')
        hca[kk].set_ylabel('')

      # plot colorbars
      if plot_cb[ii,jj] == 1:
        hcb[kk] = plt.axes(rectcb, xticks=[])
        hcb[kk].yaxis.tick_right()

  # add letters for subplots
  if axlab_kw is not None:
    hca = axlab(hca, fontdict=axlab_kw)
  
  # return axes handles
  if nohcb:
    #plotsettings(hca)
    return hca, hcb
  else:
    #plotsettings(hca,hcb)
    return hca, hcb

# ================================================================================ 
def axlab(hca, figstr=[], posx=[-0.0], posy=[1.08], fontdict=None):
  """
input:
----------
  hca:      list with axes handles
  figstr:   list with strings that label the subplots
  posx:     list with length 1 or len(hca) that gives the x-coordinate in ax-space
  posy:     list with length 1 or len(hca) that gives the y-coordinate in ax-space
last change:
----------
2015-07-21
  """

  # make list that looks like [ '(a)', '(b)', '(c)', ... ]
  if len(figstr)==0:
    lett = "abcdefghijklmnopqrstuvwxyz"
    lett = lett[0:len(hca)]
    figstr = ["z"]*len(hca)
    for nn, ax in enumerate(hca):
      figstr[nn] = "(%s)" % (lett[nn])
  
  if len(posx)==1:
    posx = posx*len(hca)
  if len(posy)==1:
    posy = posy*len(hca)
  
  # draw text
  for nn, ax in enumerate(hca):
    ht = hca[nn].text(posx[nn], posy[nn], figstr[nn], 
                      transform = hca[nn].transAxes, 
                      horizontalalignment = 'right',
                      fontdict=fontdict)
    # add text handle to axes to give possibility of changing text properties later
    # e.g. by hca[nn].axlab.set_fontsize(8)
    hca[nn].axlab = ht
  return hca

