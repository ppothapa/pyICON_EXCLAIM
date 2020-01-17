import sys
import numpy as np
from netCDF4 import Dataset
from ipdb import set_trace as mybreak

# --- dummy object to collect variables
class Obj(object):
  def __init__(self):
    return

# --- function to load the tripolar grid
def load_grid(fpath_tgrid):
  IcD = Obj()
  IcD.fpath_tgrid = fpath_tgrid
  
  f = Dataset(IcD.fpath_tgrid, 'r')
  IcD.orientation_of_normal = f.variables['orientation_of_normal'][:].transpose()
  IcD.edge_vertices = f.variables['edge_vertices'][:].transpose()-1
  IcD.edge_of_cell = f.variables['edge_of_cell'][:].transpose()-1
  IcD.clon = f.variables['clon'][:]*180./np.pi
  IcD.clat = f.variables['clat'][:]*180./np.pi
  IcD.elon = f.variables['elon'][:]*180./np.pi
  IcD.elat = f.variables['elat'][:]*180./np.pi
  IcD.vlon = f.variables['vlon'][:]*180./np.pi
  IcD.vlat = f.variables['vlat'][:]*180./np.pi
  IcD.ne_all = IcD.elon.size
  f.close()
  return IcD

# --- main function to derive section edge
def find_section_edges_orientation(IcD, lon1, lat1, lon2, lat2, dlon=10., dlat=10.):

  # --- renaming 
  elon = IcD.elon
  elat = IcD.elat
  clon = IcD.clon
  clat = IcD.clat
  vlon = IcD.vlon
  vlat = IcD.vlat

  # --- define domain
  if lon1==lon2:
    if (lat1<lat2):
      ireg = np.where((clon>=lon1) & (clon<=lon2+dlon) & (clat<=lat2) & (clat>=lat1))[0]
    elif (lat1>lat2):
      ireg = np.where((clon>=lon1) & (clon<=lon2+dlon) & (clat>=lat2) & (clat<=lat1))[0]
  elif lat1==lat2:
    if (lon1<lon2):
      ireg = np.where((clon>=lon1) & (clon<=lon2) & (clat<=lat2+dlat) & (clat>=lat1))[0]
    else:
      ireg = np.where((clon<=lon1) & (clon>=lon2) & (clat<=lat2+dlat) & (clat>=lat1))[0]
  elif (lon1!=lon2) and (lat1!=lat2):
    g1 = (lat2-lat1)/(lon2-lon1)*(clon-lon1) + lat1
    g2 = (lat2-lat1)/(lon2-lon1)*(clon-lon1-dlon) + lat1
    if   (lon1<lon2) and (lat1<lat2):
      ireg = np.where((clat<=g1) & (clat<=lat2) & (clat>=g2) & (clat>=lat1))[0]
    elif (lon1>lon2) and (lat1<lat2):
      ireg = np.where((clat>=g1) & (clat<=lat2) & (clat<=g2) & (clat>=lat1))[0]
    elif (lon1<lon2) and (lat1>lat2):
      ireg = np.where((clat>=g1) & (clat>=lat2) & (clat<=g2) & (clat<=lat1))[0]
    elif (lon1>lon2) and (lat1>lat2):
      ireg = np.where((clat<=g1) & (clat>=lat2) & (clat>=g2) & (clat<=lat1))[0]
  else:
    raise ValueError('::: Unappropriate choice of lon1, lon2, lat1, lat2! :::')

  if ireg.size==0:
    raise ValueError('::: No points in domain! Choose different points or dlon, dlat.')
  
  # --- find all vertices and edges that belong to domain
  # ------ all edges that belong to the cells of domain
  iedge = IcD.edge_of_cell[ireg]
  iedge = iedge.reshape(iedge.size)
  edge_orient = IcD.orientation_of_normal[ireg] 
  edge_orient = edge_orient.reshape(iedge.size)
  # ------ edges that appear only once and which are thus domain boundaries
  iedge_out, ind, cnts = np.unique(iedge, return_index=True, return_counts=True)
  iedge_out = iedge_out[cnts==1]
  edge_orient = edge_orient[ind]
  edge_orient = edge_orient[cnts==1]
  
  # ----- vertices which correspond to edges
  iv_peri_gd = IcD.edge_vertices[iedge_out]
  
  # --- find start edge and vertex
  # ------ find first candidate
  ip1 = np.unravel_index(np.argmin((vlon[iv_peri_gd]-lon1)**2+(vlat[iv_peri_gd]-lat1)**2), iv_peri_gd.shape)
  # ------ find second candidate
  iv_peri_gd_tmp = 1*iv_peri_gd
  iv_peri_gd_tmp[ip1[0],:] = -1
  ip2 = np.unravel_index(np.argmin((vlon[iv_peri_gd_tmp]-lon1)**2+(vlat[iv_peri_gd_tmp]-lat1)**2), iv_peri_gd.shape)
  # ----- decide between two candidates whether the other edge vertex is closer to end point
  vlon1 = vlon[iv_peri_gd[ip1[0],1-ip1[1]]]
  vlat1 = vlat[iv_peri_gd[ip1[0],1-ip1[1]]]
  vlon2 = vlon[iv_peri_gd[ip2[0],1-ip2[1]]]
  vlat2 = vlat[iv_peri_gd[ip2[0],1-ip2[1]]]
  itmp = np.argmin((np.array([vlon1,vlon2])-lon2)**2+(np.array([vlat1,vlat2])-lat2)**2)
  if itmp==0:
    ie_d = ip1[0]
    iv_g = iv_peri_gd[ie_d, ip1[1]]
    iv_next_g = iv_peri_gd[ie_d, 1-ip1[1]]
  else:
    ie_d = ip2[0]
    iv_g = iv_peri_gd[ie_d, ip2[1]]
    iv_next_g = iv_peri_gd[ie_d, 1-ip2[1]]
  
  # --- go around domain
  iedge_out_sorted_gd = np.zeros(iedge_out.size, dtype=int)
  ivert_out_sorted_gd = np.zeros(iedge_out.size+1, dtype=int)
  edge_orient_sorted = np.zeros(iedge_out.size, dtype=int)
  iedge_out_sorted_gd[0] = iedge_out[ie_d]
  ivert_out_sorted_gd[0] = iv_g
  ivert_out_sorted_gd[1] = iv_next_g
  edge_orient_sorted[0] = edge_orient[ie_d]
  for nn in range(iedge_out_sorted_gd.size-1):
    # ----- delete old vertices such that they are not found again
    iv_peri_gd[ie_d,:] = -1
    # ----- find next edge by common vertex between this and next edge
    ie_d = np.where(iv_peri_gd==iv_next_g)[0]
    iedge_out_sorted_gd[nn+1] = iedge_out[ie_d]
    edge_orient_sorted[nn+1] = edge_orient[ie_d]
    # ----- find next vertex as other vertex of edge
    ind_01 = np.where(iv_peri_gd==iv_next_g)[1]
    iv_next_g  = iv_peri_gd[ie_d, 1-ind_01]
    ivert_out_sorted_gd[nn+2] = iv_next_g
  
  # --- reduce perimeter to section by cutting it at final point
  ie_final_d = np.argmin((vlon[ivert_out_sorted_gd]-lon2)**2+(vlat[ivert_out_sorted_gd]-lat2)**2)
  iedge_section = iedge_out_sorted_gd[:ie_final_d]
  ivert_section = ivert_out_sorted_gd[:ie_final_d]
  edge_orient_section = edge_orient_sorted[:ie_final_d]

  edge_mask = np.zeros((IcD.elon.size))
  edge_mask[iedge_section] = edge_orient_section

  return edge_mask

# --- vector containing all section objects
Ms = []

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# START USER INPUT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# --- path to triploar grid file
fpath_tgrid = '/Users/nbruggemann/work/icon_playground/icon_r2b4_test_data/icon_08/icon-oes/experiments/nib0002/OceanOnly_Icos_0158km_etopo40.nc'
# --- path to output netcdf file
fpath_ncfile_out = 'test_icon_bstr_preb.nc'

# --- sec 1
M = Obj()
M.lon1, M.lat1 = -110, -20
M.lon2, M.lat2 = -120, -50
M.dlon, M.dlat = 10., 10.
M.name = 'test_section1'
Ms.append(M)

# --- sec 2
M = Obj()
M.lon1, M.lat1 = -110, -20
M.lon2, M.lat2 = -120, -50
M.dlon, M.dlat = 10., 10.
M.name = 'test_section2'
Ms.append(M)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# END USER INPUT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

# --- load tripolar grid
IcD = load_grid(fpath_tgrid)

# --- open netcdf file
fo = Dataset(fpath_ncfile_out, 'w')

# --- add dimensions
fo.createDimension('edge', IcD.ne_all)
ncv = fo.createVariable('elon', 'f4', ('edge',))
ncv[:] = IcD.elon
ncv = fo.createVariable('elat', 'f4', ('edge',))
ncv[:] = IcD.elat

# --- loop over all sections
for M in Ms:
  print(M.name)

  # --- derive edge_mask
  edge_mask = find_section_edges_orientation(
     IcD, M.lon1, M.lat1, M.lon2, M.lat2, dlon=M.dlon, dlat=M.dlat)

  # --- save edge_mask to nc file
  ncv = fo.createVariable(M.name,'f4',('edge',))
  ncv[:] = edge_mask

# --- close nc file
fo.close()

