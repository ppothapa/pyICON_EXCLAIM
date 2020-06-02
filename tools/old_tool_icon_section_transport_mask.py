#!/usr/bin/env python
import sys
import numpy as np
from netCDF4 import Dataset
#from ipdb import set_trace as mybreak

# --- dummy object to collect variables
class Obj(object):
  def __init__(self):
    self.along_const_lat = False
    return

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

# --- function to load the tripolar grid
def load_grid(fpath_tgrid):
  IcD = Obj()
  IcD.fpath_tgrid = fpath_tgrid
  
  f = Dataset(IcD.fpath_tgrid, 'r')
  IcD.orientation_of_normal = f.variables['orientation_of_normal'][:].transpose()
  IcD.edge_orientation = f.variables['edge_orientation'][:].transpose()
  IcD.edge_vertices = f.variables['edge_vertices'][:].transpose()-1
  IcD.edge_of_cell = f.variables['edge_of_cell'][:].transpose()-1
  IcD.vertex_of_cell = f.variables['vertex_of_cell'][:].transpose()-1
  IcD.vertices_of_vertex = f.variables['vertices_of_vertex'][:].transpose()-1
  IcD.edges_of_vertex = f.variables['edges_of_vertex'][:].transpose()-1
  IcD.adjacent_cell_of_edge = f.variables['adjacent_cell_of_edge'][:].transpose()-1
  IcD.neighbor_cell_index = f.variables['neighbor_cell_index'][:].transpose()-1
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
def find_section_edges_orientation(IcD, lon1, lat1, lon2, lat2, along_const_lat=False):

  # --- renaming 
  elon = IcD.elon
  elat = IcD.elat
  clon = IcD.clon
  clat = IcD.clat
  vlon = IcD.vlon
  vlat = IcD.vlat

  # --- path along great circle
  if not along_const_lat:
    p1 = lon1*np.pi/180.
    t1 = lat1*np.pi/180.
    p2 = lon2*np.pi/180.
    t2 = lat2*np.pi/180.
    p3 = IcD.vlon*np.pi/180.
    t3 = IcD.vlat*np.pi/180.
    crit = ( np.cos(p3)*np.cos(t3)*( np.sin(p1)*np.cos(t1)*np.sin(t2) - np.sin(t1)*np.sin(p2)*np.cos(t2) )
           + np.sin(p3)*np.cos(t3)*( np.sin(t1)*np.cos(p2)*np.cos(t2) - np.cos(p1)*np.cos(t1)*np.sin(t2) )
           + np.sin(t3)           *( np.cos(p1)*np.cos(t1)*np.sin(p2)*np.cos(t2) - np.sin(p1)*np.cos(t1)*np.cos(p2)*np.cos(t2) ) )
    mask_e = crit[IcD.edge_vertices[:,0]]*crit[IcD.edge_vertices[:,1]]<0.
    # --- cut to longitude range
    if lon1<lon2:
      mask_e[ (elon<lon1) | (elon>lon2) ] = False
    elif lon1>lon2:
      mask_e[ (elon>lon1) | (elon<lon2) ] = False
    # --- cut to latitude range if lon1==lon2
    else:
      mask_e[ (elat<lat1) | (elat>lat2) ] = False
      mask_e[ np.abs(elon-lon1)>5. ] = False
  # --- path along const. latitude
  else:
    crit = (vlat-lat1)
    mask_e = crit[IcD.edge_vertices[:,0]]*crit[IcD.edge_vertices[:,1]]<0.
    mask_e[ (elon<lon1) | (elon>lon2) ] = False
  
  # --- all cells in domain
  ic = IcD.adjacent_cell_of_edge[mask_e,:]
  ic = np.unique(ic)
  ic = ic[ic!=-1]
  
  if ic.size==0:
    raise ValueError('::: Error: No points in domain! :::')
  
  # --- all edges that belong to outer perimeter
  ie, ind, cnts = np.unique(IcD.edge_of_cell[ic,:], return_index=True, return_counts=True)
  ie = ie[cnts==1]
  # --- all vertices that belong to perimeter edges
  iv = 1*IcD.edge_vertices[ie,:]
  # --- find start and end point
  iv_start = iv.flat[ np.argmin((vlon[iv]-lon1)**2+(vlat[iv]-lat1)**2) ]
  iv_end   = iv.flat[ np.argmin((vlon[iv]-lon2)**2+(vlat[iv]-lat2)**2) ]
  
  #for kk in range(2):
  for kk in range(1):
    iv = 1*IcD.edge_vertices[ie,:]
    iv_next = iv_start
    iv_sec = [iv_next]
    ie_sec = []
    or_sec = []
    
    itmp = np.where((iv==iv_next).sum(axis=1))[0][kk]
  
    cont = True
    start_before_end = False
    nn = 0
    while cont:
      nn+=1
      # --- save old value
      iv_previous = iv_next
      # --- index where vertex list matches iv_previous to find iv_next
      if nn==1:
        # --- first time there are two choices acoording to two ways around domain
        itmp = np.where((iv==iv_next).sum(axis=1))[0][kk]
      else:
        # --- after that there should be only one choice
        itmp = np.where((iv==iv_next).sum(axis=1))[0][0]
      # --- next edge
      ie_next = ie[itmp]
      # --- next vertex
      iv_next = iv[itmp,iv[itmp,:]!=iv_previous][0]
      # --- orientation
      itmp2 = np.where(IcD.vertices_of_vertex[iv_previous,:]==iv_next)[0][0]
      or_next = IcD.edge_orientation[iv_previous, itmp2]
      # --- delete iv entry such that it is not found again
      iv[itmp,:] = np.ma.masked
      # --- save indices
      iv_sec.append(iv_next)
      ie_sec.append(ie_next)
      or_sec.append(or_next)
  
      # --- look for hanging cells
      ie_tmp = ie[np.isin(ie,IcD.edges_of_vertex[iv_next,:])]
      if ie_tmp.size>2:
        print('::: Warning, hanging cell.:::')
        print(ie_tmp)
        cont = False
    
      # --- check whether... 
      # --- ... we are at the final destination
      if iv_next==iv_end:
        cont = False
      # --- ... back to the beginning
      elif iv_next==iv_start:
        print('::: Warning: Reached start vertex again before reaching end vertex! :::')
        cont = False
        start_before_end = True
      # --- ... have not found anything after X iterations
      elif nn>20000:
        print('::: Warning: Max. number of iterations exceeded. :::')
        cont = False
    else:
      if start_before_end:
        dist_sec = haversine_dist(lon2, lat2, vlon[iv_sec], vlat[iv_sec], degree=True)
        ddist = np.diff(dist_sec)
        count = 0.
        for nn in range(ddist.size):
          if ddist[nn]>0:  # we are getting farer away from point 2
            count += 1.
          else:        # we are getting closer to point 2
            count = 0.
          if count>5.:
            print('Cutting section at point closest to end point.')
            nn += -5
            break
        iv_sec = iv_sec[:nn+1]
        ie_sec = ie_sec[:nn]
        or_sec = or_sec[:nn]
      # --- save the two index lists
      if kk==0:
        iv_sec1 = np.array(iv_sec)
        ie_sec1 = np.array(ie_sec)
        or_sec1 = np.array(or_sec)
      else:
        iv_sec2 = np.array(iv_sec)
        ie_sec2 = np.array(ie_sec)
        or_sec2 = np.array(or_sec)


  # --- randomly change sec1 for output
  edge_mask = np.zeros((IcD.elon.size))
  edge_mask[ie_sec1] = or_sec1

  return edge_mask

def find_section_edges_orientation_2(IcD, lon1, lat1, lon2, lat2, along_const_lat=False):

  # --- renaming 
  elon = IcD.elon
  elat = IcD.elat
  clon = IcD.clon
  clat = IcD.clat
  vlon = IcD.vlon
  vlat = IcD.vlat

  # --- path along great circle
  if not along_const_lat:
    p1 = lon1*np.pi/180.
    t1 = lat1*np.pi/180.
    p2 = lon2*np.pi/180.
    t2 = lat2*np.pi/180.
    p3 = IcD.vlon*np.pi/180.
    t3 = IcD.vlat*np.pi/180.
    crit = ( np.cos(p3)*np.cos(t3)*( np.sin(p1)*np.cos(t1)*np.sin(t2) - np.sin(t1)*np.sin(p2)*np.cos(t2) )
           + np.sin(p3)*np.cos(t3)*( np.sin(t1)*np.cos(p2)*np.cos(t2) - np.cos(p1)*np.cos(t1)*np.sin(t2) )
           + np.sin(t3)           *( np.cos(p1)*np.cos(t1)*np.sin(p2)*np.cos(t2) - np.sin(p1)*np.cos(t1)*np.cos(p2)*np.cos(t2) ) )
    mask_e = crit[IcD.edge_vertices[:,0]]*crit[IcD.edge_vertices[:,1]]<0.
    # --- cut to longitude range
    if lon1<lon2:
      mask_e[ (elon<lon1) | (elon>lon2) ] = False
    elif lon1>lon2:
      mask_e[ (elon>lon1) | (elon<lon2) ] = False
    # --- cut to latitude range if lon1==lon2
    else:
      mask_e[ (elat<lat1) | (elat>lat2) ] = False
      mask_e[ np.abs(elon-lon1)>5. ] = False
  # --- path along const. latitude
  else:
    crit = (vlat-lat1)
    mask_e = crit[IcD.edge_vertices[:,0]]*crit[IcD.edge_vertices[:,1]]<0.
    mask_e[ (elon<lon1) | (elon>lon2) ] = False

  # --- all vertices belonging to edges
  iv_all = IcD.edge_vertices[mask_e,:]
  # --- delete vertices that only occur once - but not start and end vertex
  itmp, ind, cnts = np.unique(iv_all, return_index=True, return_counts=True)
  iv_inval = itmp[cnts==1]
  iv_s = iv_all.flatten()[np.argmin((vlon[iv_all]-lon1)**2+(vlat[iv_all]-lat1)**2)]
  iv_e = iv_all.flatten()[np.argmin((vlon[iv_all]-lon2)**2+(vlat[iv_all]-lat2)**2)]
  iv_inval = iv_inval[iv_inval!=iv_s]
  iv_inval = iv_inval[iv_inval!=iv_e]
  # --- all edges
  ie_all = np.where(mask_e)[0]
  # --- valid edges are those who are not attached to invalid vertices 
  mtmp = np.isin(ie_all,IcD.edges_of_vertex[iv_inval])
  ie_valid = ie_all[mtmp==False]
  iv_valid = IcD.edge_vertices[ie_valid,:]

  # --- find start and end vertices
  i, c = np.unique(iv_valid, return_counts=True) 
  iv_se = i[c==1]
  iv_start = iv_se[np.argmin((vlon[iv_se]-lon1)**2+(vlat[iv_se]-lat1)**2)]
  iv_next = iv_start
  # --- allocate final lists
  iv_sec = [iv_start]
  ie_sec = []
  or_sec = []

  for nn in range(ie_valid.size):
    # --- save old value
    iv_previous = iv_next
    # --- index where vertex list matches iv_next
    itmp = np.where((iv_valid==iv_next).sum(axis=1))[0][0]
    # --- next edge
    ie_next = ie_valid[itmp]
    # --- next vertex
    iv_next = iv_valid[itmp,iv_valid[itmp,:]!=iv_previous][0]
    # --- orientation
    #itmp2 = np.where(IcD.vertices_of_vertex[iv_previous,:]==iv_next)[0][0]
    ##itmp2 = np.where((iv_valid==[iv_previous,iv_next]).sum(axis=1))[0]
    ##if itmp2.size==1:
    ##  itmp2 = itmp2[0]
    ##else:
    ##  itmp2 = np.where((iv_valid==[iv_next,iv_previous]).sum(axis=1))[0][0]
    itmp2 = np.where(IcD.edges_of_vertex[iv_previous]==ie_next)[0][0]
    or_next = IcD.edge_orientation[iv_previous, itmp2]
    # --- delete iv entry such that it is not found again
    iv_valid[itmp,:] = np.ma.masked
    # --- save indices
    iv_sec.append(iv_next)
    ie_sec.append(ie_next)
    or_sec.append(or_next)

  #print(np.array(ie_sec).size)
  #print(ie_sec)

  # --- output
  edge_mask = np.zeros((IcD.elon.size))
  edge_mask[ie_sec] = or_sec

  return edge_mask

# --- vector containing all section objects
Ms = []

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# START USER INPUT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# --- path to triploar grid file
#fpath_tgrid = '/pool/data/ICON/oes/input/r0003/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min.nc'
#fpath_tgrid = '/pool/data/ICON/oes/input/r0002/OceanOnly_Icos_0158km_etopo40/OceanOnly_Icos_0158km_etopo40.nc'
fpath_tgrid = '/Users/nbruggemann/work/icon_playground/r2b4/r2b4_tgrid.nc'
# --- path to output netcdf file
fpath_ncfile_out = 'icon_section_transport_mask.nc'

# --- sec 1 barents_opening
M = Obj()
#M.lon1, M.lat1 = 16.8, 76.5
#M.lon2, M.lat2 = 19.2, 70.2
M.lon1, M.lat1 = 16.6, 77.
M.lon2, M.lat2 = 19.5, 69.8
M.name = 'barents_opening'
Ms.append(M)

# --- sec 2 bering_strait
M = Obj()
M.lon1, M.lat1 = -171, 66.2
M.lon2, M.lat2 = -166, 65
M.name = 'bering_strait'
Ms.append(M)

# --- sec 6 drake_passage
M = Obj()
#M.lon1, M.lat1 = -68, -54.
#M.lon1, M.lat1 = -60, -64.7
#M.lon2, M.lat2 = -68, -54.

M.lon1, M.lat1 = -60.5, -64.3
M.lon2, M.lat2 = -67, -55.
M.name = 'drake_passage'
Ms.append(M)

# --- sec 1
M = Obj()
#M.lon1, M.lat1 = -98., 26.
M.lon1, M.lat1 = -80.5, 26.
M.lon2, M.lat2 = -14., 26.
M.along_const_lat = True
#M.lon1, M.lat1 = -60., 54.
#M.lon2, M.lat2 = -43., 70.
#M.name = '26N'
#M.lon1, M.lat1 = -50., 0.
#M.lon2, M.lat2 =  12., 0.
#M.name = 'Atl.Eq.'
#M.lon1, M.lat1 = -30., -78.
#M.lon2, M.lat2 = -30.,  70.
M.name = 'atl26N'
#M.along_const_lat = True
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
  edge_mask = find_section_edges_orientation_2(
     IcD, M.lon1, M.lat1, M.lon2, M.lat2, M.along_const_lat)

  # --- save edge_mask to nc file
  ncv = fo.createVariable(M.name,'f4',('edge',))
  ncv[:] = edge_mask

# --- close nc file
fo.close()

if True:
  sys.path.append('/Users/nbruggemann/Promotion/src/pyicon/')
  import pyicon as pyic
  import matplotlib.pyplot as plt
  import matplotlib
  import my_toolbox as my
  fo = Dataset(fpath_ncfile_out, 'r')
  vlist = np.array(list(fo.variables.keys()))
  vlist=vlist[vlist!='elon']
  vlist=vlist[vlist!='elat']
  for var in fo.variables.keys():
    exec('%s = fo.variables[var][:]'%var)
  fo.close()

  elon = IcD.elon
  elat = IcD.elat
  clon = IcD.clon
  clat = IcD.clat
  vlon = IcD.vlon
  vlat = IcD.vlat

  # --- for plotting the grid
  IcD.Tri = matplotlib.tri.Triangulation(IcD.vlon, IcD.vlat, 
                                          triangles=IcD.vertex_of_cell)
  mask_bt = (
    (IcD.vlon[IcD.vertex_of_cell[:,0]] - IcD.vlon[IcD.vertex_of_cell[:,1]])**2
  + (IcD.vlon[IcD.vertex_of_cell[:,0]] - IcD.vlon[IcD.vertex_of_cell[:,2]])**2
  + (IcD.vlat[IcD.vertex_of_cell[:,0]] - IcD.vlat[IcD.vertex_of_cell[:,1]])**2
  + (IcD.vlat[IcD.vertex_of_cell[:,0]] - IcD.vlat[IcD.vertex_of_cell[:,2]])**2
             ) > 2.*180./np.pi
  IcD.Tri.set_mask(mask_bt)
  
  empt_data = np.zeros(IcD.clon.shape)

  plt.close("all")
  hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, sasp=0.5, fig_size_fac=2.,
                              sharex=True, sharey=True, xlabel="", ylabel="")
  ii=-1
  
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  #ax.plot([lon1, lon2], [lat1, lat2])
  pyic.trishade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k', clim=1)
  #for var in vlist:
  for M in Ms:
    var = M.name
    print(var)
    exec('mask = 1.*%s'%var)
    ax.scatter(elon[mask!=0], elat[mask!=0], c='b', s=10)
    ax.scatter(M.lon1, M.lat1, c='g', s=20)
    ax.scatter(M.lon2, M.lat2, c='r', s=20)

  plt.show()

