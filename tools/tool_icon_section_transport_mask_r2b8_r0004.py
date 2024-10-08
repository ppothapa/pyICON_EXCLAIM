import sys
import numpy as np
from netCDF4 import Dataset
from ipdb import set_trace as mybreak
from cartopy import crs as ccrs

# --- dummy object to collect variables
class Section(object):
  def __init__(self, name, p1, p2, along_const_lat=False):
    self.name = name
    self.along_const_lat = along_const_lat
    self.lon1 = p1[0]
    self.lat1 = p1[1]
    self.lon2 = p2[0]
    self.lat2 = p2[1]
    return

class Obj(object):
  def __init__(self):
    return

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

def find_section_edges_orientation(IcD, lon1, lat1, lon2, lat2, along_const_lat=False):
  lon1o, lat1o = lon1, lat1
  lon2o, lat2o = lon2, lat2
  iv1 = np.argmin((IcD.vlon-lon1o)**2+(IcD.vlat-lat1o)**2)
  lon1, lat1 = IcD.vlon[iv1], IcD.vlat[iv1]
  iv2 = np.argmin((IcD.vlon-lon2o)**2+(IcD.vlat-lat2o)**2)
  lon2, lat2 = IcD.vlon[iv2], IcD.vlat[iv2]

  # make sure that we are slightly next to target points
  lon1 += 1e-10
  lat1 += 1e-10
  lon2 += 1e-10
  lat2 += 1e-10
  
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
    if False:
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
  #try:
  iv_s = iv_all.flatten()[np.argmin((vlon[iv_all]-lon1)**2+(vlat[iv_all]-lat1)**2)]
  iv_e = iv_all.flatten()[np.argmin((vlon[iv_all]-lon2)**2+(vlat[iv_all]-lat2)**2)]
  #except:
  #  print('::: Warning: Cannot continue finding section points. :::')
  #  return
  iv_inval = iv_inval[iv_inval!=iv_s]
  iv_inval = iv_inval[iv_inval!=iv_e]
  # --- all edges
  ie_all = np.where(mask_e)[0]
  # --- valid edges are those who are not attached to invalid vertices
  mtmp = np.isin(ie_all,IcD.edges_of_vertex[iv_inval])
  ie_valid = ie_all[mtmp==False]
  iv_valid = IcD.edge_vertices[ie_valid,:]
  
  iv_all = np.unique(iv_valid)
  iv1 = iv_all[ np.argmin((vlon[iv_all]-lon1)**2+(vlat[iv_all]-lat1)**2) ]
  iv2 = iv_all[ np.argmin((vlon[iv_all]-lon2)**2+(vlat[iv_all]-lat2)**2) ]
  
  # there are two edges belonging to the potential start vortex
  ind = np.where(iv_valid == iv1) # 1st dim: edges that match; 2nd dim: vert of edge that match
  
  if ind[0].size==1:
      iv1ed = iv_valid[ind[0][0],1-ind[1][0]]
      iv_next = iv1ed
      iel_prev = ind[0][0]
      ivl_prev = ind[1][0]
  else:
      # ind of other edge vortex (edge 1/2)
      iv1ed = iv_valid[ind[0][0],1-ind[1][0]]
      iv2ed = iv_valid[ind[0][1],1-ind[1][1]]
      # decide which edge is start edge by finding associated vortex of both edges
      # and check which is closer to final vertex
      d1 = (vlon[iv1ed]-lon2)**2+(vlat[iv1ed]-lat2)**2
      d2 = (vlon[iv2ed]-lon2)**2+(vlat[iv2ed]-lat2)**2
      if d1 < d2:
          iv_next = iv1ed
          iel_prev = ind[0][0]
          ivl_prev = ind[1][0]
      else:
          iv_next = iv2ed
          iel_prev = ind[0][1]
          ivl_prev = ind[1][1]
  
  iv_search = 1*iv_valid
  #iel_prev = 10
  #ivl_prev = 0
  #iv_prev = iv_search[iel_prev, ivl_prev]
  ie_list = [ie_valid[iel_prev]]
  iv_list = [iv_valid[iel_prev, ivl_prev]]
  iv_end = iv2
  
  icount = -1
  while True:
      icount += 1
      iv_next = iv_valid[iel_prev, 1-ivl_prev]
      iv_list.append(iv_next)
      if iv_next==iv_end:
          break
      elif icount > 10e6:
          print('did not find final vertex')
          break
      iv_search[iel_prev,:] = np.ma.masked
      #iel_prev = np.where(iv_search==iv_next)[0][0]
      #ivl_prev = np.where(iv_search==iv_next)[1][0]
      #ie_list.append(ie_valid[iel_prev])
      try:
          iel_prev = np.where(iv_search==iv_next)[0][0]
          ivl_prev = np.where(iv_search==iv_next)[1][0]
          ie_list.append(ie_valid[iel_prev])
      except:
          print('::: Warning: Cannot continue finding section points. :::')
          break
      #print(vlon[iv_list[-1]], vlat[iv_list[-1]])
      #print('iv: ', iv_valid[iel_prev, ivl_prev])
      #print('ie: ', ie_valid[iel_prev])
  
  ie_list = np.array(ie_list)
  iv_list = np.array(iv_list)

  or_list = np.zeros((ie_list.size))
  for nn in range(ie_list.size):
      iel = IcD.edges_of_vertex[iv_list[nn],:]==ie_list[nn]
      or_list[nn] = IcD.edge_orientation[iv_list[nn], iel]

  # --- output
  edge_mask = np.zeros((IcD.elon.size))
  edge_mask[ie_list] = or_list

  return ie_list, iv_list, edge_mask

# --- vector containing all section objects
Ms = []

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# START USER INPUT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
do_plot_sections_in_the_end = True

# --- path to triploar grid file
gname = 'r2b8_oce_r0004'
path_grid = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
fpath_tgrid = f'{path_grid}/{gname}_tgrid.nc'
fpath_fx = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/{gname}_L128_fx.nc'
# --- path to output netcdf file
fpath_ncfile_out = f'{path_grid}section_mask_{gname}.nc'

# --- sec: barents_opening
Ms.append(Section('barents_opening',
            p1=[16.6, 77.],
            p2=[19.5, 69.8],
         ))
# --- sec: bering_strait
Ms.append(Section('bering_strait',
            p1=[-170.2, 66.1],
            p2=[-167.7, 65.6],
         ))
# --- sec: carribean_windward_passage
Ms.append(Section('caribbean_windward_passage',
            p1=[-74.33, 20.2],
            p2=[-73.36, 19.78],
         ))
# --- sec: davis_strait
Ms.append(Section('davis_strait',
            p1=[-62.6, 66],
            p2=[-50, 65],
         ))
# --- sec: denmark_strait
Ms.append(Section('denmark_strait',
            p1=[-37, 66.1],
            p2=[-22.5, 66.],
         ))
# --- sec: drake_passage
Ms.append(Section('drake_passage',
            p1=[-67, -55],
            p2=[-60.5, -64.3],
         ))
# --- sec: english_channel
Ms.append(Section('english_channel',
            p1=[1.21, 51.2],
            p2=[1.76, 50.82],
         ))
# --- sec: faroe_scotland_channel
Ms.append(Section('faroe_scotland_channel',
            p1=[-6.9, 62,],
            p2=[-4.9, 58.61],
         ))
# --- sec: florida_bahamas_strait
Ms.append(Section('florida_bahamas_strait',
            p1=[-80.17, 26.48],
            p2=[-78.64, 26.5],
         ))
# --- sec: fram_strait
Ms.append(Section('fram_strait',
            p1=[-20, 79,],
            p2=[ 12.16, 78.94],
         ))
# --- sec: gilbraltar_strait
Ms.append(Section('gilbraltar_strait',
            p1=[-5.611, 36.03],
            p2=[-5.507, 35.83],
         ))
# --- sec: iceland_faroe_channel
Ms.append(Section('iceland_faroe_channel',
            p1=[-13.81, 65.13],
            p2=[-7.4, 62.2],
         ))
# --- sec: indonesian_throughflow
Ms.append(Section('indonesian_throughflow',
            p1=[105.7, -5.84],
            p2=[138.9, -7.125],
         ))
# --- sec: mozambique_channel
Ms.append(Section('mozambique_channel',
            p1=[39, -16],
            p2=[45, -18,],
         ))
# --- sec: pacific_equatorial_undercurrent
Ms.append(Section('pacific_equatorial_undercurrent',
            p1=[-155, 2],
            p2=[-155, -2],
         ))
# --- sec: taiwan_and_luzon_straits
Ms.append(Section('taiwan_and_luzon_straits',
            p1=[120.8, 22.4],
            p2=[120.9, 18.3],
         ))

# --- sec: 26N
Ms.append(Section('atl26N',
            p1=[-80.17, 26.48],
            p2=[-13.8, 26.],
            along_const_lat=True,
         ))

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
# END USER INPUT
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

# --- load tripolar grid
print(f'Loading grid: {fpath_tgrid}')
IcD = load_grid(fpath_tgrid)

# --- open netcdf file
print(f'Writing file {fpath_ncfile_out}')
fo = Dataset(fpath_ncfile_out, 'w')

# --- add dimensions
fo.createDimension('edge', IcD.ne_all)
ncv = fo.createVariable('elon', 'f4', ('edge',))
ncv[:] = IcD.elon
ncv = fo.createVariable('elat', 'f4', ('edge',))
ncv[:] = IcD.elat
fo.createDimension('vertex', IcD.vlon.size)
ncv = fo.createVariable('vlon', 'f4', ('vertex',))
ncv[:] = IcD.vlon
ncv = fo.createVariable('vlat', 'f4', ('vertex',))
ncv[:] = IcD.vlat

# --- loop over all sections
for M in Ms:
  print(M.name)

  # --- derive edge_mask
  ie_list, iv_list, edge_mask = find_section_edges_orientation(
     IcD, M.lon1, M.lat1, M.lon2, M.lat2, M.along_const_lat)

  # --- save edge_mask to nc file
  ncv = fo.createVariable('mask_'+M.name,'i4',('edge',))
  ncv[:] = edge_mask
  
  # --- save edge_list and vertex list to nc file
  ncv = fo.createVariable('ie_'+M.name, 'i4', ('edge'))
  ncv[:ie_list.size] = ie_list
  ncv = fo.createVariable('iv_'+M.name, 'i4', ('vertex'))
  ncv[:iv_list.size] = iv_list

# --- close nc file
fo.close()

if do_plot_sections_in_the_end:
  #sys.path.append('/Users/nbruggemann/Promotion/src/pyicon/')
  import pyicon as pyic
  import matplotlib.pyplot as plt
  import matplotlib
  fo = Dataset(fpath_ncfile_out, 'r')
  vlist = np.array(list(fo.variables.keys()))
  vlist=vlist[vlist!='elon']
  vlist=vlist[vlist!='elat']
  for var in fo.variables.keys():
    exec('%s = fo.variables[var][:]'%var)
  fo.close()

  f = Dataset(fpath_fx, 'r')
  wet_c = f.variables['wet_c'][0,:]
  f.close()

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
  ccrs_proj = ccrs.PlateCarree()
  hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                              sharex=True, sharey=True, xlabel="", ylabel="", projection=ccrs_proj)
  ii=-1
  
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  #ax.plot([lon1, lon2], [lat1, lat2])
  #pyic.shade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k', clim=1)
  #ax.triplot(IcD.Tri, color='k', zorder=1., linewidth=0.2, transform=ccrs_proj)
  ax.triplot(IcD.Tri, color='k', zorder=1., linewidth=0.2)
  if True:
    pyic.shade(IcD.Tri, wet_c, ax=ax, cax=cax)
  pyic.plot_settings(ax, template='global', land_facecolor='none', coastlines_color='orange')
  #for var in vlist:
  for M in Ms:
    var = M.name
    print(var)
    exec('mask = 1.*mask_%s'%var)
    exec('iv_list = 1*iv_%s'%var)
    ax.scatter(elon[mask!=0], elat[mask!=0], c='b', s=10, zorder=3)
    ax.scatter(vlon[iv_list], vlat[iv_list], c='m', s=10, zorder=3)
    ax.text(elon[mask!=0].mean(), elat[mask!=0].mean(), var, ha='center', va='center', color='r')
    ax.scatter(M.lon1, M.lat1, c='g', s=20, zorder=3)
    ax.scatter(M.lon2, M.lat2, c='r', s=20, zorder=3)

  #print(f'Saving figure tool_icon_section_transport_mask_{gname}.pdf')
  #plt.savefig(f'tool_icon_section_transport_mask_{gname}.pdf')
  plt.show()
