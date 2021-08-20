print('sys glob os')
import sys
import glob, os
#import datetime
print('numpy')
import numpy as np
print('netcdf')
from netCDF4 import Dataset, num2date
#from scipy import interpolate
#from scipy.spatial import cKDTree
#print('ipdb')
#from ipdb import set_trace as mybreak  
#from .pyicon_tb import *
print('Done modules calc.')

#def distance(p1, p2):
#  """
#  p1: cartesian vector with 2nd dim (x,y,z) dim(npoints, 3)
#  p2: same as p1
#  """
#  # distance vector
#  dv = (p1-p2)
#  distance = np.sqrt(dv[:,0]**2+dv[:,1]**2+dv[:,2]**2)
#  return dist

def vector_product(v1, v2):
  v3 = np.zeros_like(v1)
  v3[...,0] = v1[...,1]*v2[...,2]-v1[...,2]*v2[...,1]
  v3[...,1] = v1[...,2]*v2[...,0]-v1[...,0]*v2[...,2]
  v3[...,2] = v1[...,0]*v2[...,1]-v1[...,1]*v2[...,0]
  return v3

def scalar_product(v1, v2, dim=1, numpy_sum=False):
  if numpy_sum:
    scalar_product = (v1*v2).sum(axis=dim)
  else:
    # this seems to be faster
    if dim==1:
      scalar_product = v1[:,0]*v2[:,0]+v1[:,1]*v2[:,1]+v1[:,2]*v2[:,2]
    elif dim==2:
      scalar_product = v1[:,:,0]*v2[:,:,0]+v1[:,:,1]*v2[:,:,1]+v1[:,:,2]*v2[:,:,2]
    elif dim==3:
      scalar_product = v1[:,:,:,0]*v2[:,:,:,0]+v1[:,:,:,1]*v2[:,:,:,1]+v1[:,:,:,2]*v2[:,:,:,2]
  return scalar_product

def planar_triangle_area(p1, p2, p3):
  """
  p1: cartesian vector with 2nd dim (x,y,z) dim(npoints, 3)
  p2: same as p1
  p3: same as p1
  """

  #iv = 3738
  dv1 = (p2-p1)
  dv2 = (p3-p1)
  p_vector = vector_product(dv1, dv2)
  planar_triangle_area = 0.5 * np.sqrt(scalar_product(p_vector, p_vector))
  #print('dv1 = ', dv1[iv,:])
  #print('dv2 = ', dv2[iv,:])
  #print('p_vector = ', p_vector[iv,:])
  return planar_triangle_area

def edges2cell(IcD, ve):
  p_vn_c = (   IcD.edge2cell_coeff_cc[np.newaxis,:,:,:]
             * ve[:,IcD.edge_of_cell,np.newaxis]
             * IcD.prism_thick_e[:,IcD.edge_of_cell,np.newaxis]
           ).sum(axis=2)
  # dim(fixed_vol_norm) = (nCells)
  #dist_vector = IcD.edge_cart_vec[IcD.edge_of_cell,:] - IcD.cell_cart_vec[:,np.newaxis,:]
  #fixed_vol_norm = (0.5 * np.sqrt(scalar_product(dist_vector,dist_vector,dim=2)) * IcD.edge_length[IcD.edge_of_cell]).sum(axis=1)
  #p_vn_c *= 1./(fixed_vol_norm[np.newaxis,:,np.newaxis]*IcD.prism_thick_c[:,:,np.newaxis])
  p_vn_c *= 1./(IcD.fixed_vol_norm[np.newaxis,:,np.newaxis]*IcD.prism_thick_c[:,:,np.newaxis])
  return p_vn_c

def calc_fixed_volume_norm(IcD):
  # --- fixed volume norm
  dist_vector = IcD.edge_cart_vec[IcD.edge_of_cell,:] - IcD.cell_cart_vec[:,np.newaxis,:]
  norm = np.sqrt(scalar_product(dist_vector,dist_vector,dim=2)) 
  prime_edge_length = IcD.edge_length/IcD.grid_sphere_radius
  fixed_vol_norm = (  0.5 * norm
                    * (prime_edge_length[IcD.edge_of_cell]))
  fixed_vol_norm = fixed_vol_norm.sum(axis=1)
  
  # ------ check fixed volume
  # ic30w26n = np.argmin((IcD.clon+30.)**2+(IcD.clat-26.)**2)
  # ic = ic30w26n
  # ni = 16
  # block = (ic+1)//ni + 1 
  # index = (ic+1)-(block-1)*ni 
  # block, index, ic
  # print('clon = ', IcD.clon[ic]*np.pi/180., 'clat = ', IcD.clat[ic]*np.pi/180.)
  # print('fixed_vol_norm = ', fixed_vol_norm[ic])
  # print('dist_vector = ', dist_vector[ic,0,:])
  # print('norm = ', norm[ic,:])
  # print(np.sqrt((dist_vector[ic,0,:]**2).sum()))
  # print('prime_edge_length = ', prime_edge_length[IcD.edge_of_cell][ic,:])
  # #print(np.sqrt(scalar_product(dist_vector[ic,0,:],dist_vector[ic,0,:])) )
  return fixed_vol_norm

def calc_edge2edge_viacell_coeff(IcD):
  cell_index = IcD.adjacent_cell_of_edge
  
  # dist_vector_basic: distance vector between edge center and center of the two neighbouring cells
  # dim(dist_vector_basic) = (nEdges, nCellOfEdges, nCartDims)
  dist_vector_basic = IcD.edge_cart_vec[:,np.newaxis,:] - IcD.cell_cart_vec[cell_index,:]
  dist_edge_cell_basic  = np.sqrt(scalar_product(dist_vector_basic,dist_vector_basic, dim=2))
  dist_vector_basic *= 1./dist_edge_cell_basic[:,:,np.newaxis]
  orientation = scalar_product(dist_vector_basic, IcD.edge_prim_norm[:,np.newaxis,:], dim=2)
  dist_vector_basic *= np.sign(orientation)[:,:,np.newaxis]
  
  cell_center = IcD.cell_cart_vec[cell_index,:]
  edge_center = IcD.edge_cart_vec
  
  # dist_vector: distance vector between adjacent cell centers of edge and their sourounding edges
  # dim(edge_index_cell) = (nEdges, neighbCells, neighbEdges)
  edge_index_cell = IcD.edge_of_cell[cell_index,:]
  # dim(dist_vector) = (nEdges, nCellOfEdges, nEdgesOfCell, nCartDims)
  dist_vector = IcD.edge_cart_vec[edge_index_cell,:] - cell_center[:,:,np.newaxis,:]
  dist_edge_cell = np.sqrt(scalar_product(dist_vector, dist_vector, dim=3))
  dist_vector *= 1./ dist_edge_cell[:,:,:,np.newaxis]
  dist_vector *= IcD.orientation_of_normal[cell_index][:,:,:,np.newaxis]
  
  edge2edge_viacell_coeff_2D = scalar_product(dist_vector_basic[:,:,np.newaxis,:],dist_vector, dim=3)
  
  # math/mo_operator_ocean_coeff_3d.f90: init_operator_coeffs_cell
  edge2edge_viacell_coeff_2D *= (  (IcD.edge_length[edge_index_cell]/IcD.grid_sphere_radius)
                                 * dist_edge_cell*dist_edge_cell_basic[:,:,np.newaxis] 
                                 / (IcD.dual_edge_length[:,np.newaxis,np.newaxis]/IcD.grid_sphere_radius)
                                )
  
  # ni = 16
  # ie30w26n = np.argmin((IcD.elon+30.)**2+(IcD.elat-26.)**2)
  # ip = ie30w26n
  # block = (ip+1)//ni + 1 
  # index = (ip+1)-(block-1)*ni 
  # block, index
  
  # neigbor = 0
  # ictr = 0
  # print('elon = ', IcD.elon[ip]*np.pi/180., 'elat = ', IcD.elat[ip]*np.pi/180.)
  # print('dist_edge_cell = ', dist_edge_cell[ip, neigbor, ictr])
  # print('dist_vector = ', dist_vector[ip, neigbor, ictr])
  # print('dist_edge_cell_basic = ', dist_edge_cell_basic[ip,neigbor])
  # print('dist_vector_basic = ', dist_vector_basic[ip,neigbor])
  # # print('edge2edge_viacell_coeff_2D = ', edge2edge_viacell_coeff_2D[ip, neigbor, ictr])
  # print('edge2edge_viacell_coeff_2D = ', edge2edge_viacell_coeff_2D[ip, :, :])
  
  # --- make edge2edge_viacell_coeff 3D and mask it
  edge2edge_viacell_coeff = np.tile(edge2edge_viacell_coeff_2D, (IcD.nz,1,1,1))
  
  # ------ mask if center edge is not sea
  lsm_e_ext = np.tile(IcD.lsm_e[:,:,np.newaxis,np.newaxis], (1,1,2,3))
  edge2edge_viacell_coeff[lsm_e_ext!=-2] = 0.0
  # ------ mask each edge of stencil which is not sea 
  edge2edge_viacell_coeff[IcD.lsm_e[:,IcD.edge_of_cell[IcD.adjacent_cell_of_edge,:]]!=-2] = 0.0
  
  
  # --- normalize by fixed_vol_norm
  edge2edge_viacell_coeff[:,:,0,:] *= 1./IcD.fixed_vol_norm[IcD.adjacent_cell_of_edge[:,0]][np.newaxis,:,np.newaxis]
  edge2edge_viacell_coeff[:,:,1,:] *= 1./IcD.fixed_vol_norm[IcD.adjacent_cell_of_edge[:,1]][np.newaxis,:,np.newaxis]
  
  # print('edge2edge_viacell_coeff = ', edge2edge_viacell_coeff[0,ip, :, :])
  return edge2edge_viacell_coeff

def calc_edge2cell_coeff_cc(IcD):
  """
From math/mo_scalar_product.f90 map_edges2cell_3d (interface to map_edges2cell_no_height_3d -> map_edges2cell_no_height_3d_onTriangles):
and from math/mo_operator_ocean_coeff_3d.f90 init_operator_coeffs_cell:
        edge_1_index = patch_2d%cells%edge_idx(cell_index,blockNo,1)
        edge_1_block = patch_2d%cells%edge_blk(cell_index,blockNo,1)
        edge_2_index = patch_2d%cells%edge_idx(cell_index,blockNo,2)
        edge_2_block = patch_2d%cells%edge_blk(cell_index,blockNo,2)
        edge_3_index = patch_2d%cells%edge_idx(cell_index,blockNo,3)
        edge_3_block = patch_2d%cells%edge_blk(cell_index,blockNo,3)

        DO level = startLevel, MIN(patch_3D%p_patch_1D(1)%dolic_c(cell_index,blockNo), endLevel)
          p_vn_c(cell_index,level,blockNo)%x =                                            &
            & (  operators_coefficients%edge2cell_coeff_cc(cell_index,level,blockNo,1)%x  &
            &      * vn_e(edge_1_index,level,edge_1_block)                                &
            &*patch_3d%p_patch_1d(1)%prism_thick_e(edge_1_index,level,edge_1_block)       &
            &  + operators_coefficients%edge2cell_coeff_cc(cell_index,level,blockNo,2)%x  &
            &      * vn_e(edge_2_index,level,edge_2_block)                                &
            &*patch_3d%p_patch_1d(1)%prism_thick_e(edge_2_index,level,edge_2_block)       &
            &  + operators_coefficients%edge2cell_coeff_cc(cell_index,level,blockNo,3)%x  &
            &       * vn_e(edge_3_index,level,edge_3_block)                               &
            &*patch_3d%p_patch_1d(1)%prism_thick_e(edge_3_index,level,edge_3_block))      &
            & / (operators_coefficients%fixed_vol_norm(cell_index,level,blockNo)          &
            &    * patch_3d%p_patch_1d(1)%prism_thick_c(cell_index,level,blockNo))
        END DO

          edge_index = patch_2D%cells%edge_idx(cell_index, cell_block, neigbor)
          edge_block = patch_2D%cells%edge_blk(cell_index, cell_block, neigbor)

          IF (edge_block > 0 ) THEN
            ! we have an edge
            dist_vector = distance_vector( &
              & patch_2D%edges%cartesian_center(edge_index,edge_block), &
              & cell_center, &
              & patch_2D%geometry_info)

            ! compute edge2cell_coeff_cc
            edge2cell_coeff_cc(cell_index,cell_block,neigbor)%x =  &
              & dist_vector%x *                                             &
              & prime_edge_length(edge_index,edge_block) *                  &
              & patch_2D%cells%edge_orientation(cell_index,cell_block,neigbor)
  """
  # dim(dist_vector) = (nCells, nEdgesOfCell, nCartDims)
  dist_vector = (  IcD.edge_cart_vec[IcD.edge_of_cell,:] 
                 - IcD.cell_cart_vec[:,np.newaxis,:] )
  # dim(edge2cell_coeff_cc) = (nCells, nEdgesOfCell, nCartDims)
  edge2cell_coeff_cc = (  dist_vector 
                        * (IcD.edge_length[IcD.edge_of_cell,np.newaxis] / IcD.grid_sphere_radius)
                        * IcD.orientation_of_normal[:,:,np.newaxis] )
  # dim(edge2cell_coeff_cc[np.newaxis,:,:,:]) = (nDepth, nCells, nEdgesOfCell, nCartDims)
  return edge2cell_coeff_cc

def calc_edge2cell_coeff_cc_t(IcD):
  # dim(dist_vector) = (nEdges, nCellsOfEdges, nCartDims)
  dist_vector = (  IcD.edge_cart_vec[:,np.newaxis,:] 
                 - IcD.cell_cart_vec[IcD.adjacent_cell_of_edge,:] )
  orientation = scalar_product(dist_vector, IcD.edge_prim_norm[:,np.newaxis,:], dim=2) 
  dist_vector *= np.sign(orientation)[:,:,np.newaxis]
  # dim(edge2cell_coeff_cc_t) = (nEdges, nCellsOfEdges, nCartDims)
  edge2cell_coeff_cc_t = (  IcD.edge_prim_norm[:,np.newaxis,:]*IcD.grid_sphere_radius
                          * np.sqrt(scalar_product(dist_vector,dist_vector, dim=2))[:,:,np.newaxis] 
                          / IcD.dual_edge_length[:,np.newaxis,np.newaxis] )
  return edge2cell_coeff_cc_t
  
def edges2edges_via_cell(IcD, vn_e, dze='const'):
  """ Transfer from edge to 3D cell and back to edge. Supposed to mimic the M operator (K2017).

  Used e.g. to calculate the mass flux and therewith vert. velocity.
  """
  if isinstance(dze,str) and dze=='const':
    dze = IcD.prism_thick_e
  il_c = IcD.adjacent_cell_of_edge[:,0]
  il_e = IcD.edge_of_cell[il_c]
  out_vn_e  = (vn_e[:,il_e] * IcD.edge2edge_viacell_coeff[:,:,0,:] * dze[:,il_e]).sum(axis=2)
  il_c = IcD.adjacent_cell_of_edge[:,1]
  il_e = IcD.edge_of_cell[il_c]
  out_vn_e += (vn_e[:,il_e] * IcD.edge2edge_viacell_coeff[:,:,1,:] * dze[:,il_e]).sum(axis=2)
  return out_vn_e

def edges2edges_via_cell_scalar(IcD, vn_e, scalar, dze='const'):
  """ Same as edges2edges_via_cell but with scalar mutiplied at cell centers. Supposed to mimic M[v,phi] (K2017).

  Used e.g. to calculate advective tracer fluxes (before flux limiter are applied).
  """
  if isinstance(dze,str) and dze=='const':
    dze = IcD.prism_thick_e
  il_c = IcD.adjacent_cell_of_edge[:,0]
  il_e = IcD.edge_of_cell[il_c]
  out_vn_e  = (   (vn_e[:,il_e] * IcD.edge2edge_viacell_coeff[:,:,0,:] * dze[:,il_e]).sum(axis=2) 
                * scalar[:,il_c])
  il_c = IcD.adjacent_cell_of_edge[:,1]
  il_e = IcD.edge_of_cell[il_c]
  out_vn_e += (   (vn_e[:,il_e] * IcD.edge2edge_viacell_coeff[:,:,1,:] * dze[:,il_e]).sum(axis=2)
                * scalar[:,il_c])
  return out_vn_e

def calc_wvel(IcD, mass_flux):
  div_mass_flux = (
    mass_flux[:,IcD.edge_of_cell]*IcD.div_coeff[np.newaxis,:,:]).sum(axis=2)
  wvel = np.zeros((IcD.nz+1, IcD.clon.size), dtype=IcD.dtype)
  wvel[:IcD.nz,:] = -div_mass_flux[::-1,:].cumsum(axis=0)[::-1,:]
  return wvel

def calc_curl(IcD, ve):
  # FIXME: this needs to be tested
  curl_v = (ve[:,IcD.edges_of_vertex] * IcD.rot_coeff).sum(axis=2)
  return curl_v

def cell2edges(IcD, p_vn_c):
  """
  math/mo_scalar_product.f90: map_cell2edges_3d_mlevels
  """
  if p_vn_c.ndim==3:
    ptp_vn = (   scalar_product(p_vn_c[:,IcD.adjacent_cell_of_edge[:,0],:], 
                                IcD.edge2cell_coeff_cc_t[np.newaxis,:,0,:], dim=2)
               + scalar_product(p_vn_c[:,IcD.adjacent_cell_of_edge[:,1],:], 
                                IcD.edge2cell_coeff_cc_t[np.newaxis,:,1,:], dim=2)
             )
  elif p_vn_c.ndim==2:
    ptp_vn = (   scalar_product(p_vn_c[IcD.adjacent_cell_of_edge[:,0],:], 
                                IcD.edge2cell_coeff_cc_t[:,0,:], dim=1)
               + scalar_product(p_vn_c[IcD.adjacent_cell_of_edge[:,1],:], 
                                IcD.edge2cell_coeff_cc_t[:,1,:], dim=1)
             )
  else:
    raise ValueError(f"::: Error: Unsupport p_vn_c.ndim={p_vn_c.ndim}! :::")
  return ptp_vn

def calc_2dlocal_from_3d(IcD, p_vn_c):
  """
  ! these should be calclulated once and stored in the coefficients structure
  sinLon = SIN(position_local(this_index,blockNo)%lon)
  cosLon = COS(position_local(this_index,blockNo)%lon)
  sinLat = SIN(position_local(this_index,blockNo)%lat)
  cosLat = COS(position_local(this_index,blockNo)%lat)

  DO level = 1, levels(this_index,blockNo)
    cartesian_x = vector(this_index,level,blockNo)%x(1)
    cartesian_y = vector(this_index,level,blockNo)%x(2)
    cartesian_z = vector(this_index,level,blockNo)%x(3)

    x(this_index,level,blockNo) = cosLon * cartesian_y - sinLon * cartesian_x
    y_help                      = cosLon * cartesian_x + sinLon * cartesian_y
    y_help = sinLat * y_help
    y(this_index,level,blockNo) = cosLat * cartesian_z - y_help
  """
  sinLon = np.sin(IcD.clon*np.pi/180.)
  cosLon = np.cos(IcD.clon*np.pi/180.)
  sinLat = np.sin(IcD.clat*np.pi/180.)
  cosLat = np.cos(IcD.clat*np.pi/180.)

  u1 = p_vn_c[:,:,0]
  u2 = p_vn_c[:,:,1]
  u3 = p_vn_c[:,:,2]

  uo =   u2*cosLon - u1*sinLon
  vo = -(u1*cosLon + u2*sinLon)*sinLat + u3*cosLat

  #sinLon = np.sin(IcD.clon*np.pi/180.)
  #cosLon = np.cos(IcD.clon*np.pi/180.)
  #sinLat = np.sin(IcD.clat*np.pi/180.)
  #cosLat = np.cos(IcD.clat*np.pi/180.)

  #cartesian_x = p_vn_c[:,:,0]
  #cartesian_y = p_vn_c[:,:,1]
  #cartesian_z = p_vn_c[:,:,2]

  #x      = cosLon*cartesian_y - sinLon*cartesian_x
  #y_help = cosLon*cartesian_x + sinLon*cartesian_y
  #y_help = sinLat * y_help
  #y      = cosLat * cartesian_z - y_help
  #uo, vo = x, y
  return uo, vo

def calc_3d_from_2dlocal(IcD, uo, vo):
  sinLon = np.sin(IcD.clon*np.pi/180.)
  cosLon = np.cos(IcD.clon*np.pi/180.)
  sinLat = np.sin(IcD.clat*np.pi/180.)
  cosLat = np.cos(IcD.clat*np.pi/180.)

  u1 = -uo*sinLon - vo*sinLat*cosLon
  u2 =  uo*cosLon - vo*sinLat*sinLon
  u3 =  vo*cosLat

  conc_dim = u1.ndim
  p_vn_c = np.ma.concatenate((u1[...,np.newaxis],u2[...,np.newaxis],u3[...,np.newaxis]), axis=conc_dim)
  return p_vn_c

# //////////////////////////////////////////////////////////////////////////////// 
# \\\\\ Calculation for ICON

def calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90., verbose=False, old_version=True):
  """ Calculates barotropic streamfunction in Sv from mass_flux_vint on vertex-grid.

  This function determines neighbouring vertices starting from lon_start, lat_start 
  vertex. It determines source and target vertices and the corresponding edges. 
  Then the bstr value for each target vertex is that of the source vertex plus the 
  transport through the edge between source and target.

  Algorithm is taken from mo_postprocess.f90 (Leonidas Linardakis, MPI-M).
  """

  # --- allocations
  edge_integration_list = np.zeros((IcD.elon.size), dtype=int)
  orientation_path = np.zeros((IcD.elon.size), dtype=int)
  source_vertex_list = np.zeros((IcD.vlon.size), dtype=int)
  target_vertex_list = np.zeros((IcD.vlon.size), dtype=int)
  vertexIsAccounted_list = np.zeros((IcD.vlon.size), dtype=IcD.dtype)
  next_vertex_list = []
  
  # --- start vertex
  list_vertex_index = ((IcD.vlon-lon_start)**2+(IcD.vlat-lat_start)**2).argmin()
  vertexIsAccounted_list[list_vertex_index] = 1.
  next_vertex_list.append(list_vertex_index)
  
  if verbose:
    print('start finding indices')
  aa = 0
  totalListedEdges = 0 # index for all listed edges
  while next_vertex_list:
    aa += 1
    #if aa%100==0:
    #  print(f'aa = {aa}/')
    
    # --- take last index from least and delete it from list
    list_vertex_index = next_vertex_list.pop(-1) 
    for nn in range(6): # all neighbors
      check_vertex = IcD.vertices_of_vertex[list_vertex_index, nn] 
  
      # --- find edge that is in between list_vertex_index and check_vertex
      edge_index = IcD.edges_of_vertex[list_vertex_index, nn]
  
      if (edge_index>-1):
        # --- check if check_vertex is not in vertexIsAccounted_list
        orientation = IcD.edge_orientation[list_vertex_index,nn]
        if (vertexIsAccounted_list[check_vertex]==0.):
          totalListedEdges += 1
          # --- save everything
          edge_integration_list[totalListedEdges] = edge_index
          orientation_path[totalListedEdges]      = orientation
          source_vertex_list[totalListedEdges]    = list_vertex_index
          target_vertex_list[totalListedEdges]    = check_vertex
  
          # --- add check_vertex to next_vertex_list and mark it as accounted
          next_vertex_list.append(check_vertex)
          vertexIsAccounted_list[check_vertex] = 1
  
  # --- calculate streamfunction
  if verbose:
    print('start finding indices')
  if old_version:
    stream_variable = np.zeros((IcD.vlon.size), dtype=IcD.dtype)
    for target_list_index in range(target_vertex_list.size):
      #if target_list_index%100==0:
      #  print(f'target_list_index = {target_list_index}')
      source_vertex = source_vertex_list[target_list_index]
      target_vertex = target_vertex_list[target_list_index]
      edge_index = edge_integration_list[target_list_index]
      orientation = orientation_path[target_list_index]
    
      # --- add transport between source and target vertex to stream function of
      #     source vertex
      stream_variable[target_vertex] = stream_variable[source_vertex] \
        + orientation * IcD.edge_length[edge_index] * mass_flux_vint[edge_index]
    bstr = stream_variable * 1e-6
  else:
    # This version is much quicker but needs to be better tested
    source_vertex = source_vertex_list[target_vertex_list]
    target_vertex = target_vertex_list[target_vertex_list]
    edge_index = edge_integration_list[target_vertex_list]
    orientation = orientation_path[target_vertex_list]

    mflux_oriented = orientation * IcD.edge_length[edge_index] * mass_flux_vint[edge_index]
    mflux_sorted = mflux_oriented[source_vertex]
    a = mflux_sorted.cumsum() * 1e-6
    bstr = np.zeros((IcD.vlon.size), dtype=IcD.dtype)
    bstr[source_vertex] = a
    #bstr = mflux_sorted.cumsum() * 1e-6

  #bstr = IconVariable('bstr', units='Sv', long_name='barotropic streamfunction',
  #                   coordinates='vlat vlon', is3d=False)
  #bstr.data = stream_variable * 1e-6

  return bstr

def calc_bstr_rgrid(IcD, mass_flux_vint, lon_rg, lat_rg, dtype='float64'):
  """ Calculates barotropic streamfunction in Sv from mass_flux_vint on regular grid.

  """
  nx = lon_rg.size
  ny = lat_rg.size
  Lon_rg, Lat_rg = np.meshgrid(lon_rg, lat_rg)

  mass_flux_vint = mass_flux_vint.astype(dtype)

  imat_edge = np.zeros((IcD.elon.size), dtype=int)
  jmat_edge = np.zeros((IcD.elon.size), dtype=int)
  orie_edge = np.zeros((IcD.elon.size), dtype=dtype)
  #nx = 10
  #ny = 5
  for i in range(nx-1):
    if (i%5==0):
      print(f'i = {i}/{nx}')
    for j in range(ny-1):
      #if (i%5==0) and (j%5==0):
      #  print(f'i = {i}/{nx}, j = {j}/{ny}')
  
      # --- all cells in stripe
      # <\> for u integration
      ireg = ((IcD.clon>=lon_rg[i]) & (IcD.clon<lon_rg[i+1]))
      # <\> for v integration
      #ireg = ((clat>=lat_rg[j]) & (clat<lat_rg[j+1]))
      
      # --- all edges that belong to the cells of the stripe
      iedge = IcD.edge_of_cell[ireg]
      iedge = iedge.reshape(iedge.size)
      oedge = IcD.orientation_of_normal[ireg]
      oedge = oedge.reshape(iedge.size)
      # --- edges that appear only once and which are thus stripe boundaries
      #iedge_out, cnts = np.unique(iedge, return_counts=True)
      iedge_out, ind, cnts = np.unique(iedge, return_index=True, return_counts=True)
      #iedge_out = iedge[ind]
      iedge_out = iedge_out[cnts==1]
      oedge_out = oedge[ind]
      oedge_out = oedge_out[cnts==1]
      
      # <\> for u integration
      # --- only edges of western part
      mask = (  (IcD.elat[iedge_out]>=lat_rg[j]) & (IcD.elat[iedge_out]<lat_rg[j+1])
              & (IcD.elon[iedge_out]-lon_rg[i]<(lon_rg[1]-lon_rg[0])/2.) )
      # <\> for v integration
      ## --- only edges of southern part
      #mask = (  (elon[iedge_out]>=lon_rg[i]) & (elon[iedge_out]<lon_rg[i+1])
      #        & (elat[iedge_out]-lat_rg[j]<res/2.) )
      iedge_west = iedge_out[mask]
      oedge_west = oedge_out[mask] 
      imat_edge[iedge_west] = i
      jmat_edge[iedge_west] = j
      orie_edge[iedge_west] = oedge_west
  
  # <\> for u integration
  bstr = np.zeros((ny,nx), dtype=dtype)
  for i in range(nx-1):
    if (i%5==0):
      print(f'i = {i}/{nx}')
    for j in range(1,ny):
      mask = (imat_edge==i)&(jmat_edge==j)
      bstr[j,i] = bstr[j-1,i] + (mass_flux_vint[mask]*IcD.edge_length[mask]*orie_edge[mask]).sum()
  
  # <\> for v integration
  #bstr = np.zeros((ny,nx))
  #for i in range(1,nx):
  #  if (i%5==0):
  #    print(f'i = {i}/{nx}')
  #  for j in range(ny-1):
  #    mask = (imat_edge==i)&(jmat_edge==j)
  #    bstr[j,i] = bstr[j,i-1] + (mass_flux_vint[mask]*IcD.edge_length[mask]*orie_edge[mask]).sum()
  
  # --- subtract land value (find nearest point to Moscow)
  jl, il = np.unravel_index(np.argmin((Lon_rg-37)**2+(Lat_rg-55)**2), Lon_rg.shape)
  bstr += -bstr[jl,il]
  bstr *= 1e-6

  bstr = bstr.astype(IcD.dtype)
  
  # DEBUGGIN:
  if False:
    empt_data = np.ma.array(np.zeros(IcD.clon.shape), mask=True)

    hca, hcb = arrange_axes(3,2, plot_cb=True, sasp=0.5, fig_size_fac=2.,
                                sharex=True, sharey=True, xlabel="", ylabel="")
    ii=-1
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    shade(lon_rg, lat_rg, bstr, ax=ax, cax=cax, clim=60)
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    trishade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k')
    ax.scatter(Lon_rg, Lat_rg, s=5, c='r')
    #ax.set_xlim(-100,0)
    #ax.set_ylim(0,50)
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    trishade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k')
    # --- plotting
    ax.scatter(IcD.clon[ireg], IcD.clat[ireg], s=2, c='r')
    ax.scatter(IcD.elon[iedge], IcD.elat[iedge], s=2, c='b')
    ax.scatter(IcD.elon[iedge_out], IcD.elat[iedge_out], s=2, c='g')
    ax.scatter(IcD.elon[iedge_west], IcD.elat[iedge_west], s=2, c='y')
    #ax.scatter(elon[iedge_upp], elat[iedge_upp], s=2, c='y')
    #ax.set_xlim(-100,0)
    #ax.set_ylim(0,50)
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    trishade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k')
    imat_edge = np.ma.array(imat_edge, mask=imat_edge==0)
    ax.scatter(IcD.elon, IcD.elat, s=2, c=imat_edge, cmap='prism')
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    trishade(IcD.Tri, empt_data, ax=ax, cax=cax, edgecolor='k')
    jmat_edge = np.ma.array(jmat_edge, mask=jmat_edge==0)
    ax.scatter(IcD.elon, IcD.elat, s=2, c=jmat_edge, cmap='prism')

    plt.show()
    sys.exit()
  
  return bstr

def calc_moc(clat, wTransp, basin='global', fpath_fx='', res=1.0, dtype='float32'):
  if not os.path.exists(fpath_fx):
    raise ValueError('::: Error: Cannot find file %s! :::' % (fpath_fx))

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

  lat_mg = np.arange(-90.,90.,res)
  ny = lat_mg.size
  moc = np.zeros((wTransp.shape[0],lat_mg.size), dtype=dtype)
  for j in range(lat_mg.size-1):
    #ind = (clat>=lat_mg[j]) & (clat<lat_mg[j+1])
    #moc[:,j+1] = moc[:,j] + (wTransp[:,ind]*mask_basin[np.newaxis,ind]).sum(axis=1)
    ind = (clat>=lat_mg[ny-(j+2)]) & (clat<lat_mg[ny-(j+1)])
    moc[:,ny-(j+2)] = moc[:,ny-(j+1)] - (wTransp[:,ind]*mask_basin[np.newaxis,ind]).sum(axis=1)
  return moc

