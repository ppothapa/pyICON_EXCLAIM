print('numpy')
import numpy as np
print('xarray')
import xarray as xr
print('Done modules calc.')

def convert_tgrid_data(ds_tg):
    """ Convert xarray grid file to grid file compatible with pyicon function.

    Open classical ICON grid file by:
    ds_tg = xr.open_dataset(fpath_tg, chunks=dict())

    Then convert by:
    ds_IcD = pyic.convert_tgrid_data(ds_tg)

    ds_tg and ds_IcD are both lazy xarray data sets containing dask arrays.
    """
    ds_IcD = xr.Dataset()
    
    # --- constants (from src/shared/mo_physical_constants.f90)
    ds_IcD['grid_sphere_radius'] = 6.371229e6
    ds_IcD['grav'] = 9.80665
    ds_IcD['earth_angular_velocity'] = 7.29212e-05
    ds_IcD['rho0'] = 1025.022 
    ds_IcD['rhoi'] = 917.0
    ds_IcD['rhos'] = 300.0
    ds_IcD['sal_ref'] = 35.
    ds_IcD['sal_ice'] = 5.
    rcpl = 3.1733
    cpd = 1004.64 
    ds_IcD['cp'] = (rcpl + 1.0) * cpd
    ds_IcD['tref'] = 273.15
    ds_IcD['tmelt'] = 273.15
    ds_IcD['tfreeze'] = -1.9
    ds_IcD['alf'] = 2.8345e6-2.5008e6 # [J/kg]   latent heat for fusion

    # --- distances and areas 
    ds_IcD['cell_area'] = ds_tg['cell_area']
    ds_IcD['cell_area_p'] = ds_tg['cell_area_p']
    ds_IcD['dual_area'] = ds_tg['dual_area']
    ds_IcD['edge_length'] = ds_tg['edge_length']
    ds_IcD['dual_edge_length'] = ds_tg['dual_edge_length']
    ds_IcD['edge_cell_distance'] = ds_tg['edge_cell_distance'].transpose()
    # --- neighbor information
    ds_IcD['vertex_of_cell'] = ds_tg['vertex_of_cell'].transpose()-1
    ds_IcD['edge_of_cell'] = ds_tg['edge_of_cell'].transpose()-1
    ds_IcD['vertices_of_vertex'] = ds_tg['vertices_of_vertex'].transpose()-1
    ds_IcD['edges_of_vertex'] = ds_tg['edges_of_vertex'].transpose()-1
    ds_IcD['edge_vertices'] = ds_tg['edge_vertices'].transpose()-1
    ds_IcD['adjacent_cell_of_edge'] = ds_tg['adjacent_cell_of_edge'].transpose()-1
    ds_IcD['cells_of_vertex'] = ds_tg['cells_of_vertex'].transpose()-1
    # --- orientation
    ds_IcD['orientation_of_normal'] = ds_tg['orientation_of_normal'].transpose()
    ds_IcD['edge_orientation'] = ds_tg['edge_orientation'].transpose()
    ds_IcD['tangent_orientation'] = ds_tg['edge_system_orientation'].transpose()

    # --- masks
    ds_IcD['cell_sea_land_mask'] = ds_tg['cell_sea_land_mask']
    ds_IcD['edge_sea_land_mask'] = ds_tg['edge_sea_land_mask']
    
    # --- coordinates
    ds_IcD['cell_cart_vec'] = xr.concat([
        ds_tg['cell_circumcenter_cartesian_x'],
        ds_tg['cell_circumcenter_cartesian_y'],
        ds_tg['cell_circumcenter_cartesian_z'],
    ], dim='cart').transpose()
    
    ds_IcD['vert_cart_vec'] = xr.concat([
        ds_tg['cartesian_x_vertices'],
        ds_tg['cartesian_y_vertices'],
        ds_tg['cartesian_z_vertices'],
    ], dim='cart').transpose()
    
    ds_IcD['edge_cart_vec'] = xr.concat([
        ds_tg['edge_middle_cartesian_x'],
        ds_tg['edge_middle_cartesian_y'],
        ds_tg['edge_middle_cartesian_z'],
    ], dim='cart').transpose()
    
    ds_IcD['dual_edge_cart_vec'] = xr.concat([
        ds_tg['edge_dual_middle_cartesian_x'],
        ds_tg['edge_dual_middle_cartesian_y'],
        ds_tg['edge_dual_middle_cartesian_z'],
    ], dim='cart').transpose()
    
    ds_IcD['edge_prim_norm'] = xr.concat([
        ds_tg['edge_primal_normal_cartesian_x'],
        ds_tg['edge_primal_normal_cartesian_y'],
        ds_tg['edge_primal_normal_cartesian_z'],
    ], dim='cart').transpose()
    
    ds_IcD['clon'] *= 180./np.pi
    ds_IcD['clat'] *= 180./np.pi
    ds_IcD['elon'] *= 180./np.pi
    ds_IcD['elat'] *= 180./np.pi
    ds_IcD['vlon'] *= 180./np.pi
    ds_IcD['vlat'] *= 180./np.pi

    ds_IcD['fc'] = 2.* ds_IcD.earth_angular_velocity * np.sin(ds_IcD.clat*np.pi/180.)
    ds_IcD['fe'] = 2.* ds_IcD.earth_angular_velocity * np.sin(ds_IcD.elat*np.pi/180.)
    ds_IcD['fv'] = 2.* ds_IcD.earth_angular_velocity * np.sin(ds_IcD.vlat*np.pi/180.)
    
    try:
        ds_IcD = ds_IcD.rename({'ncells': 'cell'})
    except:
        pass

    return ds_IcD


def convert_fxgrid_data(ds_fxg):
    # Here we assume the name mapping is alwys in the same order
    # This may not necessarily be the case so proceed with caution
    name_mapping = {
        "ncells": "cell",
        "ncells_2": "edge",
        "ncells_3": "vertex",
        "vertices": "nv",
        "vertices_2": "nc",
        "vertices_3": "ne"
        }

    for key in name_mapping:
        if key in ds_fxg.dims:
            ds_fxg = ds_fxg.rename_dims({key: name_mapping[key]})

    return ds_fxg
    

def convert_data(ds_data):
    # For some reason edge points sometimes have an nc of 4 rather than 2
    # So proceed with caution. We use the value nc, nv or ne to ascertain
    # What type of point something is.
    name_mapping = {
        2: ("nc", "edge"),
        3: ("nv", "cell"),
        4: ("nc", "edge"),
        6: ("ne", "vertex")
    }

    for key in ds_data.dims:
        if key.startswith("vertices"):
            vertex_suffix = key[8:]
            
            vertices = ds_data.dims[key]
            
            ds_data = ds_data.rename_dims({
                key: name_mapping[vertices][0],
                "ncells" + vertex_suffix: name_mapping[vertices][1]
                                           })
    return ds_data


def print_verbose(verbose=1, message="", verbose_stage=1):
  if verbose>=verbose_stage:
    print(message)
  return

def xr_crop_tgrid(ds_tg, ireg_c, verbose=1):
  """ Crop a grid file. 

  Input: 
  ------
  ds_tg: xarray Dataset, which contains the grid file
  ireg_c: numpy index list of cell-points which should by in cropped domain

  Output:
  ------
  ds_tg_cut: xarray Dataset, which contains (most of) the cropped grid variables.

  Example usage:
  --------------
  ds_tg = xr.open_mfdataset(fpath_tgrid)
  clon = ds_tg.clon.compute().data * 180./np.pi
  clat = ds_tg.clat.compute().data * 180./np.pi

  lon_reg_3 = [6, 10]
  lat_reg_3 = [-32, -30]
  ireg_c = np.where(
      (clon>lon_reg[0]) & (clon<=lon_reg[1]) & (clat>lat_reg[0]) & (clat<=lat_reg[1])
  )[0]

  ds_tg_cut = pyic.xr_crop_tgrid(ds_tg, ireg_c)
  """
  # --- find edges and vertices belonging to cells of cutted domain
  print_verbose(verbose, "find edges")
  vertex_of_cell = ds_tg.vertex_of_cell[:,ireg_c].compute().data.transpose()-1
  edge_of_cell = ds_tg.edge_of_cell[:,ireg_c].compute().data.transpose()-1
  ireg_e, inde = np.unique(edge_of_cell, return_index=True)
  ireg_v, indv = np.unique(vertex_of_cell, return_index=True)
  
  # --- new dataset with cutted coordinates
  print_verbose(verbose, "cut coordinates")
  ds_tg_cut = xr.Dataset(coords=dict(
      clon=ds_tg['clon'][ireg_c],
      clat=ds_tg['clat'][ireg_c],
      elon=ds_tg['elon'][ireg_e],
      elat=ds_tg['elat'][ireg_e],
      vlon=ds_tg['vlon'][ireg_v],
      vlat=ds_tg['vlat'][ireg_v],
  ))
  ds_tg_cut['ireg_e'] = xr.DataArray(ireg_e, dims=['edge'])
  ds_tg_cut['ireg_v'] = xr.DataArray(ireg_v, dims=['vertex'])
  
  # --- re-index
  print_verbose(verbose, "reindex")
  reindex_c = np.zeros_like(ds_tg.clon, dtype='int32')-1
  reindex_c[ireg_c] = np.arange(ireg_c.size, dtype='int32')
  reindex_e = np.zeros_like(ds_tg.elon, dtype='int32')-1
  reindex_e[ireg_e] = np.arange(ireg_e.size, dtype='int32')
  reindex_v = np.zeros_like(ds_tg.vlon, dtype='int32')-1
  reindex_v[ireg_v] = np.arange(ireg_v.size, dtype='int32')
  
  var = 'vertex_of_cell'
  da = ds_tg[var][:,ireg_c]-1
  data = reindex_v[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  var = 'vertices_of_vertex'
  da = ds_tg[var][:,ireg_v]-1
  data = reindex_v[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  var = 'edge_of_cell'
  da = ds_tg[var][:,ireg_c]-1
  data = reindex_e[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  var = 'edges_of_vertex'
  da = ds_tg[var][:,ireg_v]-1
  data = reindex_e[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  var = 'adjacent_cell_of_edge'
  da = ds_tg[var][:,ireg_e]-1
  data = reindex_c[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  var = 'cells_of_vertex'
  da = ds_tg[var][:,ireg_v]-1
  data = reindex_c[da.data.flatten()].reshape(da.shape)
  ds_tg_cut[var] = xr.DataArray(data+1, dims=da.dims)
  
  # --- cut vertex variables
  print_verbose(verbose, "cut vertex variables")
  cvars = ['dual_area', 'edge_orientation',
          'cartesian_x_vertices', 'cartesian_y_vertices', 'cartesian_z_vertices']
  for var in cvars:
      ds_tg_cut[var] = ds_tg[var].isel(vertex=ireg_v)
  # --- cut edge variables
  print_verbose(verbose, "cut edge variables")
  cvars = ['edge_length', 'dual_edge_length', 'edge_sea_land_mask', 'edge_cell_distance',
          'edge_system_orientation',
          'edge_middle_cartesian_x', 'edge_middle_cartesian_y', 'edge_middle_cartesian_z',
          'edge_dual_middle_cartesian_x', 'edge_dual_middle_cartesian_y', 'edge_dual_middle_cartesian_z',
          'edge_primal_normal_cartesian_x', 'edge_primal_normal_cartesian_y', 'edge_primal_normal_cartesian_z']
  for var in cvars:
      ds_tg_cut[var] = ds_tg[var].isel(edge=ireg_e)
  # --- cut cell variables
  print_verbose(verbose, "cut cell variables")
  cvars = ['cell_area', 'cell_area_p', 'cell_sea_land_mask', 'orientation_of_normal', 
          'cell_circumcenter_cartesian_x', 'cell_circumcenter_cartesian_y', 'cell_circumcenter_cartesian_z']
  for var in cvars:
      ds_tg_cut[var] = ds_tg[var].isel(cell=ireg_c) 

  return ds_tg_cut

## Functions to map between 3D Cartesian and 2D local vectors
def xr_calc_2dlocal_from_3d(ds_IcD, p_vn_c):
    sinLon = np.sin(ds_IcD.clon*np.pi/180.)
    cosLon = np.cos(ds_IcD.clon*np.pi/180.)
    sinLat = np.sin(ds_IcD.clat*np.pi/180.)
    cosLat = np.cos(ds_IcD.clat*np.pi/180.)

    u1 = p_vn_c.isel(cart=0)
    u2 = p_vn_c.isel(cart=1)
    u3 = p_vn_c.isel(cart=2)

    uo =   u2*cosLon - u1*sinLon
    vo = -(u1*cosLon + u2*sinLon)*sinLat + u3*cosLat

    uo =   u2*cosLon - u1*sinLon
    vo = -(u1*cosLon + u2*sinLon)*sinLat + u3*cosLat
    return uo, vo

def xr_calc_3d_from_2dlocal(ds_IcD, uo, vo):
    sinLon = np.sin(ds_IcD.clon*np.pi/180.)
    cosLon = np.cos(ds_IcD.clon*np.pi/180.)
    sinLat = np.sin(ds_IcD.clat*np.pi/180.)
    cosLat = np.cos(ds_IcD.clat*np.pi/180.)

    u1 = -uo*sinLon - vo*sinLat*cosLon
    u2 =  uo*cosLon - vo*sinLat*sinLon
    u3 =  vo*cosLat

    new_dims = list(uo.dims)+['cart']
    p_vn_c = xr.concat([u1,u2,u3], dim='cart', coords='minimal').transpose(*new_dims)
    return p_vn_c

## Mapping between cells and edges

def xr_calc_edge2cell_coeff_cc_t(ds_IcD):
    dist_vector = ds_IcD.edge_cart_vec - ds_IcD.cell_cart_vec.isel(cell=ds_IcD.adjacent_cell_of_edge)#).transpose('edge', 'nc', 'cart')
    orientation = (dist_vector*ds_IcD.edge_prim_norm).sum(dim='cart')
    dist_vector *= np.sign(orientation)
    edge2cell_coeff_cc_t = (  ds_IcD.edge_prim_norm*ds_IcD.grid_sphere_radius
                              * np.sqrt((dist_vector**2).sum(dim='cart'))
                              / ds_IcD.dual_edge_length )
    edge2cell_coeff_cc_t = edge2cell_coeff_cc_t.transpose('edge', 'nc', 'cart')
    return edge2cell_coeff_cc_t

def xr_cell2edges(ds_IcD, p_vn_c, edge2cell_coeff_cc_t=None):
    if edge2cell_coeff_cc_t is None:
      edge2cell_coeff_cc_t = xr_calc_edge2cell_coeff_cc_t(ds_IcD)
    ic0 = ds_IcD.adjacent_cell_of_edge.isel(nc=0).data
    ic1 = ds_IcD.adjacent_cell_of_edge.isel(nc=1).data
    ptp_vn = (
        (
            p_vn_c.isel(cell=ic0).rename({'cell': 'edge'})#.chunk(dict(edge=ic0.size))
            * edge2cell_coeff_cc_t.isel(nc=0)
        ).sum(dim='cart')                                            
        +
        (
            p_vn_c.isel(cell=ic1).rename({'cell': 'edge'})#.chunk(dict(edge=ic0.size))
            * edge2cell_coeff_cc_t.isel(nc=1)
        ).sum(dim='cart') 
    )
    return ptp_vn

## Mapping between edges and cells

def xr_calc_fixed_volume_norm(ds_IcD):
    dist_vector = (
        ds_IcD.edge_cart_vec.isel(edge=ds_IcD.edge_of_cell) 
        - ds_IcD.cell_cart_vec
    )
    norm = np.sqrt((dist_vector**2).sum(dim='cart'))
    fixed_vol_norm = (
        0.5 * norm 
        * ds_IcD.edge_length.isel(edge=ds_IcD.edge_of_cell)
        / ds_IcD.grid_sphere_radius
    )
    fixed_vol_norm = fixed_vol_norm.sum(dim='nv')
    return fixed_vol_norm

def xr_calc_edge2cell_coeff_cc(ds_IcD):
    dist_vector = (
        ds_IcD.edge_cart_vec.isel(edge=ds_IcD.edge_of_cell) - ds_IcD.cell_cart_vec
    )
    edge2cell_coeff_cc = (  
        dist_vector * ds_IcD.edge_length.isel(edge=ds_IcD.edge_of_cell) / ds_IcD.grid_sphere_radius * ds_IcD.orientation_of_normal
    )
    edge2cell_coeff_cc = edge2cell_coeff_cc.compute()
    return edge2cell_coeff_cc

def xr_edges2cell(ds_IcD, ve, dze, dzc, edge2cell_coeff_cc=None, fixed_vol_norm=None):
    if fixed_vol_norm is None:
        fixed_vol_norm = xr_calc_fixed_volume_norm(ds_IcD)
    if edge2cell_coeff_cc is None:
        edge2cell_coeff_cc = xr_calc_edge2cell_coeff_cc(ds_IcD)
    if ve.dims != dze.dims:
      raise ValueError('::: Dims of ve and dze have to be the same!:::')
    p_vn_c = (
        edge2cell_coeff_cc 
        * ve.isel(edge=ds_IcD.edge_of_cell) 
        #* ds_fx.prism_thick_e.isel(edge=ds_IcD.edge_of_cell)
        * dze.isel(edge=ds_IcD.edge_of_cell)
    ).sum(dim='nv')
    if 'depth' in p_vn_c.dims:
        p_vn_c = p_vn_c.transpose('depth', 'cell', 'cart')
    
    p_vn_c = p_vn_c / (
        fixed_vol_norm 
        #* ds_fx.prism_thick_c
        * dzc
    )
    return p_vn_c

## Mapping between edges and edges

def xr_calc_edge2edge_viacell_coeff(ds_IcD):
    # FIXME: Continue here
    raise NotImplementedError("method not yet implemented")
    edge2edge_viacell_coeff = ()
    return edge2edge_viacell_coeff


def xr_edges2edges_via_cell(ds_IcD, vn_e, dze='const'):
    # FIXME: Continue here
    raise NotImplementedError("method not yet implemented")
    out_vn_e = ()
    return out_vn_e


## Divergence

def xr_calc_div_coeff(ds_IcD):
    div_coeff = (
        ds_IcD.edge_length.isel(edge=ds_IcD.edge_of_cell) * ds_IcD.orientation_of_normal / ds_IcD.cell_area
    )
    return div_coeff

def xr_calc_div(ds_IcD, vector, div_coeff=None):
    if div_coeff is None:
        div_coeff = xr_calc_div_coeff(ds_IcD)
    div_of_vector = (
        vector.isel(edge=ds_IcD.edge_of_cell) 
        * div_coeff
    ).sum(dim='nv')
    return div_of_vector

## Gradient

def xr_calc_grad_coeff(ds_IcD):
    grad_coeff = (
        1./ds_IcD.dual_edge_length
    )
    return grad_coeff

def xr_calc_grad(ds_IcD, scalar, grad_coeff=None):
    if grad_coeff is None:
        grad_coeff = xr_calc_grad_coeff(ds_IcD)
    grad_of_scalar = (
          scalar.isel(cell=ds_IcD.adjacent_cell_of_edge.isel(nc=1))
        - scalar.isel(cell=ds_IcD.adjacent_cell_of_edge.isel(nc=0))
    ) * grad_coeff
    return grad_of_scalar
