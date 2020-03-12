
def calc_wvel(IcD, mass_flux):
  div_mass_flux = (
    mass_flux[:,IcD.edge_of_cell]*IcD.div_coeff[np.newaxis,:,:]).sum(axis=2)
  wvel = np.zeros((IcD.nz+1, IcD.clon.size))
  wvel[:IcD.nz,:] = -div_mass_flux[::-1,:].cumsum(axis=0)[::-1,:]
  return wvel

def calc_vort(IcD, ve):
  # FIXME: this needs to be tested
  vort_v = (ve[:,IcD.edges_of_vertex] * IcD.rot_coeff).sum(axis=2)
  return vort_v

def edges2cell(IcD, ve):
  """
From math/mo_scalar_product.f90 map_edges2cell_no_height_3d_onTriangles:
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

  #edge2cell_coeff_cc = dist_vector * IcD.edge_length[:,np.newaxis] * IcD.orientation_of_normal[
  #p_vn_c = (edge2cell_coeff_cc[np.newaxis,:,:]*ve[:,IcD.edge_of_cell]*IcD.prism_thick_e[:,IcD.edge_of_cell]).sum(axis=1)
  return

# //////////////////////////////////////////////////////////////////////////////// 
# \\\\\ Calculation for ICON

def calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.):
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
  vertexIsAccounted_list = np.zeros((IcD.vlon.size))
  next_vertex_list = []
  
  # --- start vertex
  list_vertex_index = np.argmin((IcD.vlon-lon_start)**2+(IcD.vlat-lat_start)**2)
  vertexIsAccounted_list[list_vertex_index] = 1.
  next_vertex_list.append(list_vertex_index)
  
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
  print('start calculating stream')
  stream_variable = np.zeros((IcD.vlon.size))
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

  #bstr = IconVariable('bstr', units='Sv', long_name='barotropic streamfunction',
  #                   coordinates='vlat vlon', is3d=False)
  #bstr.data = stream_variable * 1e-6

  return bstr

def calc_bstr_rgrid(IcD, mass_flux_vint, lon_rg, lat_rg):
  """ Calculates barotropic streamfunction in Sv from mass_flux_vint on regular grid.

  """
  nx = lon_rg.size
  ny = lat_rg.size
  Lon_rg, Lat_rg = np.meshgrid(lon_rg, lat_rg)

  imat_edge = np.zeros((IcD.elon.size), dtype=int)
  jmat_edge = np.zeros((IcD.elon.size), dtype=int)
  orie_edge = np.zeros((IcD.elon.size))
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
  bstr = np.zeros((ny,nx))
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
