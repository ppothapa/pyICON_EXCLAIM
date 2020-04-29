import sys, glob, os
import datetime
import numpy as np
from netCDF4 import Dataset, num2date
from scipy import interpolate
from scipy.spatial import cKDTree
import matplotlib
from ipdb import set_trace as mybreak  
from .pyicon_tb import *

class IconData(object):
  """
  Used by Jupyter
  """
  def __init__(self, 
               # data
               fname             = "",
               path_data         = "",
               # original grid   
               path_grid         = "",
               gname             = "",
               lev               = "",
               fpath_tgrid       = "auto",
               fpath_fx          = "auto",
               # interpolation   
               path_ckdtree      = "auto",
               #path_rgrid        = "auto", # not needed if conventions are followed
               #path_sections     = "auto", # not needed if convections are followed
               rgrid_name        = "",
               section_name      = "",
               # 
               run               = "auto",
               lon_reg           = [-180, 180],
               lat_reg           = [-90, 90],
               do_triangulation      = True,
               omit_last_file        = False,  # set to true to avoid data damage for running simulations
               load_vertical_grid    = True,
               load_triangular_grid  = True,
               load_rectangular_grid = True,
               load_variable_info    = True,
               calc_coeff            = True,
               time_mode             = 'num2date',
               model_type            = 'oce',
              ):


    # --- paths data and grid
    self.path_data     = path_data
    self.path_grid     = path_grid
    self.gname         = gname
    self.lev           = lev
    
    # --- automatically identify grid from data
    # (does not work anymore, maybe not necessary)
    if gname=='auto':
      pass
      #self.Dgrid = identify_grid(path_grid='', fpath_data=self.fpath_fx)
    
    # --- fpaths original grid
    if fpath_tgrid=='auto':
      self.fpath_tgrid   = self.path_grid + gname + '_tgrid.nc'
    else:
      self.fpath_tgrid   = fpath_tgrid

    if fpath_fx=='auto':
      self.fpath_fx = self.path_grid + self.gname + '_' + self.lev + '_fx.nc'
    else:
      self.fpath_fx = fpath_fx

    # --- paths ckdtree
    if path_ckdtree=='auto':
      self.path_ckdtree = self.path_grid + 'ckdtree/'
    else:
      self.path_ckdtree = path_ckdtree
    self.path_rgrid    = self.path_ckdtree + 'rectgrids/'
    self.path_sections = self.path_ckdtree + 'sections/'

    if run=='auto':
      self.run = self.path_data.split('/')[-2]
    else: 
      self.run = run

    # --- check if all important files and paths exist
    for pname in ['path_data', 'path_ckdtree', 'fpath_tgrid']: #, 'fpath_fx']:
      fp = getattr(self, pname)
      if not os.path.exists(fp):
        raise ValueError('::: Error: Cannot find %s: %s! :::' % (pname, fp))

    # --- global variables
    if rgrid_name=='orig':
      use_tgrid = True
      rgrid_name = ""
    else:
      use_tgrid = False
    self.interpolate = True
    self.units=dict()
    self.long_name=dict()
    self.data=dict()

    self.lon_reg = lon_reg
    self.lat_reg = lat_reg
    self.use_tgrid = use_tgrid
    self.fname = fname

    self.model_type = model_type

    # --- find regular grid ckdtrees for this grid
    sec_fpaths = np.array(
      glob.glob(self.path_sections+self.gname+'_*.npz'))
    sec_names = np.zeros(sec_fpaths.size, '<U200')
    self.sec_fpath_dict = dict()
    for nn, fpath_ckdtree in enumerate(sec_fpaths): 
      ddnpz = np.load(fpath_ckdtree)
      sec_names[nn] = ddnpz['sname']
      self.sec_fpath_dict[sec_names[nn]] = fpath_ckdtree
    self.sec_fpaths = sec_fpaths
    self.sec_names = sec_names

    if self.sec_names.size==0:
      print('::: Warning: Could not find any section-npz-file in %s. :::' 
                        % (self.path_sections))
      section_name = 'no_section_found'

    # --- find section grid ckdtrees for this grid
    rgrid_fpaths = np.array(
      glob.glob(self.path_rgrid+self.gname+'_*.npz'))
    rgrid_names = np.zeros(rgrid_fpaths.size, '<U200')
    self.rgrid_fpath_dict = dict()
    for nn, fpath_ckdtree in enumerate(rgrid_fpaths): 
      ddnpz = np.load(fpath_ckdtree)
      rgrid_names[nn] = ddnpz['sname']
      self.rgrid_fpath_dict[rgrid_names[nn]] = fpath_ckdtree
    self.rgrid_fpaths = rgrid_fpaths
    self.rgrid_names = rgrid_names

    if self.rgrid_names.size==0:
      print('::: Warning: Could not find any rgrid-npz-file in %s. :::' 
                        % (self.path_rectgrids))

    # --- choose rgrid and section
    # (do we need this? - yes, we load the rgrid later on)
    self.set_rgrid(rgrid_name)
    self.set_section(section_name)

    # ---------- 
    # the following can be computatinally expensive
    # ---------- 
    # --- load grid
    if load_triangular_grid:
      self.load_tgrid()
    if load_rectangular_grid:
      self.load_rgrid()
    if load_vertical_grid:
      self.load_vgrid()

    # --- calculate coefficients for divergence, curl, etc.
    if calc_coeff:
      self.calc_coeff()

    #self.crop_grid(lon_reg=self.lon_reg, lat_reg=self.lat_reg)
    # --- triangulation
    if do_triangulation:
      self.Tri = matplotlib.tri.Triangulation(self.vlon, self.vlat, 
                                              triangles=self.vertex_of_cell)
      self.mask_big_triangles()

    # --- list of variables and time steps / files
    if self.fname!="":
      self.get_files_of_timeseries()
      if omit_last_file:
        self.flist = self.flist[:-1]
      self.get_timesteps(time_mode=time_mode)

    if load_variable_info:
      self.get_varnames(self.flist[0])
      self.associate_variables(fpath_data=self.flist[0], skip_vars=[])
      #self.get_timesteps(time_mode='float2date')

    return

  def get_files_of_timeseries(self):
    self.times_flist, self.flist = get_files_of_timeseries(self.path_data, self.fname)
    return 
  
  def get_timesteps(self, time_mode='num2date'):
    self.times, self.flist_ts, self.its = get_timesteps(self.flist, time_mode=time_mode)
    self.nt = self.its.size
    return
  
  def get_varnames(self, fpath, skip_vars=[]):
    skip_vars = ['clon', 'clat', 'elon', 'elat', 'time', 'depth', 'lev']
    varnames = get_varnames(fpath, skip_vars)
    self.varnames = varnames
    return
  
  def reduce_tsteps(self, inds):
    if isinstance(inds, int):
      inds = np.arange(inds, dtype=int)
    self.times = self.times[inds]
    self.flist_ts = self.flist_ts[inds]
    self.its = self.its[inds]
    self.nt = self.its.size
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
      if hasattr(self, 'nz'):
        if (self.nz in shape) or ((self.nz+1) in shape):
          is3d = True
        else:
          is3d = False 
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
      rgrid_name = 'global_0.3'
      if not rgrid_name in self.rgrid_names:
        # if default does not exist, take first of list
        rgrid_name  = rgrid_names[0]
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
    elif sec_name=='no_section_found':
      print('::: Warning: no section found.:::')
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
    fpaths = glob.glob(self.path_rgrid+self.gname+'*.npz')
    print('regular grid files:')
    print(self.path_rgrid)
    for fp in fpaths:
      ddnpz = np.load(fp)
      info = ('{:40s} {:20s}').format(fp.split('/')[-1]+':', ddnpz['sname'])
      print(info)
    
    print('------------------------------------------------------------')
    fpaths = glob.glob(self.path_sections+self.gname+'*.npz')
    print('section files:')
    print(self.path_sections)
    for fp in fpaths:
      ddnpz = np.load(fp)
      info = ('{:40s} {:20s}').format(fp.split('/')[-1]+':', ddnpz['sname'])
      print(info)
    
    print('------------------------------------------------------------') 
    return

  def load_vgrid(self, lon_reg='all', lat_reg='all'):
    """ Load certain variables from self.fpath_fx which are typically related to a specification of the vertical grid.
    """

    if self.model_type=='oce':
      # --- vertical levels
      f = Dataset(self.fpath_fx, 'r')
      #self.clon = f.variables['clon'][:] * 180./np.pi
      #self.clat = f.variables['clat'][:] * 180./np.pi
      self.depthc = f.variables['depth'][:]
      self.depthi = f.variables['depth_2'][:]
      self.nz = self.depthc.size

      # --- the variables prism_thick_flat_sfc_c seem to be corrupted in fx file
      self.prism_thick_flat_sfc_c = f.variables['prism_thick_flat_sfc_c'][:] # delete this later
      self.prism_thick_c = f.variables['prism_thick_flat_sfc_c'][:]
      self.prism_thick_e = f.variables['prism_thick_flat_sfc_e'][:]
      self.constantPrismCenters_Zdistance = f.variables['constantPrismCenters_Zdistance'][:]
      self.dzw           = self.prism_thick_c
      self.dze           = self.prism_thick_e
      self.dzt           = self.constantPrismCenters_Zdistance

      self.dolic_c = f.variables['dolic_c'][:]-1
      self.dolic_e = f.variables['dolic_e'][:]-1
      self.wet_c = f.variables['wet_c'][:]
      self.wet_e = f.variables['wet_e'][:]

      #self.wet_e = f.variables['wet_e'][:]
      #for var in f.variables.keys():
      #  print(var)
      #  print(f.variables[var][:].max())
      #mybreak()
      f.close()
    elif self.model_type=='atm':
      pass
    else:
      raise ValueError('::: Error: Unknown model_type %s'%model_type)
    return

  def load_rgrid(self, lon_reg='all', lat_reg='all'):
    """ Load lon and lat from the ckdtree rectangular grid file self.rgrid_fpath.
    """
    # --- rectangular grid
    ddnpz = np.load(self.rgrid_fpath)
    self.lon = ddnpz['lon']
    self.lat = ddnpz['lat']
    self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
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
    """ Load certain variables related to the triangular grid from the grid file self.fpath_tgrid.
    """
    f = Dataset(self.fpath_tgrid, 'r')

    # --- lonn lat of cells, vertices and edges
    self.clon = f.variables['clon'][:] * 180./np.pi
    self.clat = f.variables['clat'][:] * 180./np.pi
    self.vlon = f.variables['vlon'][:] * 180./np.pi
    self.vlat = f.variables['vlat'][:] * 180./np.pi
    self.elon = f.variables['elon'][:] * 180./np.pi
    self.elat = f.variables['elat'][:] * 180./np.pi

    # --- distances and areas 
    self.cell_area = f.variables['cell_area'][:]
    self.cell_area_p = f.variables['cell_area_p'][:]
    self.edge_length = f.variables['edge_length'][:]
    self.dual_edge_length = f.variables['dual_edge_length'][:]
    # --- neighbor information
    self.vertex_of_cell = f.variables['vertex_of_cell'][:].transpose()-1
    self.edge_of_cell = f.variables['edge_of_cell'][:].transpose()-1
    self.vertices_of_vertex = f.variables['vertices_of_vertex'][:].transpose()-1
    self.edges_of_vertex = f.variables['edges_of_vertex'][:].transpose()-1
    self.edge_vertices = f.variables['edge_vertices'][:].transpose()-1
    self.adjacent_cell_of_edge = f.variables['adjacent_cell_of_edge'][:].transpose()-1
    # --- orientation
    self.orientation_of_normal = f.variables['orientation_of_normal'][:].transpose()
    self.edge_orientation = f.variables['edge_orientation'][:].transpose()
    f.close()

    return

  def calc_coeff(self):
    # --- derive coefficients
    self.div_coeff = (  self.edge_length[self.edge_of_cell] 
                      * self.orientation_of_normal 
                      / self.cell_area_p[:,np.newaxis] )
    # FIXME: Is grid_sphere_radius okay?
    #        Necessary to scale with grid_rescale_factor? (configure_model/mo_grid_config.f90)
    grid_sphere_radius = 6371e3
    self.rot_coeff = (  self.dual_edge_length[self.edges_of_vertex]
                      * grid_sphere_radius
                      * self.edge_orientation )
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
    self.Tri, self.maskTri = mask_big_triangles(self.vlon, self.vertex_of_cell, 
                                                self.Tri)
    return

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

  def time_average(self, IcD, t1, t2, it_ave=[], iz='all', always_use_loop=False, fpath_ckdtree=''):
    self.t1 = t1
    self.t2 = t2
    self.iz = iz
    self.data, self.it_ave = time_average(IcD, self.name, t1, t2, it_ave, iz, always_use_loop)
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
    self.lon, self.lat, self.data = interp_to_rectgrid(self.data, fpath_ckdtree, coordinates=self.coordinates)
    return

  def interp_to_section(self, fpath_ckdtree):
    if self.isinterpolated:
      raise ValueError('::: Variable %s is already interpolated. :::'%self.name)
    self.lon_sec, self.lat_sec, self.dist_sec, self.data = interp_to_section(self.data, fpath_ckdtree, coordinates=self.coordinates)
    return

