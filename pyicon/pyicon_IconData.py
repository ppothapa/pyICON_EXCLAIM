import sys, glob, os
import datetime
import numpy as np
from netCDF4 import Dataset, num2date
from scipy import interpolate
from scipy.spatial import cKDTree
import matplotlib
#from ipdb import set_trace as mybreak  
from .pyicon_tb import *
from .pyicon_calc import *

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
               lon_reg           = [],
               lat_reg           = [],
               do_triangulation      = True,
               omit_last_file        = False,  # set to true to avoid data damage for running simulations
               load_vertical_grid    = True,
               load_vgrid_depth      = 'auto',
               load_vgrid_dz         = 'auto',
               load_vgrid_mask       = 'auto',
               load_triangular_grid  = True,
               load_rectangular_grid = True,
               load_variable_info    = True,
               load_grid_from_data_file = False,
               load_xarray_dset      = False,
               xr_chunks             = None,
               calc_coeff            = True,
               calc_coeff_mappings   = False,
               do_only_timesteps     = False,
               time_mode             = 'num2date',
               model_type            = 'oce',
               do_conf_dwd           = False,
               output_freq           = 'auto',
               verbose               = False,
               dtype                 = 'float32',
              ):


    # ---
    self.verbose = verbose

    self.diag_out('set paths and fnames')

    # --- paths data and grid
    self.path_data     = path_data
    self.path_grid     = path_grid
    self.gname         = gname
    self.lev           = lev

    self.dtype         = dtype

    if do_only_timesteps:
      do_triangulation = False
      calc_coeff = False
      calc_coeff_mappings = False
      load_vertical_grid = False
      load_vgrid_depth = False
      load_vgrid_dz = False
      load_vgrid_mask = False
      load_triangular_grid = False
      load_rectangular_grid = False
      load_variable_info = True
      load_grid_from_data_file = False
      load_xarray_dset = False

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
    self.diag_out('set global variables')
    if rgrid_name=='orig':
      use_tgrid = True
      rgrid_name = ""
    else:
      use_tgrid = False
    self.interpolate = True
    self.units=dict()
    self.long_name=dict()
    self.data=dict()

    if len(lon_reg)==2:
      self.lon_reg = lon_reg
      self.lat_reg = lat_reg
      self.crop_data = True
    else:
      self.lon_reg = [-180, 180]
      self.lat_reg = [-90, 90]
      self.crop_data = False
    self.use_tgrid = use_tgrid
    self.fname = fname

    self.model_type = model_type
    self.iw = None

    # --- constants (from src/shared/mo_physical_constants.f90)
    self.grid_sphere_radius = 6.371229e6
    self.grav = 9.80665
    self.earth_angular_velocity = 7.29212e-05
    self.rho0 = 1025.022 
    self.rhoi = 917.0
    self.rhos = 300.0
    self.sal_ref = 35.
    self.sal_ice = 5.
    rcpl = 3.1733
    cpd = 1004.64 
    self.cp = (rcpl + 1.0) * cpd  # = 4192.6641119999995 J/kg/K
    self.tref = 273.15
    self.tmelt = 273.15
    self.tfreeze = -1.9
    self.alf = 2.8345e6-2.5008e6 # [J/kg]   latent heat for fusion

    # ---
    self.diag_out('find  ckdtrees etc.')

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
      #print('::: Warning: Could not find any rgrid-npz-file in %s. :::' 
      #                  % (self.path_rectgrids))
      print('::: Warning: Could not find any rgrid-npz-file. :::')

    # --- choose rgrid and section
    # (do we need this? - yes, we load the rgrid later on)
    self.set_rgrid(rgrid_name)
    self.set_section(section_name)

    # ---------- 
    # the following can be computatinally expensive
    # ---------- 
    # --- load grid
    if load_triangular_grid:
      self.diag_out('load tgrid')
      self.load_tgrid(do_conf_dwd=do_conf_dwd)
    if load_rectangular_grid:
      self.diag_out('load rgrid')
      self.load_rgrid()
    self.nz = 1
    if load_vertical_grid:
      if isinstance(load_vgrid_depth, str) and load_vgrid_depth=='auto':
        load_vgrid_depth = True
      if isinstance(load_vgrid_dz, str) and load_vgrid_dz=='auto':
        load_vgrid_dz = True
      if isinstance(load_vgrid_mask, str) and load_vgrid_mask=='auto':
        load_vgrid_mask = True
      self.diag_out('load vgrid')
      self.load_vgrid(load_vgrid_depth=load_vgrid_depth, load_vgrid_dz=load_vgrid_dz, load_vgrid_mask=load_vgrid_mask)

    # --- crop the grid
    if self.crop_data:
      self.diag_out('crop grid')
      self.crop_tgrid(lon_reg=self.lon_reg, lat_reg=self.lat_reg)
      self.crop_rgrid(lon_reg=self.lon_reg, lat_reg=self.lat_reg)
    else:
      self.ind_reg_rec = None
      self.indx = 'all'
      self.indy = 'all'

    # --- calculate coefficients for divergence, curl, etc.
    if calc_coeff:
      self.diag_out('calc_coeff')
      self.calc_coeff()
    
    if calc_coeff_mappings:
      self.diag_out('calc_coeff_mappings')
      self.calc_coeff_mappings() 

    # --- triangulation
    if do_triangulation:
      self.diag_out('do_triangulation')
      self.make_triangulation()

    # --- list of variables and time steps / files
    self.diag_out('list of variables and time steps')
    if self.fname!="":
      self.get_files_of_timeseries()
      if omit_last_file:
        self.flist = self.flist[:-1]
      self.get_timesteps(time_mode=time_mode)
    else:
      self.times = np.array([])
    
    # --- load xarray data set
    if load_xarray_dset:
      self.diag_out('open xarray mfdataset')
      self.ds = xr.open_mfdataset(
        self.path_data+self.fname, combine='nested', concat_dim='time', data_vars='minimal',
        coords='minimal', compat='override', join='override', 
        decode_times=False,
        chunks = xr_chunks,
      )
      self.ds['times'] = self.times
    
    # --- decide whether the data set consists of monthly or yearly averages (or something else)
    if output_freq=='auto':
      if self.times.size<2:
        raise ValueError("::: Error: Only one time step in data set found. Cannot determine output frequency from this. Either use longer time series or specify output frequency in IconData. :::")
      dt1 = (self.times[1]-self.times[0]).astype(float)/(86400)
      if dt1==365 or dt1==366:
        self.output_freq = 'yearly'
      elif dt1==28 or dt1==29 or dt1==30 or dt1==31:
        self.output_freq = 'monthly'
      else:
        self.output_freq = 'unknown'
    else:
      self.output_freq = output_freq

    if load_variable_info:
      self.diag_out('load_variable_info')
      self.get_varnames(self.flist[0])
      self.associate_variables(fpath_data=self.flist[0], skip_vars=[])
      #self.get_timesteps(time_mode='float2date')

    if load_grid_from_data_file: 
      self.diag_out('load_grid_from_file')
      self.load_grid_from_file()
    return

  def make_triangulation(self):
    self.Tri = matplotlib.tri.Triangulation(self.vlon, self.vlat, 
                                            triangles=self.vertex_of_cell)
    self.mask_big_triangles()
    return

  def diag_out(self, txt):
    if self.verbose==True:
      print('-v-: '+txt)
    return
  def get_files_of_timeseries(self):
    self.times_flist, self.flist = get_files_of_timeseries(self.path_data, self.fname)
    return 
  
  def get_timesteps(self, time_mode='num2date'):
    self.times, self.flist_ts, self.its = get_timesteps(self.flist, time_mode=time_mode)
    self.nt = self.its.size
    return
  
  def get_varnames(self, fpath, skip_vars=[]):
    skip_vars = ['clon', 'clat', 'elon', 'elat', 'time', 'depth', 'lev', 'height', 'height_bnds', 'lon', 'lat']
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
        rgrid_name  =self.rgrid_names[0]
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

  def load_vgrid(self, lon_reg='all', lat_reg='all', load_vgrid_depth=True, load_vgrid_dz=True, load_vgrid_mask=True):
    """ Load certain variables from self.fpath_fx which are typically related to a specification of the vertical grid.
    """

    if self.model_type=='oce':
      # --- vertical levels
      f = Dataset(self.fpath_fx, 'r')
      #self.clon = f.variables['clon'][:] * 180./np.pi
      #self.clat = f.variables['clat'][:] * 180./np.pi
      if load_vgrid_depth:
        self.depthc = f.variables['depth'][:]
        self.depthi = f.variables['depth_2'][:]
        self.nz = self.depthc.size

      if load_vgrid_dz:
      # FIXME (2020-06-05): still true? the variables prism_thick_flat_sfc_c seem to be corrupted in fx file
#      self.prism_thick_flat_sfc_c = f.variables['prism_thick_flat_sfc_c'][:] # delete this later
        self.prism_thick_c = f.variables['prism_thick_flat_sfc_c'][:]
        self.prism_thick_e = f.variables['prism_thick_flat_sfc_e'][:]
        self.constantPrismCenters_Zdistance = f.variables['constantPrismCenters_Zdistance'][:]
        self.dzw           = self.prism_thick_c
        self.dze           = self.prism_thick_e
        self.dzt           = self.constantPrismCenters_Zdistance

      if load_vgrid_mask:
        #self.dolic_c = f.variables['dolic_c'][:]-1
        #self.dolic_e = f.variables['dolic_e'][:]-1
        self.wet_c = f.variables['wet_c'][:]
        self.wet_e = f.variables['wet_e'][:]
  
        # land=2, land_bound=1, bound=0, sea_bound=-1, sea=-2 
        self.lsm_c = f.variables['lsm_c'][:]
        self.lsm_e = f.variables['lsm_e'][:]

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

  def load_grid_from_file(self):
    """ Load lon and lat from first netcdf file of list (e.g. if nc file was created by cdo)
    """
    # --- rectangular grid
    f = Dataset(self.flist[0], 'r')
    self.lon = f.variables['lon'][:]
    self.lat = f.variables['lat'][:]
    if (self.lon>180.).any():
      self.iw = (self.lon<180.).sum()
      self.lon = np.concatenate((self.lon[self.iw:], self.lon[:self.iw]))
      self.lon[self.lon>=180.] += -360. 
    else:
      self.iw = None
    try:
      self.height = f.variables['height'][:]
    except:
      print('::: Warning: No variable \'height\' in netcdf file. :::')
    try:
      self.depth = f.variables['depth'][:]
    except:
      print('::: Warning: No variable \'depth\' in netcdf file. :::')
    self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
    f.close()
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

  def load_tgrid(self,do_conf_dwd=False):
    """ Load certain variables related to the triangular grid from the grid file self.fpath_tgrid.
    """
    f = Dataset(self.fpath_tgrid, 'r')

    # --- lon lat of cells, vertices and edges
    self.clon = f.variables['clon'][:] * 180./np.pi
    self.clat = f.variables['clat'][:] * 180./np.pi
    self.vlon = f.variables['vlon'][:] * 180./np.pi
    self.vlat = f.variables['vlat'][:] * 180./np.pi
    self.elon = f.variables['elon'][:] * 180./np.pi
    self.elat = f.variables['elat'][:] * 180./np.pi

    # --- distances and areas 
    self.cell_area = f.variables['cell_area'][:]
    self.cell_area_p = f.variables['cell_area_p'][:]
    self.dual_area = f.variables['dual_area'][:]
    self.edge_length = f.variables['edge_length'][:]
    self.dual_edge_length = f.variables['dual_edge_length'][:]
    self.edge_cell_distance = f.variables['edge_cell_distance'][:].transpose()
    # --- neighbor information
    self.vertex_of_cell = f.variables['vertex_of_cell'][:].transpose()-1
    self.edge_of_cell = f.variables['edge_of_cell'][:].transpose()-1
    self.vertices_of_vertex = f.variables['vertices_of_vertex'][:].transpose()-1
    self.edges_of_vertex = f.variables['edges_of_vertex'][:].transpose()-1
    self.edge_vertices = f.variables['edge_vertices'][:].transpose()-1
    self.adjacent_cell_of_edge = f.variables['adjacent_cell_of_edge'][:].transpose()-1
    self.cells_of_vertex = f.variables['cells_of_vertex'][:].transpose()-1
    # --- orientation
    self.orientation_of_normal = f.variables['orientation_of_normal'][:].transpose()
    self.edge_orientation = f.variables['edge_orientation'][:].transpose()
    self.tangent_orientation = f.variables['edge_system_orientation'][:].transpose()

    if not do_conf_dwd:

       # --- masks
       self.cell_sea_land_mask = f.variables['cell_sea_land_mask'][:]
       self.edge_sea_land_mask = f.variables['edge_sea_land_mask'][:]

       # --- coordinates
       self.cell_cart_vec = np.ma.zeros((self.clon.size,3), dtype=self.dtype)
       self.cell_cart_vec[:,0] = f.variables['cell_circumcenter_cartesian_x'][:]
       self.cell_cart_vec[:,1] = f.variables['cell_circumcenter_cartesian_y'][:]
       self.cell_cart_vec[:,2] = f.variables['cell_circumcenter_cartesian_z'][:]

       self.vert_cart_vec = np.ma.zeros((self.vlon.size,3), dtype=self.dtype)
       self.vert_cart_vec[:,0] = f.variables['cartesian_x_vertices'][:]
       self.vert_cart_vec[:,1] = f.variables['cartesian_y_vertices'][:]
       self.vert_cart_vec[:,2] = f.variables['cartesian_z_vertices'][:]

       self.edge_cart_vec = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       self.edge_cart_vec[:,0] = f.variables['edge_middle_cartesian_x'][:]
       self.edge_cart_vec[:,1] = f.variables['edge_middle_cartesian_y'][:]
       self.edge_cart_vec[:,2] = f.variables['edge_middle_cartesian_z'][:]

       self.dual_edge_cart_vec = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       self.dual_edge_cart_vec[:,0] = f.variables['edge_dual_middle_cartesian_x'][:]
       self.dual_edge_cart_vec[:,1] = f.variables['edge_dual_middle_cartesian_y'][:]
       self.dual_edge_cart_vec[:,2] = f.variables['edge_dual_middle_cartesian_z'][:]

       self.edge_prim_norm = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       self.edge_prim_norm[:,0] = f.variables['edge_primal_normal_cartesian_x'][:]
       self.edge_prim_norm[:,1] = f.variables['edge_primal_normal_cartesian_y'][:]
       self.edge_prim_norm[:,2] = f.variables['edge_primal_normal_cartesian_z'][:]

    else:

       # --- coordinates
       #GB: at DWD no cartesian info in grid files --> calculate
       clon = f.variables['clon'][:]
       clat = f.variables['clat'][:]
       vlon = f.variables['vlon'][:]
       vlat = f.variables['vlat'][:]
       elon = f.variables['elon'][:]
       elat = f.variables['elat'][:]
       elon_pn = f.variables['zonal_normal_primal_edge'][:]
       elat_pn = f.variables['meridional_normal_primal_edge'][:]

       self.cell_cart_vec = np.ma.zeros((self.clon.size,3), dtype=self.dtype)
       self.cell_cart_vec[:,0] = np.cos(clat[:])*np.cos(clon[:])
       self.cell_cart_vec[:,1] = np.cos(clat[:])*np.sin(clon[:])
       self.cell_cart_vec[:,2] = np.sin(clat[:])

       self.vert_cart_vec = np.ma.zeros((self.vlon.size,3), dtype=self.dtype)
       self.vert_cart_vec[:,0] = np.cos(vlat[:])*np.cos(vlon[:])
       self.vert_cart_vec[:,1] = np.cos(vlat[:])*np.sin(vlon[:])
       self.vert_cart_vec[:,2] = np.sin(vlat[:])

       self.edge_cart_vec = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       self.edge_cart_vec[:,0] = np.cos(elat[:])*np.cos(elon[:])
       self.edge_cart_vec[:,1] = np.cos(elat[:])*np.sin(elon[:])
       self.edge_cart_vec[:,2] = np.sin(elat[:])

       self.dual_edge_cart_vec = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       #GB: this is not used anywhere so far, so we leave it as zero
       #self.dual_edge_cart_vec[:,0] = f.variables['edge_dual_middle_cartesian_x'][:]
       #self.dual_edge_cart_vec[:,1] = f.variables['edge_dual_middle_cartesian_y'][:]
       #self.dual_edge_cart_vec[:,2] = f.variables['edge_dual_middle_cartesian_z'][:]

       self.edge_prim_norm = np.ma.zeros((self.elon.size,3), dtype=self.dtype)
       self.edge_prim_norm[:,0] = - elon_pn[:]*np.sin(elon[:]) - elat_pn[:]*np.sin(elat[:])*np.cos(elon[:])
       self.edge_prim_norm[:,1] =   elon_pn[:]*np.cos(elon[:]) - elat_pn[:]*np.sin(elat[:])*np.sin(elon[:])
       self.edge_prim_norm[:,2] =   elat_pn[:]*np.cos(elat[:])

    f.close()

    return

  def calc_coeff(self):
    print('Start with calc_coeff...')

    # --- derive Coriolis parameter
    self.fc = 2.* self.earth_angular_velocity * np.sin(self.clat*np.pi/180.)
    self.fe = 2.* self.earth_angular_velocity * np.sin(self.elat*np.pi/180.)
    self.fv = 2.* self.earth_angular_velocity * np.sin(self.vlat*np.pi/180.)

    # --- derive coefficients
    self.div_coeff = (  self.edge_length[self.edge_of_cell] 
                      * self.orientation_of_normal 
                      / self.cell_area_p[:,np.newaxis] )
    self.grad_coeff = (1./self.dual_edge_length)
    # Necessary to scale with grid_rescale_factor? (configure_model/mo_grid_config.f90)
    #grid_sphere_radius = 1.
    rot_coeff = (  self.dual_edge_length[self.edges_of_vertex]/self.grid_sphere_radius
                      * self.grid_sphere_radius
                      * self.edge_orientation )

    if self.model_type=='oce':
      #iv = 3738
      #self.zarea_fraction = 0.
      self.zarea_fraction = np.ma.zeros((self.nz, self.vlon.size), dtype=self.dtype)
      self.rot_coeff = np.ma.zeros((self.nz, self.vlon.size, 6), dtype=self.dtype)
      for k in range(self.nz):
        #print(k)
        for ii in range(6):
          ie = self.edges_of_vertex[:,ii]
          ie_full = self.lsm_e[k,ie] == -2
          ie_half = self.lsm_e[k,ie] == 0
          ie_omit = self.lsm_e[k,ie] == 2
          i1 = self.adjacent_cell_of_edge[ie,0]
          i2 = self.adjacent_cell_of_edge[ie,1]
          # if both cells are water
          c1 = self.cell_cart_vec[i1,:]
          c2 = self.cell_cart_vec[i2,:]
          # if only one cell is water:
          #   * c1: take c1 but if c1 is land take c2
          #   * c2: take edge in between c1 and c2
          ind_c1_is_land = ((self.wet_c[k,i1]==0.) & (ie_half))
          c1[ind_c1_is_land] = c2[ind_c1_is_land]
          c2[ie_half,:] = self.edge_cart_vec[ie[ie_half],:] 
          partial_area = planar_triangle_area(c1, self.vert_cart_vec, c2)
          # if edge does not exist partial_area should be zero
          partial_area[ie_omit] = 0
          #print('cell1 = ', c1)
          #print('vert  = ', self.vert_cart_vec[iv,:])
          #print('cell2 = ', c2)
          #print('partial_area = ', partial_area[iv])
          self.zarea_fraction[k,:] += partial_area
        ind_vertex_without_valid_edge = self.lsm_e[k,self.edges_of_vertex].sum(axis=1)==0
        self.zarea_fraction[k,ind_vertex_without_valid_edge] = 0.
        #self.rot_coeff *= 1./(self.dual_area[:,np.newaxis]*grid_sphere_radius**2)
        self.rot_coeff[k,:,:] = rot_coeff/(self.zarea_fraction[k,:,np.newaxis]*self.grid_sphere_radius**2)
    else:
      self.rot_coeff = rot_coeff/(self.dual_area[:,np.newaxis])
    print('Done with calc_coeff!')
    return

  def calc_coeff_mappings(self):
    print('Start with calc_coeff_mappings...')

    print('--- fixed_vol_norm')
    self.fixed_vol_norm = calc_fixed_volume_norm(self)

    print('--- edge2edge_viacell_coeff')
    self.edge2edge_viacell_coeff = calc_edge2edge_viacell_coeff(self)

    print('--- edge2cell_coeff_cc')
    self.edge2cell_coeff_cc = calc_edge2cell_coeff_cc(self)

    print('--- edge2cell_coeff_cc_t')
    self.edge2cell_coeff_cc_t = calc_edge2cell_coeff_cc_t(self)

    print('Done with calc_coeff_mappings!')
    return

  
  def crop_tgrid(self, lon_reg, lat_reg):
    """ Crop all cell related variables (data, clon, clat, vertex_of_cell, edge_of_cell to regin defined by lon_reg and lat_reg.
    """
    # --- crop tripolar grid
    (self.clon, self.clat,
     self.vertex_of_cell, self.edge_of_cell,
     self.ind_reg_tri ) = crop_tripolar_grid(lon_reg, lat_reg,
                                         self.clon, self.clat, 
                                         self.vertex_of_cell,
                                         self.edge_of_cell)
    #self.crop_data = True
    return

  def crop_rgrid(self, lon_reg, lat_reg):
    # --- crop rectangular grid
    (self.Lon, self.Lat, self.lon, self.lat, 
     self.ind_reg_rec, self.indx, self.indy) = crop_regular_grid(lon_reg, lat_reg, self.Lon, self.Lat)
    #self.crop_data = True
    return

  def mask_big_triangles(self):
    self.Tri, self.mask_bt = mask_big_triangles(self.vlon, self.vlat, self.vertex_of_cell, 
                                                self.Tri)
    return
  
  def load_timeseries(self, 
                      var,
                      ave_freq=0,
                      mode_ave='mean',
                     ):
    times = np.copy(self.times)
    if ave_freq>0:
      nskip = times.size%ave_freq
      if nskip>0:
        times = times[:-nskip]
      nresh = int(times.size/ave_freq)
      times = np.reshape(times, (nresh, ave_freq)).transpose()
      #times_ave = times.mean(axis=0)
      times = times[int(ave_freq/2),:] # get middle of ave_freq

    data = np.array([])
    for nn, fpath in enumerate(self.flist):
      f = Dataset(fpath, 'r')
      data_file = f.variables[var][:,0,0]
      data = np.concatenate((data, data_file))
      f.close()
    #print(f'{var}: {data.size}')
    # --- apply time averaging
    if ave_freq>0:
      if nskip>0:
        data = data[:-nskip]
      #if times.size%ave_freq != 0:
      #  raise ValueError(f'::: Time series has wrong size: {times.size} for ave_req={ave_freq}! :::')
      print(f'{var}: {data.size} {times.size}')
      data = np.reshape(data, (nresh, ave_freq)).transpose()
      if mode_ave=='mean':
        data = data.mean(axis=0)
      elif mode_ave=='min':
        data = data.min(axis=0)
      elif mode_ave=='max':
        data = data.max(axis=0)
    return times, data

class IconVariable(object):
  def __init__(self, name, units='', long_name='', 
                     coordinates='clat clon', fpath_ckdtree='',
                     is3d=None, isinterpolated=False,
                     dtype='float32',
               ):
    self.name = name
    self.units = units
    self.long_name = long_name
    self.is3d = is3d
    self.coordinates = coordinates
    self.isinterpolated = isinterpolated
    self.fpath_ckdtree = fpath_ckdtree
    self.dtype = dtype
    return

  def load_hsnap(self, fpath, it=0, iz=0, iw=None, step_snap=0, fpath_ckdtree='', mask_reg=None, indx='all', indy='all', verbose=True):
    self.step_snap = step_snap
    self.it = it
    self.iz = iz
    self.fpath = fpath

    self.data = load_hsnap(fpath, self.name, it=it, iz=iz, iw=iw, fpath_ckdtree=fpath_ckdtree, verbose=verbose)
    self.mask = self.data.mask

    if fpath_ckdtree=='':
      self.isinterpolated = False
    else:
      self.interp_to_rectgrid(fpath_ckdtree, mask_reg=mask_reg, indx=indx, indy=indy)
      self.isinterpolated = True
    return

  def time_average(self, IcD, t1, t2, it_ave=[], iz='all', always_use_loop=False, fpath_ckdtree='', mask_reg=None, indx='all', indy='all'):
    self.t1 = t1
    self.t2 = t2
    self.iz = iz
    self.data, self.it_ave = time_average(IcD, self.name, t1, t2, it_ave, iz, always_use_loop)
    self.mask = self.data.mask

    if fpath_ckdtree=='':
      self.isinterpolated = False
    else:
      self.interp_to_rectgrid(fpath_ckdtree, mask_reg=mask_reg, indx=indx, indy=indy)
      self.isinterpolated = True
    return
  
  def load_vsnap(self, fpath, fpath_ckdtree, it=0, step_snap=0, verbose=True):
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
    if verbose:
      print("Loading %s from %s" % (var, fpath))
    if f.variables[var].ndim==2:
      raise ValueError('::: Warning: Cannot do section of 2D variable %s! :::'%var)
    nz = f.variables[var].shape[1]
    data = np.ma.zeros((nz,self.dist_sec.size), dtype=self.dtype)
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

  def load_moc(self, fpath, it=0, step_snap=0, verbose=True):
    self.step_snap = step_snap
    self.it = it
    self.fpath = fpath

    var = self.name
    if verbose:
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

  def interp_to_rectgrid(self, fpath_ckdtree, mask_reg=None, indx='all', indy='all'):
    if self.isinterpolated:
      raise ValueError('::: Variable %s is already interpolated. :::'%self.name)
    self.lon, self.lat, self.data = interp_to_rectgrid(self.data, fpath_ckdtree, mask_reg=mask_reg, indx=indx, indy=indy, coordinates=self.coordinates)
    return

  def interp_to_section(self, fpath_ckdtree):
    if self.isinterpolated:
      raise ValueError('::: Variable %s is already interpolated. :::'%self.name)
    self.lon_sec, self.lat_sec, self.dist_sec, self.data = interp_to_section(self.data, fpath_ckdtree, coordinates=self.coordinates)
    return

