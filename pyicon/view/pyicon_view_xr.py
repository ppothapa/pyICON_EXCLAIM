import sys, glob, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cmocean
import xarray as xr
import pyicon as pyic

import ipywidgets as widgets
from ipywidgets import interact, interactive, HBox, VBox
from IPython.display import display

#from ipdb import set_trace as mybreak  

# 'global ax' is needed to avoid flickering of the plots if they are actualized
# with global ax display(ax.figure) can be used instead of display(self.ax.figure)
global ax

# ------------------------------------------------------------ 
# my_slice
# (combination of slider, +1/-1 buttons)
# ------------------------------------------------------------ 
def my_slide(name='slider:', bnds=[0,10]):
  w1 = widgets.IntSlider(
    value=0,
    min=bnds[0],
    max=bnds[1],
    step=1,
    description=name,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
  )
  
  b_inc = widgets.Button(
    description='+1',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='inc',
    icon=''
  )
  
  b_dec = widgets.Button(
    description='-1',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='dec',
    icon=''
  )
  
  def b_inc_update(b):
    val = w1.value
    val += 1
    if val > w1.max:
          val = w1.min   
    w1.value = val
      
  def b_dec_update(b):
    val = w1.value
    val -= 1
    if val < w1.min:
        val = w1.max
    w1.value = val
  
  b_inc.on_click(b_inc_update)
  b_dec.on_click(b_dec_update)
  #Box = HBox([b_dec, w1, b_inc])
  #display(Box)
  return b_dec, w1, b_inc

# ------------------------------------------------------------ 
# hplot
# (main class for horizontal plots)
# ------------------------------------------------------------ 
class hplot(object):
  output = widgets.Output()

  def __init__(self, 
    ds, 
    #grid_type='igrid', 
    path_grid='/mnt/lustre01/work/mh0033/m300602/icon/grids/',
    logplot=False, 
    verbose=False, 
    lon_reg=None, lat_reg=None, 
    land_facecolor='0.7',
    fpath_ckdtree='auto',
    fpath_tgrid='auto',
    do_mask_big_triangles=False,
    ):
    """
    Parameters
    ----------
    ds : xarray dataset
    grid_type : str
        Type of grid which should be use: Either:
          * igrid: interpolate to regular grid which in specified by ckdtree
          * tgrid: original triangular grid
          * fgrid: take the grid from the file; e.g. if data was interpolated by cdo
    path_grid: str
        Path to where all grid data is storred (needs to be set up according to pyicon standards)
    logplot : bool
        Decide whether to apply log_10 to data before plotting.
    verbose : bool 
        Switch on verbose output for debugging.
    lon_reg, lat_reg : list of len 2
        Lon and lat of region when not the full data set should be plotted
    land_facecolor : matplotlib color
        Face color of continents / land. Default is '0.7'; choose 'none' for transparency
    fpath_ckdtree : str
        Path to ckdtree interpolation file. Typically, this should stay 'auto'.
    fpath_tgrid : str
        Path to ICON grid file. Typically, this should stay 'auto'.
    do_mask_big_triangles : bool
        When plotting on the triangular grid, matplotlib can have problems with triangles
        which cross the map boundary. To avoid this, set do_mask_big_triangles=False.
      
    """
    # ------------------------------------------------------------ 
    # set parameters 
    # ------------------------------------------------------------ 
    self.verbose = verbose
    self.diag_out('parameters')
    # --- variable
    self.varnames = list(ds.keys())
    self.ds = ds.copy()
    self.var = self.varnames[0]
    # --- location
    self.iz = 0
    self.step_snap = 0
    self.step_snap = 0
    self.lon_reg = lon_reg
    self.lat_reg = lat_reg
    # --- grid
    grid_type = 'igrid'
    self.grid_type = grid_type # can be tgrid, igrid, fgrid
    if self.grid_type=='tgrid':
      self.use_tgrid = True
    else:
      self.use_tgrid = False
    Dgrid = pyic.identify_grid(path_grid, self.ds)
    self.Dgrid = Dgrid
    if fpath_ckdtree=='auto':
      if 'res0.30_180W-180E_90S-90N' in Dgrid['Drectgrids'].keys():
        # if default grid is there, take it
        self.rgrid_name = 'res0.30_180W-180E_90S-90N'
      else:
        # if default grid is not there, take first grid
        self.rgrid_name = Dgrid['Drectgrids'][Dgrid['Drectgrids'].keys()[0]]
      self.rgrid_options = Dgrid['Drectgrids']
    else:
      rgrid_name = 'pre_defined'
      self.rgrid_options = dict(pre_defined=fpath_ckdtree)
    self.rgrid_options['tgrid'] = fpath_tgrid
    if fpath_tgrid=='auto':
      self.fpath_tgrid = self.Dgrid['fpath_grid']
    else:
      self.fpath_tgrid = fpath_tgrid
    self.do_mask_big_triangles = do_mask_big_triangles
    #if self.use_tgrid:
    if True:
      #self.make_triangulation()
      ds_tgrid = xr.open_dataset(self.fpath_tgrid)
      self.ind_reg, self.Tri = pyic.triangulation(
        ds_tgrid,
        lon_reg=self.lon_reg, lat_reg=self.lat_reg,
        do_mask_big_triangles=self.do_mask_big_triangles,
      )

    # --- plotting
    self.fpath_save = './test.pdf'
    self.clim = [-1,1]
    self.cmap = 'viridis'
    self.projection = 'PlateCarree'
    self.logplot = logplot
    self.land_facecolor = land_facecolor

    # ------------------------------------------------------------ 
    # initialize plot
    # ------------------------------------------------------------ 
    self.diag_out('initialize_plot')
    self.initialize_plot()

    # debugging tips: * switch off all widget by commenting out the following lines
    #                 * from Jupyter Notebook call:
    #    PyicV.update_fig_hplot(var='to', iz=0, step_snap=10, rgrid_name=PyicV.IcD.rgrid_name)
    # ------------------------------------------------------------ 
    # make widgets
    # ------------------------------------------------------------ 
    self.diag_out('widgets')

    # --- make depth slider
    try:
      bnds=[0,self.ds.depth.size-1]
    except:
      bnds=[0,1]
    b_dec, w1, b_inc = my_slide(name='depth:', bnds=bnds)
    Box = HBox([b_dec, w1, b_inc])
    display(Box)

    # --- make time slider
    b_dec, w2, b_inc = my_slide(name='time:', bnds=[0,self.ds.time.size-1])
    Box = HBox([b_dec, w2, b_inc])
    display(Box)
    
    d2 = self.w_varname()
    t1, b1 = self.w_clim()
    d1 = self.w_cmap()
    Box = HBox([d2, t1, b1, d1])
    display(Box)

    d3 = self.w_rgrid()
    ts, bs = self.w_save_fig()
    Box = HBox([d3, ts, bs])
    display(Box)

    # ------------------------------------------------------------ 
    # Trigger update of plots
    # ------------------------------------------------------------ 
    a = interactive(self.update_fig_hplot, 
      var=d2,            # update variable name
      iz=w1,             # update depth level
      step_snap=w2,      # update time step 
      rgrid_name=d3,     # update grid information
    )
    display(self.output)
    return

  def diag_out(self, txt):
    if self.verbose==True:
      print('-v-: '+txt)
    return

  # ------------------------------------------------------------ 
  # widgets
  # ------------------------------------------------------------ 
  def w_clim(self):
    # --- make clim widget
    if isinstance(self.clim, list):
      climstr = '%g, %g' % (self.clim[0], self.clim[1])
    else:
      climstr = '%g, %g' % (-self.clim, self.clim)
    t1 = widgets.Text(
        value=climstr,
        placeholder='-100, 100',
        description='clim =',
        disabled=False
    )
    t1.continuous_update=False
    t1.observe(self.update_clim, names='value')

    # --- clim auto button
    b1 = widgets.Button(
      description='auto',
      disabled=False,
      button_style='',
      tooltip='dec',
      icon=''
    )
    b1.t1 = t1
    b1.on_click(self.auto_clim)
    return t1, b1

  def w_cmap(self):
    # --- make cmap widget
    d1 = widgets.Dropdown(
      options=['viridis', 'plasma', 'RdBu_r', 'RdYlBu_r', 'cmo.thermal', 'cmo.haline', 'cmo.ice', 'cmo.dense', 'cmo.curl', 'cmo.delta'],
      value='viridis',
      description='cmap:',
      disabled=False,
                )
    d1.observe(self.update_cmap, names='value')
    return d1

  def w_varname(self):
    # --- make varname widget
    d2 = widgets.Dropdown(
      options=self.varnames,
      value=self.varnames[0],
      description='var:',
      disabled=False,
                )
    return d2

  def w_rgrid(self):
    # --- make region widget
    d3 = widgets.Dropdown(
      options=self.rgrid_options.keys(),
      value=self.rgrid_name,
      description='rgrid:',
      disabled=False,
                )
    return d3

  def w_save_fig(self):
    # --- make save textbox
    ts = widgets.Text(
        value='./test.pdf',
        placeholder='./test.pdf',
        description='Name:',
        disabled=False
    )
    ts.continuous_update=False
    ts.observe(self.update_fpath_save, names='value')

    # --- make save button
    bs = widgets.Button(
      description='save',
      disabled=False,
      button_style='', # 'success', 'info', 'warning', 'danger' or ''
      tooltip='dec',
      icon=''
    )
    bs.on_click(self.save_fig)
    return ts, bs

  # ------------------------------------------------------------ 
  # functions to inquire ICON data from xarray object
  # ------------------------------------------------------------ 
  def get_data(self):
    self.diag_out('before arr')
    arr = self.ds[self.var].isel(time=self.step_snap)
    self.diag_out('after arr')
    self.fpath_ckdtree = self.rgrid_options[self.rgrid_name]
    # --- interpolate data 
    if self.grid_type=='igrid':
      self.diag_out('interp_to_rectgrid')
      try:
        arr = pyic.interp_to_rectgrid_xr(arr, self.fpath_ckdtree, lon_reg=self.lon_reg, lat_reg=self.lat_reg)
        self.diag_out('no error with interpolation')
      except:
        self.diag_out('error with interpolation')
    elif self.grid_type=='tgrid':
      arr = arr.isel(ncells=self.ind_reg)
    # --- take log
    if self.logplot:
      arr = xr.ufuncs.log10(xr.ufuncs.fabs(arr))
    self.arr = arr
    return 

  def get_title(self):
    try:
      long_name = self.arr.long_name
    except:
      #long_name = 'NA'
      long_name = self.arr.name
    try:
      units = self.arr.units
    except:
      units = 'NA'
    if self.logplot:
      title = f'log_10({long_name}) [{units}]'
    else:
      title = f'{long_name} [{units}]'
    return title

  def update_infotext(self):
    try:
      self.hdstr.set_text('depth = %4.1fm'%(self.ds.depth.data[self.iz]))
    except:
      self.hdstr.set_text('')
    self.htstr.set_text(str(self.ds.time.data[self.step_snap])[:16])
    self.title = self.get_title()
    self.ht_title.set_text(self.title)
    return

#  def make_triangulation(self): 
#    ds_tg = xr.open_dataset(self.fpath_tgrid)
#    clon = ds_tg['clon'].data * 180./np.pi
#    clat = ds_tg['clat'].data * 180./np.pi
#    vlon = ds_tg['vlon'].data * 180./np.pi
#    vlat = ds_tg['vlat'].data * 180./np.pi
#    vertex_of_cell = ds_tg['vertex_of_cell'].data.transpose()-1
#    if self.lon_reg is not None and self.lat_reg is not None:
#      self.ind_reg = np.where(   (clon>self.lon_reg[0])
#                               & (clon<=self.lon_reg[1])
#                               & (clat>self.lat_reg[0])
#                               & (clat<=self.lat_reg[1]) )[0]
#      vertex_of_cell = vertex_of_cell[self.ind_reg,:]
#    self.Tri = matplotlib.tri.Triangulation(vlon, vlat, triangles=vertex_of_cell)
#    return

  # ------------------------------------------------------------ 
  # initialize the plot (is only called once)
  # ------------------------------------------------------------ 
  def initialize_plot(self, ax=None, cax=None, do_infostr=True):
    # --- load data 
    self.diag_out('load_hsnap')
    self.get_data()

    # --- create axes
    if ax is None:
      if self.projection=='none':
        ccrs_proj = None
      else:
        ccrs_proj = getattr(ccrs, self.projection)()

      hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=0.5, fig_size_fac=2.,
                                   sharex=False, sharey=False, xlabel="", ylabel="",
                                   projection=ccrs_proj,
                                   #dfigb=1.4,
                                   dfigb=0.5,
                                  )
      self.ax, self.cax = hca[0], hcb[0]

    # --- do plotting
    self.shade_kwargs = dict(
      ax=self.ax, cax=self.cax, 
      clim=self.clim, cmap=self.cmap, 
      projection=ccrs.PlateCarree()
    )
    if self.lon_reg is None:
      self.xlim = 'none'
    else:
      self.xlim = self.lon_reg
    if self.lat_reg is None:
      self.ylim = 'none'
    else:
      self.ylim = self.lat_reg

    if self.use_tgrid:
      self.hm = pyic.shade(self.Tri, self.arr, **self.shade_kwargs)
    else:
      self.hm = pyic.shade(self.arr.lon, self.arr.lat, self.arr, **self.shade_kwargs)
    pyic.plot_settings(self.ax, xlim=self.xlim, ylim=self.ylim, 
      land_facecolor=self.land_facecolor)

    if do_infostr:
      # --- set info strings
      self.hrstr = self.ax.text(0.05, 0.025, 'NA', transform=plt.gcf().transFigure)
      self.hdstr = self.ax.text(0.3, 0.025, 'NA', transform=plt.gcf().transFigure)
      self.htstr = self.ax.text(0.6, 0.025, 'NA', transform=plt.gcf().transFigure)
      self.ht_title = self.ax.set_title('NA')

      self.update_infotext()

      self.fig = plt.gcf()
    self.diag_out('Done initializing!')
    return

  # ------------------------------------------------------------ 
  # main update function
  # (gathers information from widgets, loads new data, updates plot)
  # ------------------------------------------------------------ 
  @output.capture()
  def update_fig_hplot(self, var, iz, step_snap, rgrid_name):
    #print(var,iz,step_snap,rgrid_name)
    # --- update self
    self.var = var
    self.iz = iz
    self.step_snap = step_snap

    if rgrid_name=='tgrid':
      self.use_tgrid = True
      self.grid_type = 'tgrid'
    else:
      self.use_tgrid = False
      self.grid_type = 'igrid'

    # --- if the rgrid has changed, we need to re-create the plot
    if rgrid_name!=self.rgrid_name:
      # --- update self.rgrid_name
      self.rgrid_name = rgrid_name

      # --- remove pcolormesh plot
      self.hm[0].remove()

      # --- get data with new grid
      self.get_data()

      # --- make new plot
      if self.use_tgrid:
        self.hm = pyic.shade(self.Tri, self.arr, **self.shade_kwargs)
      else:
        self.hm = pyic.shade(self.arr.lon, self.arr.lat, self.arr, **self.shade_kwargs)
      pyic.plot_settings(self.ax, xlim=self.xlim, ylim=self.ylim, 
        land_facecolor=self.land_facecolor)
    # --- if rgrid is unchange or tgrid is used only update data
    else:
      self.get_data()
      self.hm[0].set_array(self.arr.data.flatten())

    # --- set info strings
    self.update_infotext()

    # 'global ax' is needed to avoid flickering of the plots if they are actualized
    # with global ax display(ax.figure) can be used instead of display(self.ax.figure)
    #display(ax.figure)
    #display(self.output)
    return

  # ------------------------------------------------------------ 
  # functions to trigger updating tha plots 
  # (called from widget action)
  # ------------------------------------------------------------ 
  def update_clim(self,w):
    climstr = w.owner.value
    if climstr!='auto':
      clim = np.array(climstr.split(',')).astype(float)  
      if clim.size==1:
        clim = np.array([-clim[0], clim[0]])
    try:
      self.hm[0].set_clim(clim)
      self.clim = clim
      self.shade_kwargs['clim'] = clim
    except:
      print('Could not convert %s into clim.' % (climstr))
    return 

  def auto_clim(self,b1):
    min_val = self.arr.min()
    max_val = self.arr.max()
    climstr = '%.2g, %.2g' % (min_val, max_val)
    b1.t1.value = climstr
    return

  def update_cmap(self,w):
    cmap = w.owner.value
    if cmap.startswith('cmo'):
      cmap = cmap.split('.')[-1]
      cmap = getattr(cmocean.cm, cmap)
    self.shade_kwargs['cmap'] = cmap
    self.hm[0].set_cmap(cmap) 
    return
    
  def update_fpath_save(self, w):
    self.fpath_save = w.owner.value
    return

  def save_fig(self, w):
    plt.savefig(self.fpath_save)
    print('Saving figure %s' % (self.fpath_save))
    return

#  def show_parameter(self):
#    print('--------------------') 
#    # --- self
#    plist = ['step_snap', 'sec_name', 'rgrid_name']
#    plist = [
#             'var', 'step_snap', #'it', 
#             'fpath_ckdtree', 
#             'sec_name', 'rgrid_name',
#             'sec_fpath', 'rgrid_fpath',
#             #'rgrid_name', 'rgrit_fpath', 'iz',
#             'fpath_save', 'cmap', 'clim',
#            ]
#    for par in plist:
#      epar = getattr(self, par)
#      print('%s = '%(par),epar)
#    print('--------------------') 
#    print('--------------------') 
#    return

##  # ------------------------------------------------------------ 
##  # vplot
##  # ------------------------------------------------------------ 
##  class vplot(hplot):
##    output = widgets.Output()
##  
##    def __init__(self, IcD, log2vax=False, path_ckdtree='', logplot=False, verbose=False):
##      # ------------------------------------------------------------ 
##      # set parameters 
##      # ------------------------------------------------------------ 
##      self.verbose = verbose
##      # --- variable
##      self.IcD = copy.copy(IcD)
##      # --- only keep 3d variables
##      #vars_new = dict()
##      varnames = []
##      for var in self.IcD.vars.keys():
##        if self.IcD.vars[var].is3d:
##          #vars_new[var] = self.IcD.vars[var]
##          varnames.append(var)
##      self.varnames = varnames
##      self.var = self.varnames[0]
##      # --- location
##      self.step_snap = 0
##      # --- grid
##      self.path_ckdtree = path_ckdtree
##      self.sec_name  = self.IcD.sec_names[0]
##      self.IcD.rgrid_name = ''
##      self.sec_fpath = self.IcD.sec_fpaths[np.where(self.IcD.sec_names==self.sec_name)[0][0] ]
##      self.IcD.rgrid_fpath = ''
##      # --- plotting
##      self.fpath_save = './test.pdf'
##      self.clim = [-1,1]
##      self.cmap = 'viridis'
##      self.log2vax = log2vax
##      self.logplot = logplot
##  
##      # ------------------------------------------------------------ 
##      # initialize plot
##      # ------------------------------------------------------------ 
##      self.initialize_plot()
##  
##      # ------------------------------------------------------------ 
##      # make widgets
##      # ------------------------------------------------------------ 
##      ## --- make depth slider
##      #b_dec, w1, b_inc = my_slide(name='depth:', bnds=[0,self.IcD.depthc.size-1])
##      #Box = HBox([b_dec, w1, b_inc])
##      #display(Box)
##  
##      # --- make time slider
##      b_dec, w2, b_inc = my_slide(name='time:', bnds=[0,self.IcD.times.size-1])
##      Box = HBox([b_dec, w2, b_inc])
##      display(Box)
##      
##      d2 = self.w_varname()
##      t1, b1 = self.w_clim()
##      d1 = self.w_cmap()
##      Box = HBox([d2, t1, b1, d1])
##      display(Box)
##  
##      #d3 = self.w_rgrid()
##      d3 = self.w_sec()
##      ts, bs = self.w_save_fig()
##      Box = HBox([d3, ts, bs])
##      display(Box)
##  
##      a = interactive(self.update_fig_vplot, var=d2, step_snap=w2, sec_name=d3)
##      display(self.output)
##      return
##    
##    def w_sec(self):
##      # --- make section widget
##      d3 = widgets.Dropdown(
##        options=self.IcD.sec_names,
##        value=self.sec_name,
##        description='section:',
##        disabled=False,
##                  )
##      #d3.observe(self.update_sec, names='value')
##      return d3
##  
##    def update_sec(self):
##      a = 2
##      return
##  
##    def initialize_plot(self, ax=None, cax=None, do_infostr=True):
##      # --- initialize active variable
##      self.IaV = self.IcD.vars[self.var]
##  
##      # --- load data
##      self.IaV.load_vsnap(fpath=self.IcD.flist_ts[self.step_snap], 
##                          fpath_ckdtree=self.sec_fpath,
##                          it=self.IcD.its[self.step_snap], 
##                          step_snap = self.step_snap,
##                          verbose = self.verbose,
##                         ) 
##  
##      # --- create axes
##      if ax is None:
##        hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=0.5, fig_size_fac=2.,
##                                     sharex=False, sharey=False, xlabel="", ylabel="",
##                                     dfigb=0.8,
##                                    )
##        ii=-1
##  
##        ii+=1; ax=hca[ii]; cax=hcb[ii]
##  
##      # --- do plotting
##      (self.ax, self.cax, 
##       self.hm,
##       self.Dstr
##      ) = pyic.vplot_base(
##                           self.IcD, self.IaV, 
##                           ax=ax, cax=cax,
##                           clim=self.clim, cmap=self.cmap,
##                           title='auto', 
##                           log2vax=self.log2vax,
##                           logplot=self.logplot,
##                          )
##      if do_infostr:
##        # --- set info strings
##        self.hsstr = self.ax.text(0.05, 0.08, 'asdf', 
##                             transform=plt.gcf().transFigure)
##        self.htstr = self.ax.text(0.05, 0.025, 'asdf', 
##                             transform=plt.gcf().transFigure)
##        self.hsstr.set_text('section: %s'%(self.sec_name))
##        self.htstr.set_text(self.IcD.times[self.step_snap])
##  
##      self.fig = plt.gcf()
##      return
##  
##    @output.capture()
##    def update_fig_vplot(self, var, step_snap, sec_name):
##      #print('hello world')
##      #print(var, step_snap, sec_name)
##      # --- update self
##      self.var = var
##      self.step_snap = step_snap
##      self.IaV = self.IcD.vars[self.var]
##  
##      if sec_name!=self.sec_name:
##        self.sec_name = sec_name
##        self.sec_fpath = self.IcD.sec_fpaths[np.where(self.IcD.sec_names==self.sec_name)[0][0] ]
##        self.initialize_plot(ax=self.ax, cax=self.cax, do_infostr=False)
##      else:
##        # synchronize with initialize_plot
##        # --- load data
##        self.IaV.load_vsnap(fpath=self.IcD.flist_ts[self.step_snap], 
##                            fpath_ckdtree=self.sec_fpath,
##                            it=self.IcD.its[self.step_snap], 
##                            step_snap = self.step_snap,
##                            verbose = self.verbose,
##                           ) 
##        # --- mask negative values for logplot 
##        data = 1.*self.IaV.data
##        if self.logplot:
##          data[data<=0.0] = np.ma.masked
##          data = np.ma.log10(data) 
##        # --- update figure
##        #self.hm[0].set_array(data[1:,1:].flatten())
##        self.hm[0].set_array(data.flatten())
##  
##      # --- set info strings
##      self.hsstr.set_text('section: %s'%(self.sec_name))
##      self.htstr.set_text(self.IcD.times[self.step_snap])
##      # --- update text
##      if not self.logplot:
##        self.ax.title.set_text(self.IaV.long_name+' ['+self.IaV.units+']')
##      else:
##        self.ax.title.set_text('log$_{10}$('+self.IaV.long_name+') ['+self.IaV.units+']')
##  
##    # 'global ax' is needed to avoid flickering of the plots if they are actualized
##    # with global ax display(ax.figure) can be used instead of display(self.ax.figure)
##      #display(ax.figure)
##      return
