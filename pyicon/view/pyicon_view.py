import sys, glob, os
import matplotlib.pyplot as plt
#import my_toolbox as my
import numpy as np
import cartopy.crs as ccrs
import cmocean
from netCDF4 import Dataset
import datetime
import time
from importlib import reload
import copy

import ipywidgets as widgets
from ipywidgets import interact, interactive, HBox, VBox
from IPython.display import display

import pyicon as pyic
reload(pyic)

#from ipdb import set_trace as mybreak  

# 'global ax' is needed to avoid flickering of the plots if they are actualized
# with global ax display(ax.figure) can be used instead of display(self.ax.figure)
global ax

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
# ------------------------------------------------------------ 
class hplot(object):
  output = widgets.Output()

  def __init__(self, IcD, grid_type='igrid', path_ckdtree='', logplot=False, verbose=False, lon_reg=[], lat_reg=[], rgrid_name=""):
    """
    Parameters
    ----------
    IcD : pyicon.IconData object
        asdf
    grid_type : str
        Type of grid which should be use: Either:
          * igrid: interpolate to regular grid which in specified by ckdtree
          * tgrid: original triangular grid
          * fgrid: take the grid from the file; e.g. if data was interpolated by cdo
    path_ckdtree: str
        Path to ckdtree interpolation file.
    logplot : bool
        Decide whether to apply log_10 to data before plotting.
    verbose : bool
        Switch on verbose output for debugging.
    lon_reg, lat_reg : list of len 2
        Lon and lat of region when not the full data set should be plotted
    rgrid_name : str
        Specify name of interplation grid. If "" default one is taken. Can be specified by dropdown manue as well.
    """
    # ------------------------------------------------------------ 
    # set parameters 
    # ------------------------------------------------------------ 
    self.verbose = verbose
    self.diag_out('parameters')
    # --- variable
    self.varnames = IcD.varnames
    self.IcD = copy.copy(IcD)
    self.var = self.varnames[0]
    # --- location
    self.iz = 0
    self.step_snap = 0
    # --- grid
    #self.use_tgrid = use_tgrid
    self.grid_type = grid_type # can be tgrid, igrid, fgrid
    if self.grid_type=='tgrid':
      self.use_tgrid = True
    else:
      self.use_tgrid = False
    self.path_ckdtree = path_ckdtree
    if (rgrid_name!="") and (rgrid_name in self.IcD.rgrid_names):
      self.IcD.rgrid_name = rgrid_name
    elif 'global_0.3' in self.IcD.rgrid_names:
      self.IcD.rgrid_name = 'global_0.3'
    else:
      self.IcD.rgrid_name  = self.IcD.rgrid_names[0]
    self.sec_name = ''
    self.IcD.rgrid_fpath = self.IcD.rgrid_fpaths[np.where(self.IcD.rgrid_names==self.IcD.rgrid_name)[0][0] ]
    self.sec_fpath = ''
    # --- plotting
    self.fpath_save = './test.pdf'
    self.clim = [-1,1]
    self.cmap = 'viridis'
    self.projection = 'PlateCarree'
    self.logplot = logplot

    # --- 
    if self.IcD.model_type=='atm':
      self.land_facecolor='none'
    else:
      self.land_facecolor='0.7'

    if len(lon_reg)==2:
      self.IcD.lon_reg = lon_reg
      self.IcD.lat_reg = lat_reg
      self.IcD.crop_data = True 

    if self.IcD.crop_data:
      if self.grid_type=='igrid' or self.grid_type=='fgrid':
        self.IcD.crop_rgrid(lon_reg=self.IcD.lon_reg, lat_reg=self.IcD.lat_reg)
      if self.grid_type=='tgrid':
        self.diag_out('make_triangulation')
        self.IcD.crop_tgrid(lon_reg=self.IcD.lon_reg, lat_reg=self.IcD.lat_reg)
        self.IcD.make_triangulation()

    # ------------------------------------------------------------ 
    # initialize plot
    # ------------------------------------------------------------ 
    self.diag_out('initialize_plot')
    self.initialize_plot()

    # debugging tips: * switch off all widget by commenting out the following line
    #                 * from Jupyter Notebook call:
    #    PyicV.update_fig_hplot(var='to', iz=0, step_snap=10, rgrid_name=PyicV.IcD.rgrid_name)
    # ------------------------------------------------------------ 
    # make widgets
    # ------------------------------------------------------------ 
    self.diag_out('widgets')

    # --- make depth slider
    b_dec, w1, b_inc = my_slide(name='depth:', bnds=[0,self.IcD.depthc.size-1])
    Box = HBox([b_dec, w1, b_inc])
    display(Box)

    # --- make time slider
    b_dec, w2, b_inc = my_slide(name='time:', bnds=[0,self.IcD.times.size-1])
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

    #print('hello world')
    a = interactive(self.update_fig_hplot, var=d2, iz=w1, step_snap=w2, rgrid_name=d3)
    display(self.output)
    return

  def diag_out(self, txt):
    if self.verbose==True:
    #if True:
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
      button_style='', # 'success', 'info', 'warning', 'danger' or ''
      tooltip='dec',
      icon=''
    )
    #b1.var = IP.var
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
      options=self.IcD.rgrid_names,
      value=self.IcD.rgrid_name,
      description='rgrid:',
      disabled=False,
                )
    #d3.observe(self.update_reg, names='value')
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

    # --- save button
    bs = widgets.Button(
      description='save',
      disabled=False,
      button_style='', # 'success', 'info', 'warning', 'danger' or ''
      tooltip='dec',
      icon=''
    )
    bs.on_click(self.save_fig)
    return ts, bs

  def initialize_plot(self, ax=None, cax=None, do_infostr=True):
    # --- initialize active variable
    self.IaV = self.IcD.vars[self.var]

    # reload grid because it might have been cropped in a wrong way before
    self.diag_out('reload grid')
    if self.grid_type=='igrid':
      self.IcD.load_rgrid()
      if self.IcD.crop_data:
        self.IcD.crop_rgrid(self.IcD.lon_reg, self.IcD.lat_reg)
    elif self.grid_type=='fgrid':
      self.IcD.load_grid_from_file()
      if self.IcD.crop_data:
        self.IcD.crop_rgrid(self.IcD.lon_reg, self.IcD.lat_reg)

    # synchronize with update_fig
    # --- load data 
    self.diag_out('load_hsnap')
    self.IaV.load_hsnap(fpath=self.IcD.flist_ts[self.step_snap], 
                        it=self.IcD.its[self.step_snap], 
                        iz=self.iz,
                        iw=self.IcD.iw,
                        step_snap = self.step_snap,
                        verbose = self.verbose,
                       ) 
    # --- interpolate data 
    if self.grid_type=='igrid':
      self.diag_out('interp_to_rectgrid')
      self.IaV.interp_to_rectgrid(fpath_ckdtree=self.IcD.rgrid_fpath, mask_reg=self.IcD.ind_reg_rec, indx=self.IcD.indx, indy=self.IcD.indy)
    # --- crop data
    elif self.IcD.crop_data and self.grid_type=='tgrid':
      self.IaV.data = self.IaV.data[self.IcD.ind_reg_tri]
    elif self.IcD.crop_data and self.grid_type=='fgrid':
      self.IaV.data = self.IaV.data[self.IcD.indy[:,np.newaxis], self.IcD.indx[np.newaxis,:]]

    # --- create axes
    # FIXME: Do we need this or can this be done by hplot_base?
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
      ii=-1
      ii+=1; ax=hca[ii]; cax=hcb[ii]

    # --- do plotting
    (self.ax, self.cax, 
     self.hm,
     self.Dstr
    ) = pyic.hplot_base(
                         self.IcD, self.IaV, 
                         ax=ax, cax=cax,
                         clim=self.clim, cmap=self.cmap,
                         title='auto', 
                         projection=self.projection,
                         use_tgrid=self.use_tgrid,
                         logplot=self.logplot,
                         do_plot_settings=False,
                        )
    #pyic.plot_settings(self.ax, template='global')
    pyic.plot_settings(self.ax, xlim=self.IcD.lon_reg, ylim=self.IcD.lat_reg, land_facecolor=self.land_facecolor)
    if do_infostr:
      # --- set info strings
      #self.hrstr = self.ax.text(0.05, 0.14, 'asdf', 
      #                     transform=plt.gcf().transFigure)
      #self.hdstr = self.ax.text(0.05, 0.08, 'asdf', 
      #                     transform=plt.gcf().transFigure)
      #self.htstr = self.ax.text(0.05, 0.025, 'asdf', 
      #                     transform=plt.gcf().transFigure)
      self.hrstr = self.ax.text(0.05, 0.025, 'asdf', transform=plt.gcf().transFigure)
      self.hdstr = self.ax.text(0.3, 0.025, 'asdf', transform=plt.gcf().transFigure)
      self.htstr = self.ax.text(0.6, 0.025, 'asdf', transform=plt.gcf().transFigure)
      self.hrstr.set_text('rgrid: %s'%(self.IcD.rgrid_name))
      self.hdstr.set_text('depth = %4.1fm'%(self.IcD.depthc[self.iz]))
      self.htstr.set_text(self.IcD.times[self.step_snap])

      self.fig = plt.gcf()
    self.diag_out('Done initializing!')
    return

  @output.capture()
  def update_fig_hplot(self, var, iz, step_snap, rgrid_name):
    #print('hello world')
    #print(var,iz,step_snap,rgrid_name)
    # --- update self
    self.var = var
    self.iz = iz
    self.step_snap = step_snap
    self.IaV = self.IcD.vars[self.var]

    # --- if the grid has changed
    if rgrid_name!=self.IcD.rgrid_name:
      self.IcD.rgrid_name = rgrid_name
      self.IcD.rgrid_fpath = self.IcD.rgrid_fpaths[np.where(self.IcD.rgrid_names==self.IcD.rgrid_name)[0][0] ]
      # grid of IcD needs to be updated as well since it is used by load_hsnap
      self.IcD.rgrid_name = self.IcD.rgrid_name
      self.IcD.load_rgrid()
      #self.initialize_plot(ax=self.ax, cax=self.cax, do_infostr=False)
      if self.IcD.crop_data:
        self.IcD.crop_tgrid(self.IcD.lon_reg, self.IcD.lat_reg)
        self.IcD.crop_rgrid(self.IcD.lon_reg, self.IcD.lat_reg)

      # synchronize with initialize_plot
      # --- load data 
      self.IaV.load_hsnap(fpath=self.IcD.flist_ts[self.step_snap], 
                          it=self.IcD.its[self.step_snap], 
                          iz=self.iz,
                          iw=self.IcD.iw,
                          step_snap = self.step_snap,
                          verbose = self.verbose,
                         ) 
      # --- interpolate data 
      if self.grid_type=='igrid':
        self.IaV.interp_to_rectgrid(fpath_ckdtree=self.IcD.rgrid_fpath, mask_reg=self.IcD.ind_reg_rec, indx=self.IcD.indx, indy=self.IcD.indy)
      # --- crop data
      elif self.IcD.crop_data and self.grid_type=='tgrid':
        self.IaV.data = self.IaV.data[self.IcD.ind_reg_tri]
      elif self.IcD.crop_data and self.grid_type=='fgrid':
        self.IaV.data = self.IaV.data[self.IcD.indy, self.IcD.indx]

      # --- remove old plot
      self.hm[0].remove()

      # --- do plotting
      #print(self.use_tgrid)
      #print(self.IcD.lon.shape)
      #print(self.IaV.data.shape)
      (self.ax, self.cax, 
       self.hm,
       self.Dstr
      ) = pyic.hplot_base(
                           self.IcD, self.IaV, 
                           ax=self.ax, cax=self.cax,
                           clim=self.clim, cmap=self.cmap,
                           title='auto', 
                           projection=self.projection,
                           use_tgrid=self.use_tgrid,
                           logplot=self.logplot,
                           do_plot_settings=False,
                          )
      #pyic.plot_settings(self.ax, template='global')
      pyic.plot_settings(self.ax, xlim=self.IcD.lon_reg, ylim=self.IcD.lat_reg, land_facecolor=self.land_facecolor)
    else:
      # synchronize with initialize_plot
      # --- load data 
      self.IaV.load_hsnap(fpath=self.IcD.flist_ts[self.step_snap], 
                          it=self.IcD.its[self.step_snap], 
                          iz=self.iz,
                          iw=self.IcD.iw,
                          step_snap = self.step_snap,
                          verbose = self.verbose,
                         ) 
      # --- interpolate data 
      if self.grid_type=='igrid':
        self.IaV.interp_to_rectgrid(fpath_ckdtree=self.IcD.rgrid_fpath, mask_reg=self.IcD.ind_reg_rec, indx=self.IcD.indx, indy=self.IcD.indy)
      # --- crop data
      elif self.IcD.crop_data and self.grid_type=='tgrid':
        self.IaV.data = self.IaV.data[self.IcD.ind_reg_tri]
      elif self.IcD.crop_data and self.grid_type=='fgrid':
        self.IaV.data = self.IaV.data[self.IcD.indy[:,np.newaxis], self.IcD.indx[np.newaxis,:]]

      # --- mask negative values for logplot 
      data = 1.*self.IaV.data
      if self.logplot:
        data[data<=0.0] = np.ma.masked
        data = np.ma.log10(data) 

      # --- update figure
      if self.use_tgrid:
        # cutting out masked values of Tri is necessary 
        # otherwise there are big messy triangles in plot
        data_nomasked_vals = data[self.IcD.mask_bt==False]
        self.hm[0].set_array(data_nomasked_vals)
      else:
        #self.hm[0].set_array(data[1:,1:].flatten())
        self.hm[0].set_array(data.flatten())

    # --- set info strings
    self.hrstr.set_text('rgrid: %s'%(self.IcD.rgrid_name))
    self.hdstr.set_text('depth = %4.1fm'%(self.IcD.depthc[self.iz]))
    self.htstr.set_text(self.IcD.times[self.step_snap])
    # --- update text
    if not self.logplot:
      self.ax.title.set_text(self.IaV.long_name+' ['+self.IaV.units+']')
    else:
      self.ax.title.set_text('log$_{10}$('+self.IaV.long_name+') ['+self.IaV.units+']')

    # 'global ax' is needed to avoid flickering of the plots if they are actualized
    # with global ax display(ax.figure) can be used instead of display(self.ax.figure)
    #display(ax.figure)
    #display(self.output)
    return

  def update_clim(self,w):
    climstr = w.owner.value
    if climstr!='auto':
      clim = np.array(climstr.split(',')).astype(float)  
      if clim.size==1:
        clim = np.array([-clim[0], clim[0]])
    try:
      #print clim
      self.hm[0].set_clim(clim)
      self.clim = clim
    except:
      print('Could not convert %s into clim.' % (climstr))
    return 

  def auto_clim(self,b1):
    if self.logplot:
      min_val = np.log10(np.abs(self.IaV.data).min())
      max_val = np.log10(np.abs(self.IaV.data).max())
    else:
      min_val = self.IaV.data.min()  
      max_val = self.IaV.data.max()  
    climstr = '%.2g, %.2g' % (min_val, max_val)
    b1.t1.value = climstr
    return

  def update_cmap(self,w):
    cmap = w.owner.value
    if cmap.startswith('cmo'):
      cmap = cmap.split('.')[-1]
      cmap = getattr(cmocean.cm, cmap)
    self.hm[0].set_cmap(cmap) 
    return

  #def update_reg(self, w):
  #  print(w.owner.value)
  #  print("Switching regions interactively is not yet supported!")
  #  return
    
  def update_fpath_save(self, w):
    self.fpath_save = w.owner.value
    return

  def save_fig(self, w):
    plt.savefig(self.fpath_save)
    print('Saving figure %s' % (self.fpath_save))
    return

  def show_parameter(self):
    print('--------------------') 
    # --- self
    plist = ['step_snap', 'sec_name', 'rgrid_name']
    plist = [
             'var', 'step_snap', #'it', 
             'path_ckdtree', 
             'sec_name', 'rgrid_name',
             'sec_fpath', 'rgrid_fpath',
             #'rgrid_name', 'rgrit_fpath', 'iz',
             'fpath_save', 'cmap', 'clim',
            ]
    for par in plist:
      epar = getattr(self, par)
      print('%s = '%(par),epar)
    print('--------------------') 
    # --- self.IcD
    plist = [
             'fpath_tgrid', 'fpath_fx',
             'lon_reg', 'lat_reg',
             'nz',
             'path_data', 'path_ckdtree',
             'path_rgrid', 
             'path_sections',
             'rgrid_fpath',
             'rgrid_name',
             'sec_names', 'rgrid_names',
            ]
    for par in plist:
      epar = getattr(self.IcD, par)
      print('IcD.%s = '%(par),epar)
    print('--------------------') 
    # --- self.IaV
    plist = [
               'name', 
               'fpath', 'is3d',
               'it', 
               'long_name', 'units'
            ]
    for par in plist:
      epar = getattr(self.IaV, par)
      print('IaV.%s = '%(par),epar)
    print('--------------------') 

# ------------------------------------------------------------ 
# vplot
# ------------------------------------------------------------ 
class vplot(hplot):
  output = widgets.Output()

  def __init__(self, IcD, log2vax=False, path_ckdtree='', logplot=False, verbose=False):
    # ------------------------------------------------------------ 
    # set parameters 
    # ------------------------------------------------------------ 
    self.verbose = verbose
    # --- variable
    self.IcD = copy.copy(IcD)
    # --- only keep 3d variables
    #vars_new = dict()
    varnames = []
    for var in self.IcD.vars.keys():
      if self.IcD.vars[var].is3d:
        #vars_new[var] = self.IcD.vars[var]
        varnames.append(var)
    self.varnames = varnames
    self.var = self.varnames[0]
    # --- location
    self.step_snap = 0
    # --- grid
    self.path_ckdtree = path_ckdtree
    self.sec_name  = self.IcD.sec_names[0]
    self.IcD.rgrid_name = ''
    self.sec_fpath = self.IcD.sec_fpaths[np.where(self.IcD.sec_names==self.sec_name)[0][0] ]
    self.IcD.rgrid_fpath = ''
    # --- plotting
    self.fpath_save = './test.pdf'
    self.clim = [-1,1]
    self.cmap = 'viridis'
    self.log2vax = log2vax
    self.logplot = logplot

    # ------------------------------------------------------------ 
    # initialize plot
    # ------------------------------------------------------------ 
    self.initialize_plot()

    # ------------------------------------------------------------ 
    # make widgets
    # ------------------------------------------------------------ 
    ## --- make depth slider
    #b_dec, w1, b_inc = my_slide(name='depth:', bnds=[0,self.IcD.depthc.size-1])
    #Box = HBox([b_dec, w1, b_inc])
    #display(Box)

    # --- make time slider
    b_dec, w2, b_inc = my_slide(name='time:', bnds=[0,self.IcD.times.size-1])
    Box = HBox([b_dec, w2, b_inc])
    display(Box)
    
    d2 = self.w_varname()
    t1, b1 = self.w_clim()
    d1 = self.w_cmap()
    Box = HBox([d2, t1, b1, d1])
    display(Box)

    #d3 = self.w_rgrid()
    d3 = self.w_sec()
    ts, bs = self.w_save_fig()
    Box = HBox([d3, ts, bs])
    display(Box)

    a = interactive(self.update_fig_vplot, var=d2, step_snap=w2, sec_name=d3)
    display(self.output)
    return
  
  def w_sec(self):
    # --- make section widget
    d3 = widgets.Dropdown(
      options=self.IcD.sec_names,
      value=self.sec_name,
      description='section:',
      disabled=False,
                )
    #d3.observe(self.update_sec, names='value')
    return d3

  def update_sec(self):
    a = 2
    return

  def initialize_plot(self, ax=None, cax=None, do_infostr=True):
    # --- initialize active variable
    self.IaV = self.IcD.vars[self.var]

    # --- load data
    self.IaV.load_vsnap(fpath=self.IcD.flist_ts[self.step_snap], 
                        fpath_ckdtree=self.sec_fpath,
                        it=self.IcD.its[self.step_snap], 
                        step_snap = self.step_snap,
                        verbose = self.verbose,
                       ) 

    # --- create axes
    if ax is None:
      hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=0.5, fig_size_fac=2.,
                                   sharex=False, sharey=False, xlabel="", ylabel="",
                                   dfigb=0.8,
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]

    # --- do plotting
    (self.ax, self.cax, 
     self.hm,
     self.Dstr
    ) = pyic.vplot_base(
                         self.IcD, self.IaV, 
                         ax=ax, cax=cax,
                         clim=self.clim, cmap=self.cmap,
                         title='auto', 
                         log2vax=self.log2vax,
                         logplot=self.logplot,
                        )
    if do_infostr:
      # --- set info strings
      self.hsstr = self.ax.text(0.05, 0.08, 'asdf', 
                           transform=plt.gcf().transFigure)
      self.htstr = self.ax.text(0.05, 0.025, 'asdf', 
                           transform=plt.gcf().transFigure)
      self.hsstr.set_text('section: %s'%(self.sec_name))
      self.htstr.set_text(self.IcD.times[self.step_snap])

    self.fig = plt.gcf()
    return

  @output.capture()
  def update_fig_vplot(self, var, step_snap, sec_name):
    #print('hello world')
    #print(var, step_snap, sec_name)
    # --- update self
    self.var = var
    self.step_snap = step_snap
    self.IaV = self.IcD.vars[self.var]

    if sec_name!=self.sec_name:
      self.sec_name = sec_name
      self.sec_fpath = self.IcD.sec_fpaths[np.where(self.IcD.sec_names==self.sec_name)[0][0] ]
      self.initialize_plot(ax=self.ax, cax=self.cax, do_infostr=False)
    else:
      # synchronize with initialize_plot
      # --- load data
      self.IaV.load_vsnap(fpath=self.IcD.flist_ts[self.step_snap], 
                          fpath_ckdtree=self.sec_fpath,
                          it=self.IcD.its[self.step_snap], 
                          step_snap = self.step_snap,
                          verbose = self.verbose,
                         ) 
      # --- mask negative values for logplot 
      data = 1.*self.IaV.data
      if self.logplot:
        data[data<=0.0] = np.ma.masked
        data = np.ma.log10(data) 
      # --- update figure
      #self.hm[0].set_array(data[1:,1:].flatten())
      self.hm[0].set_array(data.flatten())

    # --- set info strings
    self.hsstr.set_text('section: %s'%(self.sec_name))
    self.htstr.set_text(self.IcD.times[self.step_snap])
    # --- update text
    if not self.logplot:
      self.ax.title.set_text(self.IaV.long_name+' ['+self.IaV.units+']')
    else:
      self.ax.title.set_text('log$_{10}$('+self.IaV.long_name+') ['+self.IaV.units+']')

  # 'global ax' is needed to avoid flickering of the plots if they are actualized
  # with global ax display(ax.figure) can be used instead of display(self.ax.figure)
    #display(ax.figure)
    return
