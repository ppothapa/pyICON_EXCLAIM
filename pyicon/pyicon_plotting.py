import sys, glob, os
import json
# --- calculations
import numpy as np
from scipy import interpolate
from scipy.spatial import cKDTree
# --- reading data 
from netCDF4 import Dataset, num2date
import datetime
# --- plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
#import my_toolbox as my
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
# --- debugging
from ipdb import set_trace as mybreak  
from importlib import reload

def hplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               adjust_axlims=True,
               projection='none', use_tgrid='auto',
               logplot=False,
               sasp=0.5,
               fig_size_fac=2.,
               crs_features=True,
              ):
  """
  IaV variable needs the following attributes
    * name
    * long_name
    * units
    * data
    if use_tgrid==True
    * Tri
    else 
    * lon
    * lat

  returns:
    * ax
    * cax
    * mappable
  """
  Dstr = dict()

  # --- plotting on original tgrid or interpolated rectgrid
  if isinstance(use_tgrid, str) and use_tgrid=='auto':
    use_tgrid = IcD.use_tgrid

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  if cmap.startswith('cmo'):
    cmap = cmap.split('.')[-1]
    cmap = getattr(cmocean.cm, cmap)
  else:
    cmap = getattr(plt.cm, cmap)

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
  else:
    ccrs_proj = getattr(ccrs, projection)()

  # --- make axes and colorbar (taken from shade)
  if ax == 'auto':
      #fig, ax = plt.subplots(subplot_kw={'projection': ccrs_proj}) 
    hca, hcb = arrange_axes(1,1, plot_cb=True, sasp=sasp, fig_size_fac=fig_size_fac,
                                 projection=ccrs_proj,
                                )
    ax = hca[0]
    cax = hcb[0]

  # --- do plotting
  if use_tgrid:
    hm = shade(IcD.Tri, IaV.data, 
                      ax=ax, cax=cax, clim=clim, cincr=cincr, cmap=cmap,
                      transform=ccrs_proj,
                      logplot=logplot,
                      adjust_axlims=adjust_axlims,
                 )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.clon.min(), IcD.clon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.clat.min(), IcD.clat.max()]
  else:
    hm = shade(IcD.lon, IcD.lat, IaV.data,
                      ax=ax, cax=cax, clim=clim, cincr=cincr, cmap=cmap,
                      transform=ccrs_proj,
                      logplot=logplot,
                      adjust_axlims=adjust_axlims,
              )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.lon.min(), IcD.lon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.lat.min(), IcD.lat.max()]

  mappable = hm[0]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  if (projection!='none') and (crs_features):
  #if projection=='PlateCarree':
  #if False:
    ax.coastlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='0.9')
    ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7), crs=ccrs_proj)
    ax.set_yticks(np.linspace(np.round(ylim[0]),np.round(ylim[1]),7), crs=ccrs_proj)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    #ax.stock_img()
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')
  return ax, cax, mappable, Dstr

def vplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               xvar='lat',
               log2vax=False,
               logplot=False,
               sasp=0.5,
               fig_size_fac=2.0,
              ):
  """
  IaV variable needs the following attributes
    * name
    * long_name
    * units
    * data
    * lon_sec, lat_sec, dist_sec

  returns:
    * ax
    * cax
    * mappable
  """
  Dstr = dict()

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  if cmap.startswith('cmo'):
    cmap = cmap.split('.')[-1]
    cmap = getattr(cmocean.cm, cmap)
  else:
    cmap = getattr(plt.cm, cmap)

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- make axes and colorbar (taken from shade)
  if ax == 'auto':
    hca, hcb = arrange_axes(1,1, plot_cb=True, sasp=sasp, fig_size_fac=fig_size_fac,
                           )
    ax = hca[0]
    cax = hcb[0]

  nz = IaV.data.shape[0]

  # --- do plotting
  if xvar=='lon':
    x = IaV.lon_sec
    xstr = 'longitude'
  elif xvar=='lat':
    x = IaV.lat_sec
    xstr = 'latitude'
  elif xvar=='dist':
    x = IaC.dist_sec/1e3
    xstr = 'distance [km]'
  if nz==IcD.depthc.size:
    depth = IcD.depthc
  else: 
    depth = IcD.depthi
  if log2vax:
    depth = np.log(depth)/np.log(2) 
  ylabel = 'depth [m]'

  hm = shade(x, depth, IaV.data,
                    ax=ax, cax=cax, clim=clim, cmap=cmap, cincr=cincr,
                    logplot=logplot,
            )
  if isinstance(xlim, str) and (xlim=='auto'):
    xlim = [x.min(), x.max()]
  if isinstance(ylim, str) and (ylim=='auto'):
    ylim = [depth.max(), depth.min()]

  mappable = hm[0]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xstr)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  if log2vax:
    ax.set_yticklabels(2**ax.get_yticks())
  ax.set_facecolor('0.8')
  ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7))
  ax.set_yticks(np.arange(0,5000,1000.))
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')

  return ax, cax, mappable, Dstr

def calc_conts(conts, clim, cincr, nclev):
  # ------ decide how to determine contour levels
  if isinstance(conts, np.ndarray) or isinstance(conts, list):
    # use given contours 
    conts = np.array(conts)
  else:
    # calculate contours
    # ------ decide wether contours should be calculated by cincr or nclev
    if cincr>0:
      conts = np.arange(clim[0], clim[1]+cincr, cincr)
    else:
      if isinstance(nclev,str) and nclev=='auto':
        nclev = 11
      conts = np.linspace(clim[0], clim[1], nclev)
  return conts

#class shade(object):
#  def __init__(self,
def shade(
              x='auto', y='auto', datai='auto',
              ax='auto', cax=0,
              cmap='auto',
              cincr=-1.,
              norm=None,
              rasterized=True,
              clim=[None, None],
              extend='both',
              conts=None,
              nclev='auto',
              #cint='auto', # old: use cincr now
              contcolor='k',
              contthick=0.,
              contfs=None,
              contlw=1.,
              use_pcol=True,
              cbticks='auto',
              adjust_axlims=True,
              bmp=None,
              transform=None,
              logplot=False,
              cbtitle='',
              edgecolor='none',
              cbdrawedges='auto',
           ):
    """ Convenient wrapper around pcolormesh, contourf, contour and their triangular versions.
    """
    # --- decide wether regular or triangular plots should be made
    if isinstance(datai, str) and datai=='auto':
      Tri = x
      datai = y
      rectangular_grid = False
    else:
      rectangular_grid = True

    # --- mask 0 and negative values in case of log plot
    data = 1.*datai
    if logplot and isinstance(data, np.ma.MaskedArray):
      data[data<=0.0] = np.ma.masked
      data = np.ma.log10(data) 
    elif logplot and not isinstance(data, np.ma.MaskedArray):
      data[data<=0.0] = np.nan
      data = np.log10(data) 
  
    # --- clim
    if isinstance(clim, str) and clim=='auto':
      clim = [None, None]
    elif isinstance(clim, str) and clim=='sym':
      clim = np.abs(data).max()
    clim=np.array(clim)
    if clim.size==1:
      clim = np.array([-1, 1])*clim
    if clim[0] is None:
      clim[0] = data.min()
    if clim[1] is None:
      clim[1] = data.max()
  
    # --- cmap
    if (clim[0]==-clim[1]) and cmap=='auto':
      cmap = 'RdBu_r'
    elif cmap=='auto':
      #cmap = 'viridis'
      cmap = 'RdYlBu_r'
    if isinstance(cmap, str):
      cmap = getattr(plt.cm, cmap)
  
    # --- norm
    if cincr>0.:
      clevs = np.arange(clim[0], clim[1]+cincr, cincr)
      norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
      use_norm = True
    else:
      norm = None
      use_norm = False
  
    # --- decide wether to use extra contour lines
    if conts is None:
      use_cont = False
    else:
      use_cont = True
      conts = calc_conts(conts, clim, cincr, nclev)
  
    # --- decide wether to use pcolormesh or contourf plotting
    if contfs is None:
      # use pcolormesh and not contourf
      use_contf = False
      use_pcol  = True
    else:
      # use contourf and not pcolormesh
      use_contf = True
      use_pcol  = False
      contfs = calc_conts(contfs, clim, cincr, nclev)
  
    # --- decide wether there should be black edges at colorbar
    if isinstance(cbdrawedges, str) and cbdrawedges=='auto':
      if use_norm or use_contf:
        cbdrawedges = True
      else:
        cbdrawedges = False
    else:
      cbdrawedges = False
  
    # --- necessary cartopy settings
    ccrsdict = dict()
    if transform is not None:
      ccrsdict = dict(transform=transform)
      #adjust_axlims = False
      #adjust_axlims = True
    
    # --- make axes if necessary
    if ax == 'auto':
      ax = plt.gca()
  
    if rectangular_grid:
      # --- adjust x and y if necessary
      # ------ make x and y 2D
      if x.ndim==1:
        x, y = np.meshgrid(x, y)
  
      # ------ convert to Basemap maps coordinates
      if bmp is not None:
        x, y = bmp(x, y)
        
      # ------ bring x and y to correct shape for contour
      if (use_cont) or (use_contf):
        if x.shape[1] != data.shape[1]:
          xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
          yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
        else:
          xc = 1.*x
          yc = 1.*y
      
    # --- allocate list of all plot handles
    hs = []
  
    # --- color plot
    # either pcolormesh plot
    if use_pcol:
      if rectangular_grid:
        hm = ax.pcolormesh(x, y, 
                           data, 
                           vmin=clim[0], vmax=clim[1],
                           cmap=cmap, 
                           norm=norm,
                           rasterized=rasterized,
                           edgecolor=edgecolor,
                           **ccrsdict
                          )
      else:
        hm = ax.tripcolor(Tri, 
                          data, 
                          vmin=clim[0], vmax=clim[1],
                          cmap=cmap, 
                          norm=norm,
                          rasterized=rasterized,
                          edgecolor=edgecolor,
                          **ccrsdict
                         )
      hs.append(hm)
    # or contourf plot
    elif use_contf:
      if rectangular_grid:
        hm = ax.contourf(xc, yc, 
                         data, contfs,
                         vmin=clim[0], vmax=clim[1],
                         cmap=cmap, 
                         norm=norm,
                         extend=extend,
                         **ccrsdict
                        )
      else:
        raise ValueError("::: Error: Triangular contourf not supported yet. :::")
        # !!! This does not work sinc Tri.x.size!=data.size which is natural for the picon Triangulation. Not sure why matplotlib tries to enforce this.
        #hm = ax.tricontourf(Tri,
        #                 data, contfs,
        #                 vmin=clim[0], vmax=clim[1],
        #                 cmap=cmap, 
        #                 norm=norm,
        #                 extend=extend,
        #                 **ccrsdict
        #                   )
      hs.append(hm)
  
      # this prevents white lines if fig is saved as pdf
      for cl in hm.collections: 
        cl.set_edgecolor("face")
        cl.set_rasterized(True)
      # rasterize
      if rasterized:
        zorder = -5
        ax.set_rasterization_zorder(zorder)
        for cl in hm.collections:
          cl.set_zorder(zorder - 1)
          cl.set_rasterized(True)
    else:
      hm = None
  
    # --- contour plot (can be in addition to color plot above)
    if use_cont:
      if rectangular_grid:
        hc = ax.contour(xc, yc, data, conts, 
                        colors=contcolor, linewidths=contlw, **ccrsdict)
      else:
        raise ValueError("::: Error: Triangular contour not supported yet. :::")
      # ------ if there is a contour matching contthick it will be made thicker
      try:
        i0 = np.where(hc.levels==contthick)[0][0]
        hc.collections[i0].set_linewidth(2.5*contlw)
      except:
        pass
      hs.append(hc)
  
    # --- colorbar
    if (cax!=0) and (hm is not None): 
      # ------ axes for colorbar needs to be created
      if cax == 1:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="10%", pad=0.1)
      # ------ make actual colorbar
      cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
      # ------ prevent white lines if fig is saved as pdf
      cb.solids.set_edgecolor("face")
      # ------ use exponential notation for large colorbar ticks
      cb.formatter.set_powerlimits((-3, 2))
      # ------ colorbar ticks
      if isinstance(cbticks, np.ndarray) or isinstance(cbticks, list):
        cb.set_ticks(cbticks)
      else:
        if norm is not None:
          cb.set_ticks(norm.boundaries[::2]) 
        else:
          cb.locator = ticker.MaxNLocator(nbins=5)
      cb.update_ticks()
      # ------ colorbar title
      cax.set_title(cbtitle)
      # ------ add cb to list of handles
      hs.append(cb)
  
    # --- axes labels and ticks
    if adjust_axlims:
      ax.locator_params(nbins=5)
      if rectangular_grid: 
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
      else:
        ax.set_xlim(Tri.x.min(), Tri.x.max())
        ax.set_ylim(Tri.y.min(), Tri.y.max())
    return hs 

# ================================================================================ 
def trishade(Tri, data,
            ax='auto', cax=0,
            cmap='auto',
            cincr=-1.,
            norm=None,
            rasterized=True,
            clim=[None, None],
            extend='both',
            edgecolor='none',
            conts=None,
            nclev='auto',
            cint='auto',
            contcolor='k',
            contthick=0.,
            contfs=None,
            contlw=1.,
            use_pcol=True,
            adjust_axlims=True,
            bmp=None,
            transform=None,
            logplot=False,
         ):
  """ Makes a nice tripcolor plot.

last change:
----------
2018-03-08
  """
  print("::: Warning pyic.trishade is outdated and pyic.shade should be used instead.")
  # mask 0 and negative values in case of log plot
  if logplot and isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.ma.masked
    data = np.ma.log10(data) 
  elif logplot and not isinstance(data, np.ma.MaskedArray):
    data[data<=0.0] = np.nan
    data = np.log10(data) 

  #clims
  if isinstance(clim, str) and clim=='auto':
    clim = [None, None]
  elif isinstance(clim, str) and clim=='sym':
    clim = np.abs(data).max()
  clim=np.array(clim)
  if clim.size==1:
    clim = np.array([-1, 1])*clim
  if clim[0] is None:
    clim[0] = data.min()
  if clim[1] is None:
    clim[1] = data.max()

  if (clim[0]==-clim[1]) and cmap=='auto':
    cmap = 'RdBu_r'
  elif cmap=='auto':
    #cmap = 'viridis'
    cmap = 'RdYlBu_r'
  if isinstance(cmap, str):
    cmap = getattr(plt.cm, cmap)

  if cincr>0.:
    clevs = np.arange(clim[0], clim[1]+cincr, cincr)
    norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
  else:
    norm = None

  # calculate contour x/y and contour levels if needed
  if conts is None:
    use_cont = False
  elif isinstance(conts,str) and conts=='auto':
    use_cont = True
    if isinstance(nclev,str) and nclev=='auto':
      conts = np.linspace(clim[0], clim[1], 11)
    else:
      conts = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      conts = np.arange(clim[0], clim[1]+cint, cint)
  else:
    use_cont = True
    conts = np.array(conts)

  if contfs is None:
    use_contf=False
  elif isinstance(contfs, str) and contfs=='auto':
    use_contf=True
    use_pcol=False
    if isinstance(nclev,str) and nclev=='auto':
      contfs = np.linspace(clim[0], clim[1], 11)
    else:
      contfs = np.linspace(clim[0], clim[1], nclev)
    if not (isinstance(cint,str) and cint=='auto'):
      contfs = np.arange(clim[0], clim[1]+cint, cint)
  elif isinstance(contfs, str) and contfs!='auto':
    use_contf=True
    use_pcol=False
    contfs = np.linspace(clim[0], clim[1], int(contfs))
  else:
    use_contf=True
    use_pcol=False
    contfs = np.array(contfs)

  ccrsdict = dict()
  if transform is not None:
    ccrsdict = dict(transform=transform)
    #adjust_axlims = False
    adjust_axlims = True
  
  # make axes if necessary
  if ax == 'auto':
    ax = plt.gca()

  #### make x and y 2D
  ###if x.ndim==1:
  ###  x, y = np.meshgrid(x, y)

  #### convert to Basemap maps coordinates
  ###if bmp is not None:
  ###  x, y = bmp(x, y)
  ###  
  #### bring x and y to correct shape for contour
  ###if (use_cont) or (use_contf):
  ###  if x.shape[1] != data.shape[1]:
  ###    xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
  ###    yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
  ###  else:
  ###    xc = 1.*x
  ###    yc = 1.*y
    
  hs = []
  # pcolor plot
  if use_pcol:

    hm = ax.tripcolor(Tri, data, 
                        edgecolor=edgecolor,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        rasterized=rasterized,
                        **ccrsdict
                      )
    hs.append(hm)
  # contourf plot
  elif use_contf:
    hm = ax.contourf(xc, yc, data, contfs,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        norm=norm,
                        extend=extend,
                        **ccrsdict
                      )
    # this prevents white lines if fig is saved as pdf
    for cl in hm.collections: 
      cl.set_edgecolor("face")
    # add handle to hanlde list
    hs.append(hm)
  else:
    hm = None

  # extra contours
  if use_cont:
    hc = ax.contour(xc, yc, data, conts, colors=contcolor, linewidths=contlw, **ccrsdict)
    try:
      i0 = np.where(conts==contthick)[0][0]
      #hc.collections[i0].set_linewidth(1.5)
      hc.collections[i0].set_linewidth(2.5*contlw)
    except:
      #print "::: Warning: Could not make contour contthick=%g thick. :::" % (contthick)
      pass
    hs.append(hc)

  # --- colorbar
  if (cax!=0) and (hm is not None): 
    # ------ axes for colorbar needs to be created
    if cax == 1:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      div = make_axes_locatable(ax)
      cax = div.append_axes("right", size="10%", pad=0.1)
    # ------ make actual colorbar
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    # ------ prevent white lines if fig is saved as pdf
    cb.solids.set_edgecolor("face")
    # ------ use exponential notation for large colorbar ticks
    cb.formatter.set_powerlimits((-3, 2))
    # ------ colorbar ticks
    if norm is None:
      tick_locator = ticker.MaxNLocator(nbins=8)
      cb.locator = tick_locator
      cb.update_ticks()
    else:
      cb.set_ticks(norm.boundaries[::2]) 
    # ------ add cb to list of handles
    hs.append(cb)

  # labels and ticks
  if adjust_axlims:
    ax.locator_params(nbins=5)
    ax.set_xlim(Tri.x.min(), Tri.x.max())
    ax.set_ylim(Tri.y.min(), Tri.y.max())
  return hs 

# ================================================================================ 

def arrange_axes( nx,ny,
                  # height of and aspect ratio of subplot
                  asy  = 3.5,
                  sasp = 0.5,
                  # plot colorbar
                  plot_cb = False,
                  # have common x or y axes
                  sharex = False, sharey = False,
                  xlabel = "",   ylabel = "",
                  # additional space left right and above and below axes
                  oxl = 0.1, oxr = 0.0,
                  oyb = 0.0, oyt = 0.0,
                  # factor that increases distance between axes
                  axfac_x = 1., axfac_y = 1.,
                  # kw for axes labels [(a), (b), etc.]
                  axlab_kw = dict(),
                  # figure size and aspect ratio
                  fig_size     = 'auto',
                  fig_asp      = 'auto',
                  fig_size_fac = 2.,
                  # figure title
                  fig_title = None,
                  projection = None,
                  ):
  """
last change:
----------
2015-07-22
 """ 

  # all lengths are in cm
  cm2inch = 0.3937        # to convert cm into inch

  # horizontal standard spaces
  alx = 1.0
  asx = asy / sasp
  adx = 0.5    
  cdx = 0.2
  clx = 0.8
  csx = 0.32 
  
  # vertical standard spaces
  aly = 0.8
  asy = asy
  ady = 0.2  
  aty = 0.6
  fty = 1.               # extra space for figure title (set to zero if fig_title = None)

  # apply manual changes to spaces
  adx = adx * axfac_x 
  ady = ady * axfac_y 
  #cdx = cdx * axfac_x   # this is a fix I do not understand why adxv is set to cdx if icbspace==True
  clx = clx * axfac_x

  if fig_title==None:
    fty = 0.

  # make vector of plot_cb if it has been true or false before
  # plot_cb can have values [{1}, 0] 
  # with meanings:
  #   1: plot cb; 
  #   0: do not plot cb
  if isinstance(plot_cb, bool) and (plot_cb==True):
    plot_cb = np.ones((nx,ny))  
    nohcb = False
  elif isinstance(plot_cb, bool) and (plot_cb==False):
    plot_cb = np.zeros((nx,ny))
    nohcb = True
  else:
    plot_cb = np.array(plot_cb)
    if plot_cb.size!=nx*ny:
      raise ValueError('Vector plot_cb has wrong length!')
    if plot_cb.shape[0]==nx*ny:
      plot_cb = plot_cb.reshape(ny,nx).transpose()
    elif plot_cb.shape[0]==ny:
      plot_cb = plot_cb.transpose()
    nohcb = False

  if not isinstance(projection, list):
    projection = [projection]*nx*ny

  # make spaces vectors
  # horizontal
  alxv = np.array([alx]*(nx))
  asxv = np.array([asx]*(nx))
  adxv = np.array([adx]*(nx))
  clxv = np.array([clx]*(nx))
  csxv = np.array([csx]*(nx))

  icbspace = plot_cb.sum(axis=1)>0
  csxv[icbspace==False] = 0.0
  clxv[icbspace==False] = 0.0
  adxv[icbspace==True ] = cdx
  if sharey:
    alxv[1:] = 0.0  

  # vertical
  alyv = np.array([aly]*(ny))
  asyv = np.array([asy]*(ny))
  adyv = np.array([ady]*(ny))
  atyv = np.array([aty]*(ny))

  if sharex:
    alyv[:-1] = 0.0

  # calculate figure size
  fw_auto = ( oxl + (alxv+asxv+adxv+csxv+clxv).sum() + oxr       )
  fh_auto = ( oyb + (alyv+asyv+adyv+atyv).sum()      + oyt + fty )
  if fig_size == 'auto':
    fw = fw_auto 
    fh = fh_auto 
  elif fig_size == 'dina4pt':
    fw = 21.0
    fh = 29.7
  elif fig_size == 'dina4ls':
    fw = 29.7
    fh = 21.0
  elif fig_size == 'jpo':
    fw = 15.5
    if fig_asp == 'auto':
      fh = fh_auto
    else:
      fh = fw*fig_asp
  elif isinstance( fig_size, (int,float) ):
    fw = fig_size
    if fig_asp == 'auto':
      fh = fh_auto
    else:
      fh = fw*fig_asp

  # make figure
  fasp = fh/fw
  hcf = plt.figure(figsize=(fw*cm2inch*fig_size_fac, fh*cm2inch*fig_size_fac))

  if not fig_title == None:
    hcf.suptitle(fig_title)

  # handle for axes
  hca = [0]*(nx*ny) 
  hcb = [0]*(nx*ny)

  kk = -1
  for jj in range(ny):
    for ii in range(nx):
      kk += 1

      # set axes x offspring
      if ii == 0:
        oxa = oxl + alxv[ii]
      else:
        oxa = oxa + alxv[ii] + (asxv+adxv+csxv+clxv)[ii-1]

      # set axes y offsping
      #if jj == 0 and ii == 0:
      #  oya = oyb + alyv[jj]
      #elif jj != 0 and ii == 0:
      #  oya = oya + alyv[jj] + (asyv+adyv+atyv)[jj-1]

      if jj == 0 and ii == 0:
        oya = fh - oyt - fty - (atyv+asyv)[jj]
      elif jj != 0 and ii == 0:
        oya =      oya - alyv[jj-1] - (adyv+atyv+asyv)[jj]

      # set colorbar x offspring
      oxc = oxa + (asxv+adxv)[ii]

      # calculated rectangles for axes and colorbar
      rect   = np.array([oxa, oya/fasp, asxv[ii], asyv[jj]/fasp])/fw
      rectcb = np.array([oxc, oya/fasp, csxv[ii], asyv[jj]/fasp])/fw
      
      # plot axes
      if projection[kk] is None:
        hca[kk] = plt.axes(rect, xlabel=xlabel, ylabel=ylabel)
      else:
        hca[kk] = plt.axes(rect, xlabel=xlabel, ylabel=ylabel, projection=projection[kk])

      # delet labels for shared axes
      if sharex and jj!=ny-1:
        hca[kk].ticklabel_format(axis='x',style='plain',useOffset=False)
        hca[kk].tick_params(labelbottom=False)
        hca[kk].set_xlabel('')

      if sharey and ii!=0:
        hca[kk].ticklabel_format(axis='y',style='plain',useOffset=False)
        hca[kk].tick_params(labelleft=False)
        hca[kk].set_ylabel('')

      # plot colorbars
      if plot_cb[ii,jj] == 1:
        hcb[kk] = plt.axes(rectcb, xticks=[])
        hcb[kk].yaxis.tick_right()

  # add letters for subplots
  if axlab_kw is not None:
    hca = axlab(hca, fontdict=axlab_kw)
  
  # return axes handles
  if nohcb:
    #plotsettings(hca)
    return hca, hcb
  else:
    #plotsettings(hca,hcb)
    return hca, hcb

# ================================================================================ 
def axlab(hca, figstr=[], posx=[-0.0], posy=[1.08], fontdict=None):
  """
input:
----------
  hca:      list with axes handles
  figstr:   list with strings that label the subplots
  posx:     list with length 1 or len(hca) that gives the x-coordinate in ax-space
  posy:     list with length 1 or len(hca) that gives the y-coordinate in ax-space
last change:
----------
2015-07-21
  """

  # make list that looks like [ '(a)', '(b)', '(c)', ... ]
  if len(figstr)==0:
    lett = "abcdefghijklmnopqrstuvwxyz"
    lett = lett[0:len(hca)]
    figstr = ["z"]*len(hca)
    for nn, ax in enumerate(hca):
      figstr[nn] = "(%s)" % (lett[nn])
  
  if len(posx)==1:
    posx = posx*len(hca)
  if len(posy)==1:
    posy = posy*len(hca)
  
  # draw text
  for nn, ax in enumerate(hca):
    ht = hca[nn].text(posx[nn], posy[nn], figstr[nn], 
                      transform = hca[nn].transAxes, 
                      horizontalalignment = 'right',
                      fontdict=fontdict)
    # add text handle to axes to give possibility of changing text properties later
    # e.g. by hca[nn].axlab.set_fontsize(8)
    hca[nn].axlab = ht
  return hca

def plot_settings(ax, xlim='none', ylim='none', xticks='auto', yticks='auto', 
                     ticks_position='both', template='none', 
                     # cartopy specific settings
                     projection=None, 
                     coastlines_color='none', coastlines_resolution='110m',
                     land_zorder=2, land_facecolor='0.7'):

  # --- templates
  if template=='global':
    xlim = [-180,180]
    ylim = [-90,90]
  elif template=='na':
    xlim = [-80,0]
    ylim = [30,70]
  elif template=='labsea':
    pass
  elif template=='zlat':
    pass
  elif template=='zlat_noso':
    pass

  # --- xlim, ylim
  if isinstance(xlim,str) and xlim=='none':
    xlim = ax.get_xlim()
  else:
    ax.set_xlim(xlim)
  if isinstance(ylim,str) and ylim=='none':
    ylim = ax.get_ylim()
  else:
    ax.set_ylim(ylim)
  
  # --- xticks, yticks
  if isinstance(xticks,str) and xticks=='auto':
    xticks = np.linspace(xlim[0],xlim[1],5) 
  if isinstance(yticks,str) and yticks=='auto':
    yticks = np.linspace(ylim[0],ylim[1],5) 
    
  if projection is None:
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
  else:
    ax.set_xticks(xticks, crs=projection)
    ax.set_yticks(yticks, crs=projection)
  
  if ticks_position=='both':
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

  # --- cartopy specific stuff
  if not projection is None: 
    ax.xaxis.set_major_formatter(cartopy.mpl.ticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cartopy.mpl.ticker.LatitudeFormatter() )
    ax.add_feature(cartopy.feature.LAND, zorder=land_zorder, facecolor=land_facecolor)
    if isinstance(coastlines_color, str) and coastlines_color!='none':
      ax.coastlines(color=coastlines_color, resolution=coastlines_resolution)
  return

