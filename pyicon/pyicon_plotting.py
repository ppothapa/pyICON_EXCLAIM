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
from .pyicon_tb import write_dataarray_to_nc

def hplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               clevs=None,
               contfs=None,
               conts=None,
               contcolor='k',
               contthick=0.,
               contlw=1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               adjust_axlims=True,
               projection='none', use_tgrid='auto',
               logplot=False,
               asp=0.5,
               fig_size_fac=2.,
               crs_features=True,
               do_plot_settings=True,
               land_facecolor='0.7',
               do_write_data_range=False,
               save_data=False,
               fpath_nc='',
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
    * hm
  """
  Dstr = dict()

  # --- plotting on original tgrid or interpolated rectgrid
  if isinstance(use_tgrid, str) and use_tgrid=='auto':
    use_tgrid = IcD.use_tgrid

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  try:
    if cmap.startswith('cmo'):
      cmap = cmap.split('.')[-1]
      cmap = getattr(cmocean.cm, cmap)
    else:
      cmap = getattr(plt.cm, cmap)
  except:
    pass # assume that cmap is already a colormap

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
    ccrs_transform = None
  else:
    ccrs_proj = getattr(ccrs, projection)()
    ccrs_transform = ccrs.PlateCarree()

  # --- make axes and colorbar (taken from shade)
  if not do_write_data_range:
    dfigb = 0.0
  else:
    dfigb = 0.7
  if ax == 'auto':
      #fig, ax = plt.subplots(subplot_kw={'projection': ccrs_proj}) 
    hca, hcb = arrange_axes(1,1, plot_cb=True, asp=asp, fig_size_fac=fig_size_fac,
                                 projection=ccrs_proj,
                                 dfigb=dfigb,
                                )
    ax = hca[0]
    cax = hcb[0]

  # --- do plotting
  if use_tgrid:
    hm = shade(IcD.Tri, IaV.data, ax=ax, cax=cax, 
               clim=clim, cincr=cincr, cmap=cmap,
               clevs=clevs,
               contfs=contfs,
               conts=conts,
               contcolor=contcolor,
               contthick=contthick,
               contlw=contlw,
               #transform=ccrs_proj,
               transform=ccrs_transform,
               logplot=logplot,
               adjust_axlims=adjust_axlims,
              )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.clon.min(), IcD.clon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.clat.min(), IcD.clat.max()]
  else:
    hm = shade(IcD.lon, IcD.lat, IaV.data, ax=ax, cax=cax, 
               clim=clim, cincr=cincr, cmap=cmap,
               clevs=clevs,
               contfs=contfs,
               conts=conts,
               contcolor=contcolor,
               contthick=contthick,
               #transform=ccrs_proj,
               transform=ccrs_transform,
               logplot=logplot,
               adjust_axlims=adjust_axlims,
              )
    if isinstance(xlim, str) and (xlim=='auto'):
      xlim = [IcD.lon.min(), IcD.lon.max()]
    if isinstance(ylim, str) and (ylim=='auto'):
      ylim = [IcD.lat.min(), IcD.lat.max()]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  if xlim==[-180.,180.] and ylim==[-90.,90.]:
    template = 'global'
  else:
    template = 'none'
  if do_plot_settings:
    plot_settings(ax, template=template, land_facecolor=land_facecolor)

  if do_write_data_range:
    info_str = 'min: %.4g;        mean: %.4g;        std: %.4g;        max: %.4g' % (IaV.data.min(), IaV.data.mean(), IaV.data.std(), IaV.data.max())
    ax.text(0.5, -0.18, info_str, ha='center', va='top', transform=ax.transAxes)

  # --- saving data to netcdf 
  if save_data:
    if use_tgrid:
      print('::: Warning: saving variable on tripolar grid is not supported yet in hplot_base! :::')
    ds = write_dataarray_to_nc(
      fpath=fpath_nc,
      data=IaV.data,
      coords={'lat': IcD.lat, 'lon': IcD.lon},
      name=IaV.name, long_name=IaV.long_name, units=IaV.units,
      long_name_coords=['latitude', 'longitude'],
      #time_bnds=IaV.bnds,
    )

  #if (projection!='none') and (crs_features):
  ##if projection=='PlateCarree':
  ##if False:
  #  ax.coastlines()
  #  ax.add_feature(cartopy.feature.LAND, zorder=0, facecolor='0.9')
  #  ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7), crs=ccrs_proj)
  #  ax.set_yticks(np.linspace(np.round(ylim[0]),np.round(ylim[1]),7), crs=ccrs_proj)
  #  lon_formatter = LongitudeFormatter()
  #  lat_formatter = LatitudeFormatter()
  #  ax.xaxis.set_major_formatter(lon_formatter)
  #  ax.yaxis.set_major_formatter(lat_formatter)
  #  #ax.stock_img()
  #ax.xaxis.set_ticks_position('both')
  #ax.yaxis.set_ticks_position('both')
  return ax, cax, hm, Dstr

def vplot_base(IcD, IaV, clim='auto', cmap='viridis', cincr=-1.,
               clevs=None,
               contfs=None,
               conts=None,
               contcolor='k',
               contthick=0.,
               contlw=1.,
               ax='auto', cax=0,
               title='auto', xlabel='', ylabel='',
               xlim='auto', ylim='auto',
               xvar='lat',
               log2vax=False,
               vertaxtype='linear',
               daxl=1.8,
               logplot=False,
               asp=0.5,
               fig_size_fac=2.0,
               do_plot_settings=True,
               do_write_data_range=False,
               save_data=False,
               fpath_nc='',
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
    * hm 
  """
  Dstr = dict()

  # --- for backward compatibility
  if log2vax:
    vertaxtype = 'log2'

  # --- color limits and color map
  if isinstance(clim,str) and clim=='auto':
    clim = [IaV.data.min(), IaV.data.max()]

  # --- colormaps 
  try:
    if cmap.startswith('cmo'):
      cmap = cmap.split('.')[-1]
      cmap = getattr(cmocean.cm, cmap)
    else:
      cmap = getattr(plt.cm, cmap)
  except:
    pass # assume that cmap is already a colormap

  # --- annotations (title etc.) 
  if title=='auto':
    if not logplot:
      title = IaV.long_name+' ['+IaV.units+']'
    else:
      title = 'log$_{10}$('+IaV.long_name+') ['+IaV.units+']'

  # --- make axes and colorbar (taken from shade)
  if not do_write_data_range:
    dfigb = 0.0
  else:
    dfigb = 0.7
  if ax == 'auto':
    # (daxl needed to be increase for quickplots atm log v-axis)
    hca, hcb = arrange_axes(1,1, plot_cb=True, asp=asp, fig_size_fac=fig_size_fac, dfigb=dfigb, daxl=daxl,
                           )
    ax = hca[0]
    cax = hcb[0]

  nz = IaV.data.shape[0]

  # --- horizontal axes
  if xvar=='lon':
    x = IaV.lon_sec
    xstr = 'longitude'
  elif xvar=='lat':
    x = IaV.lat_sec
    xstr = 'latitude'
  elif xvar=='dist':
    x = IaV.dist_sec/1e3
    xstr = 'distance [km]'

  # --- vertical axes
  if IcD.model_type=='oce':
    if nz==IcD.depthc.size:
      z = IcD.depthc
    else: 
      z = IcD.depthi
    ylabel = 'depth [m]'
  elif IcD.model_type=='atm':
    if vertaxtype=='linear':
      z = IcD.plevc/100.
    elif vertaxtype=='log10':
      z = IcD.plev_log/100.
    ylabel = 'pressure [hPa]'

  if vertaxtype=='log2':
    z = np.log(z)/np.log(2) 
  elif vertaxtype=='log10':
    z = np.log(z)/np.log(10) 

  # --- do plotting
  hm = shade(x, z, IaV.data, ax=ax, cax=cax, 
             clim=clim, cincr=cincr, cmap=cmap,
             clevs=clevs,
             contfs=contfs,
             conts=conts,
             contcolor=contcolor,
             contthick=contthick,
             contlw=contlw,
             logplot=logplot,
            )
  if isinstance(xlim, str) and (xlim=='auto'):
    xlim = [x.min(), x.max()]
  if isinstance(ylim, str) and (ylim=='auto'):
    ylim = [z.max(), z.min()]

  # --- plot refinement
  ax.set_title(title)
  ax.set_xlabel(xstr)
  ax.set_ylabel(ylabel)
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)
  ax.set_xticks(np.linspace(np.round(xlim[0]),np.round(xlim[1]),7))
  if vertaxtype=='linear':
    if IcD.model_type=='oce':
      ax.set_yticks(np.arange(0,6500,1000.))
    elif IcD.model_type=='atm':
      ax.set_yticks(np.arange(0,1100,100.))
  elif vertaxtype=='log2':
    ax.set_yticklabels(2**ax.get_yticks())
  elif vertaxtype=='log10':
    if IcD.model_type=='atm':
      yticks = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50., 100., 200., 500., 1000.])
      ax.set_yticks(np.log10(yticks))
      ax.set_yticklabels(yticks)
  ax.set_facecolor('0.8')
  ax.xaxis.set_ticks_position('both')
  ax.yaxis.set_ticks_position('both')

  if do_write_data_range:
    info_str = 'min: %.4g;        mean: %.4g;        std: %.4g;        max: %.4g' % (IaV.data.min(), IaV.data.mean(), IaV.data.std(), IaV.data.max())
    ax.text(0.5, -0.18, info_str, ha='center', va='top', transform=ax.transAxes)

  # --- saving data to netcdf 
  if save_data:
    ds = write_dataarray_to_nc(
      fpath=fpath_nc,
      data=IaV.data,
      coords={'vert_coord': z, 'hor_coord': x},
      name=IaV.name, long_name=IaV.long_name, units=IaV.units,
      long_name_coords=[ylabel, xstr],
      #time_bnds=IaV.bnds,
    )

  return ax, cax, hm, Dstr

def calc_conts(conts, clim, cincr, nclev):
  # ------ decide how to determine contour levels
  if isinstance(conts, np.ndarray) or isinstance(conts, list):
    # use given contours 
    conts = np.array(conts)
  else:
    # calculate contours
    # ------ decide whether contours should be calculated by cincr or nclev
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
              clevs=None,
              contfs=None,
              conts=None,
              nclev='auto',
              #cint='auto', # old: use cincr now
              contcolor='k',
              contthick=0.,
              contlw=1.,
              use_pcol=True,
              use_pcol_or_contf=True,
              cbticks='auto',
              cbtitle='',
              cbdrawedges='auto',
              #cborientation='vertical',
              cborientation='auto',
              adjust_axlims=True,
              bmp=None,
              transform=None,
              projection=None,
              logplot=False,
              edgecolor='none',
           ):
    """ Convenient wrapper around pcolormesh, contourf, contour and their triangular versions.
    """
    # --- decide whether regular or triangular plots should be made
    if isinstance(datai, str) and datai=='auto':
      Tri = x
      datai = y
      rectangular_grid = False
    else:
      rectangular_grid = True

    if projection is not None:
      transform = projection


    # --- decide whether pcolormesh or contourf plot
    if use_pcol_or_contf:
      if contfs is None:
        use_pcol = True
        use_contf = False
      else:
        use_pcol = False
        use_contf = True
    else:
        use_pcol = False
        use_contf = False
    #if use_pcol and use_contf:
    #  raise ValueError('::: Error: Only one of use_pcol or use_contf can be True. :::')

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
  
    if use_pcol:
      # --- norm
      if cincr>0.:
        clevs = np.arange(clim[0], clim[1]+cincr, cincr)
        use_norm = True
      elif use_pcol and clevs is not None:
        clevs = np.array(clevs)
        use_norm = True
      else:
        norm = None
        use_norm = False
    elif use_contf:
      contfs = calc_conts(contfs, clim, cincr, nclev)
      clevs = contfs
      use_norm = True
    else:
      use_norm = False
        
    if use_norm:
      #norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
      nlev = clevs.size
      # --- expanded norm and cmap
      norm_e = matplotlib.colors.BoundaryNorm(boundaries=np.arange(0,nlev+2,1), ncolors=cmap.N)
      cmap_e = matplotlib.colors.ListedColormap(cmap(norm_e(np.arange(0,nlev+1,1))))
      # --- actuall cmap with over and under values
      cmap = matplotlib.colors.ListedColormap(cmap(norm_e(np.arange(1,nlev,1))))        
      norm = matplotlib.colors.BoundaryNorm(boundaries=clevs, ncolors=cmap.N)
      cmap.set_under(cmap_e(norm_e(0)))
      cmap.set_over(cmap_e(norm_e(nlev)))
      vmin = None
      vmax = None
    else:
      vmin = clim[0]
      vmax = clim[1]
  
    # --- decide whether to use extra contour lines
    if conts is None:
      use_cont = False
    else:
      use_cont = True
      conts = calc_conts(conts, clim, cincr, nclev)
  
    # --- decide whether there should be black edges at colorbar
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
                           shading='auto',
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
# This line causes problems with cartopy and contourfs. The plot seems to be unvisible.
#          cl.set_zorder(zorder - 1)
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
      # ------ determine cborientation
      if cborientation=='auto':
        if cax.get_xticks().size==0:
          cborientation = 'vertical'
        else:
          cborientation = 'horizontal'
      # ------ make actual colorbar
      cb = plt.colorbar(mappable=hm, cax=cax, orientation=cborientation)
      # ------ prevent white lines if fig is saved as pdf
      cb.solids.set_edgecolor("face")
      # ------ use exponential notation for large colorbar ticks
      cb.formatter.set_powerlimits((-3, 3))
      # ------ colorbar ticks
      if isinstance(cbticks, np.ndarray) or isinstance(cbticks, list):
        cb.set_ticks(cbticks)
      else:
        if use_norm:
          dcl = np.diff(clevs)
          if (np.isclose(dcl, dcl[0])).all(): 
            cb.set_ticks(clevs[::2])
          else:
            cb.set_ticks(clevs)
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
    cb = plt.colorbar(mappable=hm[0], cax=cax, extend=extend)
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

def arrange_axes_old( nx,ny,
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

def arrange_axes(nx,ny,
                 sharex = True,
                 sharey = False,
                 xlabel = '',
                 ylabel = '',
                 # labeling axes with e.g. (a), (b), (c)
                 do_axes_labels = True,
                 axlab_kw = dict(),
                 # colorbar
                 plot_cb = True,
                 # projection (e.g. for cartopy)
                 projection = None,
                 # aspect ratio of axes
                 asp = 1.,
                 sasp = 0.,  # for compability with older version of arrange_axes
                 # width and height of axes
                 wax = 'auto',
                 hax = 4.,
                 # extra figure spaces (left, right, top, bottom)
                 dfigl = 0.0,
                 dfigr = 0.0,
                 dfigt = 0.0,
                 dfigb = 0.0,
                 # space aroung axes (left, right, top, bottom) 
                 daxl = 1.8, # reset to zero if sharex==False
                 daxr = 0.8,
                 daxt = 0.8,
                 daxb = 1.2, # reset to zero if sharex==True
                 # space around colorbars (left, right, top, bottom) 
                 dcbl = -0.5,
                 dcbr = 1.4,
                 dcbt = 0.0,
                 dcbb = 0.5,
                 # width and height of colorbars
                 wcb = 0.5,
                 hcb = 'auto',
                 # factors to increase widths and heights of axes and colorbars
                 fig_size_fac = 1.,
                 f_wax = 1.,
                 f_hax = 1.,
                 f_wcb = 1.,
                 f_hcb = 1.,
                 # factors to increase spaces (figure)
                 f_dfigl = 1.,
                 f_dfigr = 1.,
                 f_dfigt = 1.,
                 f_dfigb = 1.,
                 # factors to increase spaces (axes)
                 f_daxl = 1.,
                 f_daxr = 1.,
                 f_daxt = 1.,
                 f_daxb = 1.,
                 # factors to increase spaces (colorbars)
                 f_dcbl = 1.,
                 f_dcbr = 1.,
                 f_dcbt = 1.,
                 f_dcbb = 1.,
                 # font sizes of labels, titles, ticks
                 fs_label = 10.,
                 fs_title = 12.,
                 fs_ticks = 10.,
                 # font size increasing factor
                 f_fs = 1,
                ):

  # factor to convert cm into inch
  cm2inch = 0.3937

  if sasp!=0:
    print('::: Warning: You are using keyword ``sasp`` for setting the aspect ratio but you should switch to use ``asp`` instead.:::')
    asp = 1.*sasp

  # --- set hcb in case it is auto
  if isinstance(wax, str) and wax=='auto':
    wax = hax/asp

  # --- set hcb in case it is auto
  if isinstance(hcb, str) and hcb=='auto':
    hcb = hax

  # --- rename horizontal->bottom and vertical->right
  if isinstance(plot_cb, str) and plot_cb=='horizontal':
    plot_cb = 'bottom'
  if isinstance(plot_cb, str) and plot_cb=='vertical':
    plot_cb = 'right'
  
  # --- apply fig_size_fac
  # font sizes
  #f_fs *= fig_size_fac
  # factors to increase widths and heights of axes and colorbars
  f_wax *= fig_size_fac
  f_hax *= fig_size_fac
  #f_wcb *= fig_size_fac
  f_hcb *= fig_size_fac
  ## factors to increase spaces (figure)
  #f_dfigl *= fig_size_fac
  #f_dfigr *= fig_size_fac
  #f_dfigt *= fig_size_fac
  #f_dfigb *= fig_size_fac
  ## factors to increase spaces (axes)
  #f_daxl *= fig_size_fac
  #f_daxr *= fig_size_fac
  #f_daxt *= fig_size_fac
  #f_daxb *= fig_size_fac
  ## factors to increase spaces (colorbars)
  #f_dcbl *= fig_size_fac
  #f_dcbr *= fig_size_fac
  #f_dcbt *= fig_size_fac
  #f_dcbb *= fig_size_fac
  
  # --- apply font size factor
  fs_label *= f_fs
  fs_title *= f_fs
  fs_ticks *= f_fs

  # make vector of plot_cb if it has been true or false before
  # plot_cb can have values [{1}, 0] 
  # with meanings:
  #   1: plot cb; 
  #   0: do not plot cb
  plot_cb_right  = False
  plot_cb_bottom = False
  if isinstance(plot_cb, bool) and (plot_cb==True):
    plot_cb = np.ones((nx,ny))  
  elif isinstance(plot_cb, bool) and (plot_cb==False):
    plot_cb = np.zeros((nx,ny))
  elif isinstance(plot_cb, str) and plot_cb=='right':
    plot_cb = np.zeros((nx,ny))
    plot_cb_right = True
  elif isinstance(plot_cb, str) and plot_cb=='bottom':
    plot_cb = np.zeros((nx,ny))
    plot_cb_bottom = True
  else:
    plot_cb = np.array(plot_cb)
    if plot_cb.size!=nx*ny:
      raise ValueError('Vector plot_cb has wrong length!')
    if plot_cb.shape[0]==nx*ny:
      plot_cb = plot_cb.reshape(ny,nx).transpose()
    elif plot_cb.shape[0]==ny:
      plot_cb = plot_cb.transpose()
  
  # --- make list of projections if it is not a list
  if not isinstance(projection, list):
    projection = [projection]*nx*ny
  
  # --- make arrays and multiply by f_*
  daxl = np.array([daxl]*nx)*f_daxl
  daxr = np.array([daxr]*nx)*f_daxr
  dcbl = np.array([dcbl]*nx)*f_dcbl
  dcbr = np.array([dcbr]*nx)*f_dcbr
  
  wax = np.array([wax]*nx)*f_wax
  wcb = np.array([wcb]*nx)*f_wcb
  
  daxt = np.array([daxt]*ny)*f_daxt
  daxb = np.array([daxb]*ny)*f_daxb
  dcbt = np.array([dcbt]*ny)*f_dcbt
  dcbb = np.array([dcbb]*ny)*f_dcbb
  
  hax = np.array([hax]*ny)*f_hax
  hcb = np.array([hcb]*ny)*f_hcb
  
  # --- adjust for shared axes
  if sharex:
    daxb[:-1] = 0.
  
  if sharey:
    daxl[1:] = 0.

  # --- adjust for one colorbar at the right or bottom
  if plot_cb_right:
    daxr_s = daxr[0]
    dcbl_s = dcbl[0]
    dcbr_s = dcbr[0]
    wcb_s  = wcb[0]
    hcb_s  = hcb[0]
    dfigr += dcbl_s+wcb_s+0.*dcbr_s+daxl[0]
  if plot_cb_bottom:
    hcb_s  = wcb[0]
    wcb_s  = wax[0]
    dcbb_s = dcbb[0]+daxb[-1]
    dcbt_s = dcbt[0]
    #hcb_s  = hcb[0]
    dfigb += dcbb_s+hcb_s+dcbt_s
  
  # --- adjust for columns without colorbar
  delete_cb_space = plot_cb.sum(axis=1)==0
  dcbl[delete_cb_space] = 0.0
  dcbr[delete_cb_space] = 0.0
  wcb[delete_cb_space]  = 0.0
  
  # --- determine ax position and fig dimensions
  x0 =   dfigl
  y0 = -(dfigt)
  
  pos_axcm = np.zeros((nx*ny,4))
  pos_cbcm = np.zeros((nx*ny,4))
  nn = -1
  y00 = y0
  x00 = x0
  for jj in range(ny):
    y0   += -(daxt[jj]+hax[jj])
    x0 = x00
    for ii in range(nx):
      nn += 1
      x0   += daxl[ii]
      pos_axcm[nn,:] = [x0, y0, wax[ii], hax[jj]]
      pos_cbcm[nn,:] = [x0+wax[ii]+daxr[ii]+dcbl[ii], y0, wcb[ii], hcb[jj]]
      x0   += wax[ii]+daxr[ii]+dcbl[ii]+wcb[ii]+dcbr[ii]
    y0   += -(daxb[jj])
  wfig = x0+dfigr
  hfig = y0-dfigb
  
  # --- transform from negative y axis to positive y axis
  hfig = -hfig
  pos_axcm[:,1] += hfig
  pos_cbcm[:,1] += hfig
  
  # --- convert to fig coords
  cm2fig_x = 1./wfig
  cm2fig_y = 1./hfig
  
  pos_ax = 1.*pos_axcm
  pos_cb = 1.*pos_cbcm
  
  pos_ax[:,0] = pos_axcm[:,0]*cm2fig_x
  pos_ax[:,2] = pos_axcm[:,2]*cm2fig_x
  pos_ax[:,1] = pos_axcm[:,1]*cm2fig_y
  pos_ax[:,3] = pos_axcm[:,3]*cm2fig_y
  
  pos_cb[:,0] = pos_cbcm[:,0]*cm2fig_x
  pos_cb[:,2] = pos_cbcm[:,2]*cm2fig_x
  pos_cb[:,1] = pos_cbcm[:,1]*cm2fig_y
  pos_cb[:,3] = pos_cbcm[:,3]*cm2fig_y

  # --- find axes center (!= figure center)
  x_ax_cent = pos_axcm[0,0] +0.5*(pos_axcm[-1,0]+pos_axcm[-1,2]-pos_axcm[0,0])
  y_ax_cent = pos_axcm[-1,1]+0.5*(pos_axcm[0,1] +pos_axcm[0,3] -pos_axcm[-1,1])
  
  # --- make figure and axes
  fig = plt.figure(figsize=(wfig*cm2inch, hfig*cm2inch))
  
  hca = [0]*(nx*ny)
  hcb = [0]*(nx*ny)
  nn = -1
  for jj in range(ny):
    for ii in range(nx):
      nn+=1
  
      # --- axes
      hca[nn] = fig.add_subplot(position=pos_ax[nn,:], projection=projection[nn])
      hca[nn].set_position(pos_ax[nn,:])
  
      # --- colorbar
      if plot_cb[ii,jj] == 1:
        hcb[nn] = fig.add_subplot(position=pos_cb[nn,:])
        hcb[nn].set_position(pos_cb[nn,:])
      ax  = hca[nn]
      cax = hcb[nn] 
  
      # --- label
      ax.set_xlabel(xlabel, fontsize=fs_label)
      ax.set_ylabel(ylabel, fontsize=fs_label)
      #ax.set_title('', fontsize=fs_title)
      matplotlib.rcParams['axes.titlesize'] = fs_title
      ax.tick_params(labelsize=fs_ticks)
      if plot_cb[ii,jj] == 1:
        hcb[nn].tick_params(labelsize=fs_ticks)
  
      #ax.tick_params(pad=-10.0)
      #ax.xaxis.labelpad = 0
      #ax._set_title_offset_trans(float(-20))
  
      # --- axes ticks
      # delete labels for shared axes
      if sharex and jj!=ny-1:
        hca[nn].ticklabel_format(axis='x',style='plain',useOffset=False)
        hca[nn].tick_params(labelbottom=False)
        hca[nn].set_xlabel('')
  
      if sharey and ii!=0:
        hca[nn].ticklabel_format(axis='y',style='plain',useOffset=False)
        hca[nn].tick_params(labelleft=False)
        hca[nn].set_ylabel('')
  
      # ticks for colorbar 
      if plot_cb[ii,jj] == 1:
        hcb[nn].set_xticks([])
        hcb[nn].yaxis.tick_right()
        hcb[nn].yaxis.set_label_position("right")

  #--- needs to converted to fig coords (not cm)
  if plot_cb_right:
    nn = -1
    #pos_cb = np.array([(wfig-(dfigr+dcbr_s+wcb_s))*cm2fig_x, (y_ax_cent-0.5*hcb_s)*cm2fig_y, wcb_s*cm2fig_x, hcb_s*cm2fig_y])
    pos_cb = np.array([ (pos_axcm[-1,0]+pos_axcm[-1,2]+daxr_s+dcbl_s)*cm2fig_x, 
                        (y_ax_cent-0.5*hcb_s)*cm2fig_y, 
                        (wcb_s)*cm2fig_x, 
                        (hcb_s)*cm2fig_y 
                      ])
    hcb[nn] = fig.add_subplot(position=pos_cb)
    hcb[nn].tick_params(labelsize=fs_ticks)
    hcb[nn].set_position(pos_cb)
    hcb[nn].set_xticks([])
    hcb[nn].yaxis.tick_right()
    hcb[nn].yaxis.set_label_position("right")

  if plot_cb_bottom:
    nn = -1
    pos_cb = np.array([ (x_ax_cent-0.5*wcb_s)*cm2fig_x, 
                        (dcbb_s)*cm2fig_y, 
                        (wcb_s)*cm2fig_x, 
                        (hcb_s)*cm2fig_y
                      ])
    hcb[nn] = fig.add_subplot(position=pos_cb)
    hcb[nn].set_position(pos_cb)
    hcb[nn].tick_params(labelsize=fs_ticks)
    hcb[nn].set_yticks([])

  # add letters for subplots
  if (do_axes_labels) and (axlab_kw is not None):
    hca = axlab(hca, fontdict=axlab_kw)

  return hca, hcb

# ================================================================================ 
def axlab(hca, figstr=[], posx=[-0.00], posy=[1.05], fontdict=None):
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
    #lett = "abcdefghijklmnopqrstuvwxyz"
    lett  = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
    lett += ["a2","b2","c2","d2","e2","f2","g2","h2","i2","j2","k2","l2","m2","n2","o2","p2","q2","r2","s2","t2","u2","v2","w2","x2","y2","z2"]
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
#  for nn, ax in enumerate(hca):
#    #ax.set_title(figstr[nn]+'\n', loc='left', fontsize=10)
#    ax.set_title(figstr[nn], loc='left', fontsize=10)
  return hca

def plot_settings(ax, xlim='none', ylim='none', xticks='auto', yticks='auto', xlocs=None, ylocs=None,
                     ticks_position='both', template='none', 
                     x_minor_tick_diff='auto', y_minor_tick_diff='auto',
                     # cartopy specific settings
                     projection=None,  # not necessary
                     do_xylim=True,
                     do_xyticks=True,
                     do_xyminorticks=True,
                     do_gridlines=False,
                     coastlines_color='k', coastlines_resolution='110m',
                     land_zorder=2, land_facecolor='0.7'):

  # --- templates
  if template=='global':
    xlim = [-180,180]
    ylim = [-90,90]
    xticks = np.arange(-120,121,60.)
    yticks = np.arange(-60,61,30.)
    xlocs = np.arange(-180,181,60.)
    ylocs = np.arange(-90,91,30.)
    x_minor_tick_diff = 20.
    y_minor_tick_diff = 10.
  elif template=='na':
    xlim = [-80,0]
    ylim = [30,70]
  elif template=='sonett':
    xlim=[-20,30]
    ylim=[-45,-20]
    xticks = np.arange(-10,30,10)
    yticks = np.arange(-40,-20,5,)
    #xlocs = np.arange(-180,181,60.)
    #ylocs = np.arange(-90,91,30.)
    x_minor_tick_diff = 1.
    y_minor_tick_diff = 1.
  elif template=='labsea':
    pass
  elif template=='zlat':
    pass
  elif template=='zlat_noso':
    pass
  elif template=='none':
    pass
  else:
    raise ValueError('::: Error: Uknown template %s'%template)

  if isinstance(xlim,str) and xlim=='none':
    xlim = ax.get_xlim()
  if isinstance(ylim,str) and ylim=='none':
    ylim = ax.get_ylim()
  if isinstance(xticks,str) and xticks=='auto':
    xticks = np.linspace(xlim[0],xlim[1],5) 
  if isinstance(yticks,str) and yticks=='auto':
    yticks = np.linspace(ylim[0],ylim[1],5) 
  if (isinstance(x_minor_tick_diff,str) and x_minor_tick_diff=='auto'):
    x_minor_tick_diff = (xticks[1]-xticks[0])/5.
    #xminorticks = np.linspace(xlim[0], xlim[1], (xticks.size-1)*2+xticks.size)
  xminorticks = np.arange(xlim[0], xlim[1]+x_minor_tick_diff, x_minor_tick_diff)
  if (isinstance(y_minor_tick_diff,str) and y_minor_tick_diff=='auto'):
    y_minor_tick_diff = (yticks[1]-yticks[0])/5.
  yminorticks = np.arange(ylim[0], ylim[1]+y_minor_tick_diff, y_minor_tick_diff)

  # --- decide whether to use cartopy
  if isinstance(ax, cartopy.mpl.geoaxes.GeoAxesSubplot):
    use_cartopy = True
    projection = ax.projection
  else:
    use_cartopy = False
    projection = None

  # !!! This needs to be set before minor ticks are set
  if use_cartopy:
    #ax.xaxis.set_major_formatter(cartopy.mpl.ticker.LongitudeFormatter(degree_symbol=''))
    #ax.yaxis.set_major_formatter(cartopy.mpl.ticker.LatitudeFormatter(degree_symbol=''))
    ax.xaxis.set_major_formatter(cartopy.mpl.ticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cartopy.mpl.ticker.LatitudeFormatter())
  
  # --- xticks, yticks
  if do_xyticks and not use_cartopy:
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
  elif do_xyticks and use_cartopy:
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
  
  if do_xyticks and ticks_position=='both':
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
  
  # --- gridlines
  if do_gridlines and not use_cartopy:
    ax.grid(True)
  elif do_gridlines and use_cartopy:
    ax.gridlines(xlocs=xlocs, ylocs=ylocs)
    
  # --- minor ticks
  if do_xyminorticks:
    #ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(x_minor_tick_diff))
    ax.set_xticks(xminorticks, minor=True)
    #ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(y_minor_tick_diff))
    ax.set_yticks(yminorticks, minor=True)
        
  # --- xlim, ylim
  if do_xylim and not use_cartopy:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
  elif do_xylim and use_cartopy:
    #ax.set_extent([xlim[0],xlim[1],ylim[0],ylim[1]], crs=ccrs.PlateCarree())
    ax.set_extent([xlim[0],xlim[1],ylim[0],ylim[1]], crs=projection)

  # --- cartopy specific stuff
  if use_cartopy: 
    #for tick in ax.xaxis.get_ticklabels()+ax.yaxis.get_ticklabels():
    #  tick.set_fontsize(8)
    if isinstance(land_facecolor, str) and land_facecolor!='none':
      feature = cartopy.feature.LAND
      #feature = feature.with_scale(coastlines_resolution)
      ax.add_feature(feature, zorder=land_zorder, facecolor=land_facecolor)
    if isinstance(coastlines_color, str) and coastlines_color!='none':
      #ax.coastlines(color=coastlines_color, resolution=coastlines_resolution)
      feature = cartopy.feature.COASTLINE
      #feature = feature.with_scale(coastlines_resolution)
      ax.add_feature(feature, zorder=land_zorder, edgecolor=coastlines_color)
    #if template=='global':
    #  ax.set_global()
  return

class split_axes_vertically(object):
    def __init__(self, ax, frac=0.4, space_between_axs=0.05):
        """ Make two axes out of one by splitting that axes in the vertical.
        Input:
        frac: fraction of lower axes
        space_between_axs: space between both axes as fraction of total axes height

        return ax: New modified axes object.
        ax.ax1: Lower axes (new axes)
        ax.ax2: Upper axes (original axes)
        """
        ax2 = ax

        pos_ax = ax2.get_position()
        # get old values
        xo = pos_ax.x0
        yo = pos_ax.y0
        wo = pos_ax.width
        ho = pos_ax.height

        h1 = ho*frac-ho*space_between_axs/2.
        h2 = ho*(1-frac)-ho*space_between_axs/2.
        x1 = x2 = xo
        w1 = w2 = wo
        y1 = yo+h2+space_between_axs*ho
        y2 = yo

        ax2.set_position([x1, y1, w1, h1])

        ax1 = plt.axes(position=[x2, y2, w2, h2])

        if len(ax2.get_yticklabels())==0:
            ax1.tick_params(labelleft=False)

        if len(ax2.get_xticklabels())==0:
            ax1.tick_params(labelbottom=False)

        ax2.tick_params(labelbottom=False)

        ax1.set_xlabel(ax2.get_xlabel())
        ax2.set_xlabel('')
        
        self.ax1 = ax1
        self.ax2 = ax2
        
        self.axs = [self.ax1, self.ax2]
        
        self.frac = frac
        self.space_between_axs = space_between_axs
        return
    
    def set_xlabel(self, *args, **kwargs):
        h = self.ax1.set_xlabel(*args, **kwargs)
        return h
    
    def set_xlim(self, *args, **kwargs):
        ht = self.ax1.set_xlim(*args, **kwargs)
        ht = self.ax2.set_xlim(*args, **kwargs)
        return ht
    
    def set_xticks(self, *args, **kwargs):
        ht = self.ax1.set_xticks(*args, **kwargs)
        ht = self.ax2.set_xticks(*args, **kwargs)
        return ht

    def set_ylabel(self, *args, x=None, y=None, transform=None, **kwargs):
        h1 = self.ax1.set_ylabel(*args, **kwargs)
#         h2 = self.ax2.set_ylabel(*args, **kwargs)
        if x is None:
            x = -0.15
        if y is None:
            y = 1. + 0.5*self.space_between_axs
        self.ax1.yaxis.set_label_coords(x, y, transform)
        return
    
    def set_ylim(self, ylim1, ylim2, *args, **kwargs):
        h1 = self.ax1.set_ylim(ylim1, *args, **kwargs)
        h2 = self.ax2.set_ylim(ylim2, *args, **kwargs)
        return h1, h2
    
    def set_yticks(self, yticks1, yticks2, *args, **kwargs):
        h1 = self.ax1.set_yticks(yticks1, *args, **kwargs)
        h2 = self.ax2.set_yticks(yticks2, *args, **kwargs)
        return h1, h2
    
    def grid(self, *args, **kwargs):
        h1 = self.ax1.grid(*args, **kwargs)
        h2 = self.ax2.grid(*args, **kwargs)
        return h1, h2
    
    def set_title(self, *args, **kwargs):
        ht = self.ax2.set_title(*args, **kwargs)
        return ht
    
    def set_facecolor(self, *args, **kwargs):
        self.ax1.set_facecolor(*args, **kwargs)
        self.ax2.set_facecolor(*args, **kwargs)
        return
