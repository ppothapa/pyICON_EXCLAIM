import sys, glob, os
import datetime
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
from scipy.spatial import cKDTree
# --- plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
#import my_toolbox as my
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean

def icon2regular(data, clon, clat, lons, lats, distances=None, \
                  inds=None, radius_of_influence=100000):
  """
  * example usage:

  * credits
    function taken (and slightly modified) from pyfesom (Nikolay Koldunov)
  """
  if distances.ndim == 1:
      #distances_ma = np.ma.masked_greater(distances, radius_of_influence)
      data_interpolated = data[inds]

      data_interpolated[distances>=radius_of_influence] = np.nan
      
      data_interpolated = data_interpolated.reshape(lons.shape)
      data_interpolated = np.ma.masked_invalid(data_interpolated)
  else:
      distances_ma = np.ma.masked_greater(distances, radius_of_influence)
      
      w = 1.0 / distances_ma**2
      data_interpolated = np.ma.sum(w * data[inds], axis=1) / np.ma.sum(w, axis=1)
      data_interpolated.shape = lons.shape
      data_interpolated = np.ma.masked_invalid(data_interpolated)
  return data_interpolated

def calc_ckdtree(lon_reg, lat_reg, res, 
                 fpath_grid_triangular='', 
                 fpath_grid_rectangular='',
                 ):
  """
  * example usage:
    calc_ckdtree(path_grid='/mnt/lustre01/work/mh0033/m300602/icon/rect_grids/',
                 fname_grid='ocean_r2b9.npz')
  """
  # --- load triangular grid
  f = Dataset(fpath_grid_triangular, 'r')
  clon = f.variables['clon'][:] * 180./np.pi
  clat = f.variables['clat'][:] * 180./np.pi
  f.close()

  # --- make rectangular grid 
  lon = np.arange(lon_reg[0],lon_reg[1],res)
  lat = np.arange(lat_reg[0],lat_reg[1],res)
  Lon, Lat = np.meshgrid(lon, lat)

  # --- initialize timing
  tims = np.array([0])
  tims = timing(tims)

  # --- do ckdtree
  tims = timing(tims, 'CKD: define reg grid')
  lzip = list(zip(clon, clat))
  tims = timing(tims, 'CKD: zip orig grid')
  tree = cKDTree(lzip)
  tims = timing(tims, 'CKD: CKDgrid')
  lzip_rg = list(zip(Lon.flatten(), Lat.flatten()))
  tims = timing(tims, 'CKD: zip reg grid')
  dckdtree, ickdtree = tree.query(lzip_rg , k=1, n_jobs=1)
  tims = timing(tims, 'CKD: tree query')

  # --- save grid
  np.savez(fpath_grid_rectangular,
            dckdtree=dckdtree,
            ickdtree=ickdtree,
            lon=lon,
            lat=lat,
           )
  return

def timing(ts, string=''):
  if ts[0]==0:
    ts = np.array([datetime.datetime.now()])
  else:
    ts = np.append(ts, [datetime.datetime.now()])
    print(ts[-1]-ts[-2]), ' ', (ts[-1]-ts[0]), ' '+string
  return ts

def conv_gname(gname):
  gname = gname[:-4]

  ogrid = gname.split('_')[0]
  res = float(gname.split('_')[1][1:])

  lo1 = gname.split('_')[2]
  if lo1[-1]=='w':
    lo1 = -float(lo1[:-1])
  else:
    lo1 = float(lo1[:-1])
  lo2 = gname.split('_')[3]
  if lo2[-1]=='w':
    lo2 = -float(lo2[:-1])
  else:
    lo2 = float(lo2[:-1])

  la1 = gname.split('_')[4]
  if la1[-1]=='s':
    la1 = -float(la1[:-1])
  else:
    la1 = float(la1[:-1])
  la2 = gname.split('_')[5]
  if la2[-1]=='s':
    la2 = -float(la2[:-1])
  else:
    la2 = float(la2[:-1])

  lon_reg = [lo1, lo2]
  lat_reg = [la1, la2]
  return ogrid, res, lon_reg, lat_reg

class IconData(object):
  def __init__(self, 
               fpath_grid_triangular="", 
               path_grid_rectangular="/mnt/lustre01/work/mh0033/m300602/icon/rect_grids/", 
               fname_rgrid="",
               path_data="",
               search_str="",
               region="global",
               lon_reg=[-80, -75],
               lat_reg=[16, 18],
               use_tgrid=False,
              ):
    self.Drgrids = dict()
    self.Drgrids["global"]    = "r2b9ocean_r0.3_180w_180e_90s_90n.npz"
    self.Drgrids["hurricane"] = "r2b9ocean_r0.05_100w_30w_2n_40n.npz"
    self.fpath_grid_triangular = fpath_grid_triangular
    self.path_grid_rectangular = path_grid_rectangular
    if fname_rgrid=="":
      self.fname_rgrid = self.Drgrids[region]
    else:
      self.fname_rgrid = fname_rgrid
    self.path_data = path_data
    self.interpolate = True
    self.units=dict()
    self.long_name=dict()
    self.data=dict()

    self.lon_reg = lon_reg
    self.lat_reg = lat_reg
    self.use_tgrid = use_tgrid

    self.load_grid()
    self.get_timesteps(search_str)
    self.get_varnames(self.flist[0])
    return

  def get_timesteps(self, search_str):
    flist = np.array(glob.glob(self.path_data+search_str))
    flist.sort()
    
    times = np.zeros(flist.size, dtype='datetime64[s]')
    for l, fpath in enumerate(flist):
      tstr = fpath.split('/')[-1].split('_')[-1][:-4]
      times[l] = '%s-%s-%sT%s:%s:%s' % ( (tstr[:4], tstr[4:6], tstr[6:8], 
                                          tstr[9:11], tstr[11:13], tstr[13:15]))
    self.times=times
    self.flist=flist
    return 

  def get_varnames(self, fpath):
    f = Dataset(fpath, 'r')
    varnames = f.variables.keys()
    f.close()
    varnames = [var for var in varnames if not var.startswith('clon')]
    varnames = [var for var in varnames if not var.startswith('clat')]
    varnames = [var for var in varnames if not var.startswith('elon')]
    varnames = [var for var in varnames if not var.startswith('elat')]
    varnames = [var for var in varnames if not var.startswith('time')]
    varnames = [var for var in varnames if not var.startswith('depth')]
    varnames = [var for var in varnames if not var.startswith('lev')]
    self.varnames = varnames
    return

  def load_grid(self, lon_reg='all', lat_reg='all'):
    # --- triangle grid
    f = Dataset(self.fpath_grid_triangular, 'r')
    self.clon = f.variables['clon'][:] * 180./np.pi
    self.clat = f.variables['clat'][:] * 180./np.pi
    self.depth = f.variables['depth'][:]
    f.close()
    
    if self.use_tgrid:
      # --- triangle grid
      self.ind_reg = np.where( 
          (self.clon >  self.lon_reg[0]) 
        & (self.clon <= self.lon_reg[1]) 
        & (self.clat >  self.lat_reg[0]) 
        & (self.clat <= self.lat_reg[1]) )[0]
      self.clon = self.clon[self.ind_reg]
      self.clat = self.clat[self.ind_reg]

      # triangulation
      ntr = self.clon.size
      f = Dataset(self.fpath_grid_triangular, 'r')
      clon_bnds = f.variables['clon_bnds'][:] * 180./np.pi
      clat_bnds = f.variables['clat_bnds'][:] * 180./np.pi
      clon_bnds = clon_bnds[self.ind_reg,:]
      clat_bnds = clat_bnds[self.ind_reg,:]
      f.close()

      clon_bnds_rs = clon_bnds.reshape(ntr*3)
      clat_bnds_rs = clat_bnds.reshape(ntr*3)
      triangles = np.arange(ntr*3).reshape(ntr,3)
      self.Tri = matplotlib.tri.Triangulation(
        clon_bnds_rs, clat_bnds_rs, triangles=triangles)

      mask_grid = (   (clon_bnds[:,0]-clon_bnds[:,1])**2 
                    + (clon_bnds[:,0]-clon_bnds[:,2])**2 
                    + (clat_bnds[:,0]-clat_bnds[:,1])**2 
                    + (clat_bnds[:,0]-clat_bnds[:,2])**2 
                  ) > 2.*180./np.pi
      self. maskTri = mask_grid
      self.Tri.set_mask(self.maskTri)
    else:
      # --- rectangular grid
      ddnpz = np.load(self.path_grid_rectangular+self.fname_rgrid)
      for var in ddnpz.keys():
        exec('self.%s = ddnpz[var]' % var)
      self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
    return

  def load_hsnap(self, varnames, step_snap=0, it=0, iz=0):
    self.step_snap = step_snap
    self.it = it
    self.iz = iz
    fpath = self.flist[step_snap]
    #print("Using data set %s" % fpath)
    f = Dataset(fpath, 'r')
    for var in varnames:
      print("Loading %s" % (var))
      if f.variables[var].ndim==2:
        data = f.variables[var][it,:]
      else:
        data = f.variables[var][it,iz,:]
      self.long_name[var] = f.variables[var].long_name
      self.units[var] = f.variables[var].units
      self.data[var] = var

      #if self.interpolate:
      if self.use_tgrid:
        data = data[self.ind_reg] 
      else:
        data = icon2regular(data, self.clon, self.clat, self.Lon, self.Lat, 
                            distances=self.dckdtree, inds=self.ickdtree)

      # add data to IconData object
      setattr(self, var, data)
    f.close()
    return

  def load_sec(self, varnames, fpath, isec, ksec):
    pass
    return

class IP_hor_sec_rect(object):
  def __init__(self, 
               IcD, ax='', cax='',
               var='', clim='auto', nc=1, cmap='viridis',
               transform=None, lon_reg='auto', lat_reg='auto',
               title='auto',
               time_string='auto',
               depth_string='auto',
               edgecolor='none',
               ):
    self.ax=ax
    self.cax=cax
    self.var=var

    data = getattr(IcD, var)
    if IcD.use_tgrid:
      self.hpc = trishade(IcD.Tri, 
                           data, ax=ax, cax=cax, 
                           clim=clim, cmap=cmap, transform=transform,
                           edgecolor=edgecolor,
                            )
      lon_reg = IcD.lon_reg
      lat_reg = IcD.lat_reg
    else:
      self.hpc = shade(IcD.lon, IcD.lat,
                           data, ax=ax, cax=cax, 
                           clim=clim, cmap=cmap, transform=transform,
                         ) 

      if isinstance(lon_reg, str) and lon_reg=='auto':
        lon_reg = [IcD.lon[0], IcD.lon[-1]]
      if isinstance(lat_reg, str) and lat_reg=='auto':
        lat_reg = [IcD.lat[0], IcD.lat[-1]]

    ax.set_xticks( np.linspace(lon_reg[0], lon_reg[1], 5) )
    ax.set_yticks( np.linspace(lat_reg[0], lat_reg[1], 5) )
    ax.set_xlim(*lon_reg)
    ax.set_ylim(*lat_reg)

    #ax.add_feature(cfeature.LAND, facecolor='0.7', zorder=3)
    ax.coastlines()

    if title=='auto':
      self.htitle = ax.set_title(IcD.long_name[var]+' ['+IcD.units[var]+']')
    else:
      self.htitle = ax.set_title(title)

    if time_string!='none':
      self.htstr = ax.text(0.05, 0.025, IcD.times[IcD.step_snap], 
                           transform=plt.gcf().transFigure)
    if depth_string!='none':
      self.hdstr = ax.text(0.05, 0.08, 'depth = %4.1fm'%(IcD.depth[IcD.iz]), 
                           transform=plt.gcf().transFigure)
    return
  
  def update(self, data, IcD, title='none', 
             time_string='auto', depth_string='auto'):
    if IcD.use_tgrid:
      data_nomasked_vals = data[IcD.maskTri==False]
      #print self.hpc[0].get_array.shape()
      self.hpc[0].set_array(data_nomasked_vals)
      #print self.hpc[0].get_array.shape()
      print('hello world')
    else:
      self.hpc[0].set_array(data[1:,1:].flatten())
    if title!='none':
      self.htitle.set_text(title) 
    if time_string!='none':
      self.htstr.set_text(IcD.times[IcD.step_snap])
    if depth_string!='none':
      self.hdstr.set_text('depth = %4.1fm'%(IcD.depth[IcD.iz]))
    return

# ================================================================================ 
# ================================================================================ 

def shade(x, y, data,
            ax='auto', cax=0,
            cmap='auto',
            rasterized=True,
            clim=[None, None],
            extend='both',
            conts=None,
            nclev='auto',
            cint='auto',
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
         ):
  """ Makes a nice pcolor(mesh) plot.

last change:
----------
2016-08-23
  """
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

  # make x and y 2D
  if x.ndim==1:
    x, y = np.meshgrid(x, y)

  # convert to Basemap maps coordinates
  if bmp is not None:
    x, y = bmp(x, y)
    
  # bring x and y to correct shape for contour
  if (use_cont) or (use_contf):
    if x.shape[1] != data.shape[1]:
      xc = 0.25*(x[1:,1:]+x[:-1,1:]+x[1:,:-1]+x[:-1,:-1])
      yc = 0.25*(y[1:,1:]+y[:-1,1:]+y[1:,:-1]+y[:-1,:-1])
    else:
      xc = 1.*x
      yc = 1.*y
    
  hs = []
  # pcolor plot
  if use_pcol:
    hm = ax.pcolormesh(x, y, data, 
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        rasterized=rasterized,
                        edgecolor=edgecolor,
                        **ccrsdict
                      )
    hs.append(hm)
  # contourf plot
  elif use_contf:
    hm = ax.contourf(xc, yc, data, contfs,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        extend='both',
                        **ccrsdict
                      )
    # this prevents white lines if fig is saved as pdf
    for cl in hm.collections: 
      cl.set_edgecolor("face")
      cl.set_rasterized(True)
    # add handle to hanlde list
    hs.append(hm)
    # rasterize
    if rasterized:
      zorder = -5
      ax.set_rasterization_zorder(zorder)
      for cl in hm.collections:
        cl.set_zorder(zorder - 1)
        cl.set_rasterized(True)
  else:
    hm = None

  # extra contours
  if use_cont:
    hc = ax.contour(xc, yc, data, conts, colors=contcolor, linewidths=contlw, **ccrsdict)
    try:
      i0 = np.where(hc.levels==contthick)[0][0]
      #hc.collections[i0].set_linewidth(1.5)
      hc.collections[i0].set_linewidth(2.5*contlw)
    except:
      #print "::: Warning: Could not make contour contthick=%g thick. :::" % (contthick)
      pass
    hs.append(hc)

  # colorbar
  if ((cax is not None) and (cax!=0)) and (hm is not None): 
    if cax == 1:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      div = make_axes_locatable(ax)
      cax = div.append_axes("right", size="10%", pad=0.1)
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    # this prevents white lines if fig is saved as pdf
    cb.solids.set_edgecolor("face")
    hs.append(cb)

    # colobar ticks
    if isinstance(cbticks, str) and cbticks=='auto':
      cb.formatter.set_powerlimits((-3, 2))
      tick_locator = ticker.MaxNLocator(nbins=8)
      cb.locator = tick_locator
      cb.update_ticks()
    else:
      cb.formatter.set_powerlimits((-1, 1))
      #cb.formatter.set_scientific(False)
      cb.update_ticks()

    cax.set_title(cbtitle)

  # labels and ticks
  if adjust_axlims:
    ax.locator_params(nbins=5)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
  return hs 

# ================================================================================ 
def trishade(Tri, data,
            ax='auto', cax=0,
            cmap='auto',
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
                        rasterized=rasterized,
                        **ccrsdict
                      )
    hs.append(hm)
  # contourf plot
  elif use_contf:
    hm = ax.contourf(xc, yc, data, contfs,
                        vmin=clim[0], vmax=clim[1],
                        cmap=cmap, 
                        extend='both',
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

  # colorbar
  if ((cax is not None) and (cax!=0)) and (hm is not None): 
    if cax == 1:
      from mpl_toolkits.axes_grid1 import make_axes_locatable
      div = make_axes_locatable(ax)
      cax = div.append_axes("right", size="10%", pad=0.1)
    cb = plt.colorbar(mappable=hm, cax=cax, extend=extend)
    # this prevents white lines if fig is saved as pdf
    cb.solids.set_edgecolor("face")
    hs.append(cb)

    # colobar ticks
    cb.formatter.set_powerlimits((-3, 2))
    tick_locator = ticker.MaxNLocator(nbins=8)
    cb.locator = tick_locator
    cb.update_ticks()

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
                  sasp = 1.0,
                  # plot colorbar
                  plot_cb = False,
                  # have common x or y axes
                  sharex = True, sharey = True,
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
                  fig_size_fac = 1.,
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
        hca[kk].tick_params(labelbottom='off')
        hca[kk].set_xlabel('')

      if sharey and ii!=0:
        hca[kk].ticklabel_format(axis='y',style='plain',useOffset=False)
        hca[kk].tick_params(labelleft='off')
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

