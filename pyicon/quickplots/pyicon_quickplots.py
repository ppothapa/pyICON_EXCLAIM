import sys, glob, os
import json
import shutil
import copy
# --- calculations
import numpy as np
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
#from ipdb import set_trace as mybreak  
import pyicon as pyic
import xarray as xr

# ================================================================================ 
# Quick Plots
# ================================================================================ 

# --------------------------------------------------------------------------------
# Horizontal plots
# --------------------------------------------------------------------------------
def qp_hplot(fpath, var, IcD='none', depth=-1e33, iz=0, it=0,
              t1='needs_to_be_specified', t2='none',
              it_ave=[],
              rgrid_name="orig",
              path_ckdtree="",
              var_fac = 1.,
              var_add = 0.,
              clim='auto', cincr=-1., cmap='auto',
              clevs=None,
              contfs=None,
              conts=None,
              xlim=[-180,180], ylim=[-90,90], projection='none',
              use_tgrid=False,
              crs_features=True,
              adjust_axlims=False,
              asp=0.5,
              title='auto', units='',
              xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              do_plot_settings=True,
              land_facecolor='0.7',
              do_mask=False,
              do_write_data_range=True,
              save_data=False,
              fpath_nc='./tmp.nc',
              ):

  #for fp in [fpath]:
  #  if not os.path.exists(fp):
  #    raise ValueError('::: Error: Cannot find file %s! :::' % (fp))


  # --- set-up grid and region if not given to function
  if isinstance(IcD,str) and IcD=='none':
    # get fname and path_data from fpath
    fname = fpath.split('/')[-1]
    path_data = ''
    for el in fpath.split('/')[1:-1]:
      path_data += '/'
      path_data += el
    path_data += '/'

    IcD = IconData(
                   fname   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   rgrid_name   = rgrid_name,
                   omit_last_file = False,
                  )
  #else:
  #  print('Using given IcD!')

  if depth!=-1e33:
    try:
      iz = np.argmin((IcD.depthc-depth)**2)
    except:
      iz = 0
  IaV = IcD.vars[var]
  step_snap = it

  # --- seems to be necessary for RUBY
  if IaV.coordinates=='':
    IaV.coordinates = 'clat clon'

  # --- load data 
  #IaV.load_hsnap(fpath=IcD.flist_ts[step_snap], 
  #                    it=IcD.its[step_snap], 
  #                    iz=iz,
  #                    step_snap = step_snap
  #                   ) 
  IaV.time_average(IcD, t1, t2, it_ave, iz=iz)

  # --- mask data
  if do_mask:
    IaV.data = np.ma.array(IaV.data)
    if IaV.coordinates=='clat clon':
      IaV.data[IcD.wet_c[iz,:]==0] = np.ma.masked
    elif IaV.coordinates=='elat elon':
      IaV.data[IcD.wet_e[iz,:]==0] = np.ma.masked
    else:
      raise ValueError("::: Error: Unknownc coordinates for mask!:::")

  # --- interpolate data 
  if not use_tgrid:
    IaV.interp_to_rectgrid(fpath_ckdtree=IcD.rgrid_fpath)
  # --- crop data

  IaV.data *= var_fac
  IaV.data += var_add

  if units!='':
    IaV.units = units

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
  else:
    ccrs_proj = getattr(ccrs, projection)()

  # --- do plotting
  (ax, cax, 
   hm,
   Dstr
  ) = pyic.hplot_base(
              IcD, IaV, 
              ax=ax, cax=cax,
              clim=clim, cmap=cmap, cincr=cincr,
              clevs=clevs,
              contfs=contfs,
              conts=conts,
              xlim=xlim, ylim=ylim,
              adjust_axlims=adjust_axlims,
              title=title, 
              projection=projection,
              crs_features=crs_features,
              use_tgrid=use_tgrid,
              logplot=logplot,
              asp=asp,
              do_plot_settings=do_plot_settings,
              land_facecolor=land_facecolor,
              do_write_data_range=do_write_data_range,
              save_data=save_data,
              fpath_nc=fpath_nc,
             )

  # --- contour labels
  if conts is not None:
    Cl = ax.clabel(hm[1], colors='k', fontsize=6, fmt='%.1f', inline=False)
    for txt in Cl:
      txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))

  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

def qp_vplot(fpath, var, IcD='none', it=0,
              t1='needs_to_be_specified', t2='none',
              it_ave=[],
              sec_name="specify_sec_name",
              path_ckdtree="",
              var_fac=1.,
              var_add=0.,
              clim='auto', cincr=-1., cmap='auto',
              clevs=None,
              contfs='auto',
              conts='auto',
              xlim=[-90,90], ylim=[6000,0], projection='none',
              asp=0.5,
              title='auto', xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              log2vax=False,
              mode_load='normal',
              do_plot_settings=True,
              do_write_data_range=True,
              save_data=False,
              fpath_nc='./tmp.nc',
              ):

  #for fp in [fpath]:
  #  if not os.path.exists(fp):
  #    raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  # --- load data set
  if isinstance(IcD,str) and IcD=='none':
    # get fname and path_data from fpath
    fname = fpath.split('/')[-1]
    path_data = ''
    for el in fpath.split('/')[1:-1]:
      path_data += '/'
      path_data += el
    path_data += '/'

    IcD = IconData(
                   fname   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   #rgrid_name   = rgrid_name
                   omit_last_file = False,
                  )
  #else:
  #  print('Using given IcD!')

  IaV = IcD.vars[var]
  step_snap = it

  # --- seems to be necessary for RUBY
  if IaV.coordinates=='':
    IaV.coordinates = 'clat clon'

  # --- load data
  # FIXME: MOC and ZAVE cases could go into load_vsnap
  if sec_name.endswith('moc'):
    #IaV.load_moc(
    #               fpath=IcD.flist_ts[step_snap], 
    #               it=IcD.its[step_snap], 
    #               step_snap = step_snap
    #              ) 
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    #IaV.data = IaV.data[:,:,0]/1e9 # MOC in nc-file as dim (nt,nz,ny,ndummy=1)
    IaV.data = IaV.data/1e9
    f = Dataset(IcD.flist_ts[0], 'r')
    IaV.lat_sec = f.variables['lat'][:]
    #IaV.depth = f.variables['depth'][:]
    #IcD.depthc = f.variables['depth'][:] # Fix: to avoid reading fx file in IcD_moc
    f.close()
    IaV.mask = IaV.data==0.
    IaV.data[IaV.mask] = np.ma.masked
  elif sec_name.startswith('zave'):
    basin      = sec_name.split(':')[1]
    rgrid_name = sec_name.split(':')[2]
    #IaV.lat_sec, IaV.data = pyic.zonal_average(
    #                               fpath_data=IcD.flist_ts[step_snap], 
    #                               var=var, basin=basin, it=it,
    #                               fpath_fx=IcD.fpath_fx, 
    #                               fpath_ckdtree=IcD.rgrid_fpaths[
    #                                 np.where(IcD.rgrid_names==rgrid_name)[0][0]]
    #                                     )
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(
                                   IaV.data, 
                                   basin=basin, coordinates=IaV.coordinates,
                                   fpath_fx=IcD.fpath_fx,
                                   fpath_ckdtree=IcD.rgrid_fpaths[
                                     np.where(IcD.rgrid_names==rgrid_name)[0][0]],
                                                      )
  else:
    sec_fpath = IcD.sec_fpaths[np.where(IcD.sec_names==sec_name)[0][0] ]
    #IaV.load_vsnap(
    #               fpath=IcD.flist_ts[step_snap], 
    #               fpath_ckdtree=sec_fpath,
    #               it=IcD.its[step_snap], 
    #               step_snap = step_snap
    #              ) 
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    # --- interpolate data 
    if not IcD.use_tgrid:
      IaV.interp_to_section(fpath_ckdtree=sec_fpath)

    IaV.data *= var_fac
    IaV.data += var_add

  # --- do plotting
  (ax, cax, 
   hm,
   Dstr
  ) = pyic.vplot_base(
                 IcD, IaV, 
                 ax=ax, cax=cax,
                 clim=clim, cmap=cmap, cincr=cincr,
                 clevs=clevs,
                 contfs=contfs,
                 conts=conts,
                 title=title, 
                 log2vax=log2vax,
                 logplot=logplot,
                 do_plot_settings=do_plot_settings,
                 do_write_data_range=do_write_data_range,
                 save_data=save_data,
                 fpath_nc=fpath_nc,
                )

  # --- contour labels
  if conts is not None:
    Cl = ax.clabel(hm[1], colors='k', fontsize=6, fmt='%.1f', inline=False)
    for txt in Cl:
      txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))

  # ---
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

def time_averages_monitoring(IcD, t1, t2, varlist, var_fac_list=[], var_add_list=[], var_units_list=[]): 

  # --- define time bounds for correct time averaging
  time_bnds = np.copy(IcD.times)
  dt64type = time_bnds[0].dtype
  # find year, month and day integers of first time step
  yy, mm, dd = pyic.datetime64_to_float(time_bnds[0])
  if mm==1:
    yy += -1
    mm = 13
  # first time value is first value of time series minus one month
  time_bnds = np.concatenate(([np.datetime64(f'{yy:04d}-{mm-1:02d}-{dd:02d}').astype(dt64type)],time_bnds))
  # dt is the length of a time interval
  dt = np.diff(time_bnds).astype(float)

  if len(var_fac_list)==0:
    var_fac_list = [1]*len(varlist)
  if len(var_add_list)==0:
    var_add_list = [0]*len(varlist)
  if len(var_units_list)==0:
    var_units_list = ['']*len(varlist)

  if len(var_fac_list)!=len(varlist):
    raise ValueError('::: Error: len(var_fac_list)!=len(varlist)! :::')
  if len(var_add_list)!=len(varlist):
    raise ValueError('::: Error: len(var_add_list)!=len(varlist)! :::')
  if len(var_units_list)!=len(varlist):
    raise ValueError('::: Error: len(var_units_list)!=len(varlist)! :::')

  Dvars = dict()
  for mm, var in enumerate(varlist):
    #print(var)
    Dvars[var] = dict()
    data = np.array([])
    for nn, fpath in enumerate(IcD.flist):
      f = Dataset(fpath, 'r')
      data_file = f.variables[var][:,0,0]
      data = np.concatenate((data, data_file))
      f.close()
    data *= var_fac_list[mm]
    data += var_add_list[mm]
    f = Dataset(fpath, 'r')
    Dvars[var]['long_name'] = f.variables[var].long_name
    if var_units_list[mm]=='':
      Dvars[var]['units'] = f.variables[var].units
    else: 
      Dvars[var]['units'] = var_units_list[mm]
    f.close()
    ind = (IcD.times>=t1) & (IcD.times<=t2)
    mean = (data[ind]*dt[ind]).sum()/dt[ind].sum()
    std = np.sqrt( (data[ind]**2*dt[ind]).sum()/dt[ind].sum()-mean**2 )
    Dvars[var]['ave'] = mean
    Dvars[var]['std'] = std
    Dvars[var]['min'] = data[ind].min()
    Dvars[var]['max'] = data[ind].max()
    Dvars[var]['tab_name'] = ''
    Dvars[var]['fac'] = 1.
    Dvars[var]['prec'] = '.5g'
  return Dvars

def qp_timeseries(IcD, fname, vars_plot, 
                  var_fac=1., var_add=0.,
                  title='', units='',
                  t1='none', t2='none',
                  lstart=None, lend=None,
                  ave_freq=0,
                  omit_last_file=True,
                  mode_ave=['mean'],
                  labels=None,
                  do_write_data_range=True,
                  ax='none',
                  save_data=False,
                  fpath_nc='./tmp.nc',
                 ): 

  if len(mode_ave)==1:
    mode_ave = [mode_ave[0]]*len(vars_plot)
    dfigb = 0.7
  else:
    do_write_data_range = False
    dfigb = 0.0

  # --- identify all files and time points belonging to time series
  # start: not needed if IcD.load_timeseries is used
  flist = glob.glob(IcD.path_data+fname)
  flist.sort()
  if omit_last_file:
    flist = flist[:-1]
  times, flist_ts, its = pyic.get_timesteps(flist)
  # end: not needed if IcD.load_timeseries is used

  # --- prepare time averaging
  times_plot = np.copy(times)
  if ave_freq>0:
    # skip all time points which do not fit in final year
    nskip = times.size%ave_freq
    if nskip>0:
      times = times[:-nskip]
    # define time bounds for correct time averaging
    time_bnds = np.copy(times)
    dt64type = time_bnds[0].dtype
    # find year, month and day integers of first time step
    yy, mm, dd = pyic.datetime64_to_float(time_bnds[0])
    if mm==1:
      yy += -1
      mm = 13
    # first time value is first value of time series minus one month
    time_bnds = np.concatenate(([np.datetime64(f'{yy:04d}-{mm-1:02d}-{dd:02d}').astype(dt64type)],time_bnds))
    # dt is the length of a time interval
    dt = np.diff(time_bnds).astype(float)
    nresh = int(times.size/ave_freq)
    times = np.reshape(times, (nresh, ave_freq)).transpose()
    # finally define times_plot as center or averaging time intervall
    #times_ave = times.mean(axis=0)
    times_plot = times[int(ave_freq/2),:] # get middle of ave_freq

  # --- make axes if they are not given as arguement
  if isinstance(ax, str) and ax=='none':
    hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="time [years]", ylabel="", 
                 dfigb=dfigb,
                 )
    ii=-1
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    #adjust_xylim = True
    adjust_xylim = False
  else:
    adjust_xylim = False
  
  # --- initialize nc Dataset
  if save_data:
    coords = {'times': times_plot[slice(lstart,lend)]}
    ds = xr.Dataset()
    #ds['time_bnds'] = xr.DataArray(times[sice(lstart,lend)])

  # --- loop over all variables which should be plotted
  for mm, var in enumerate(vars_plot):
    # --- load the data
    # start: not needed if IcD.load_timeseries is used
    data = np.array([])
    for nn, fpath in enumerate(flist):
      f = Dataset(fpath, 'r')
      data_file = f.variables[var][:,0,0]
      data = np.concatenate((data, data_file))
      if nn==0:
        long_name_ncout = f.variables[var].long_name
        if units!='':
          units_ncout = f.variables[var].units 
        else:
          units_ncout = units
      f.close()
    # end: not needed if IcD.load_timeseries is used

    # --- time averaging
    dtsum = np.ones((times.size))
    if ave_freq>0:
      if nskip>0:
        data = data[:-nskip]
      #print(f'{var}: {data.size} {times_plot.size}')
      data = np.reshape(data, (nresh, ave_freq)).transpose()
      dt   = np.reshape(dt  , (nresh, ave_freq)).transpose()
      if mode_ave[mm]=='mean':
        #data = data.mean(axis=0))
        data = (data*dt).sum(axis=0)/dt.sum(axis=0)
        dtsum = dt.sum(axis=0)
      elif mode_ave[mm]=='min':
        data = data.min(axis=0)
      elif mode_ave[mm]=='max':
        data = data.max(axis=0)

    # --- modify data if var_fac or var_add are given
    data *= var_fac
    data += var_add

    # --- skip data at start and end if lstart and lend are defined
    times_plot = times_plot[slice(lstart,lend)]
    data  = data[slice(lstart,lend)]
    dtsum = dtsum[slice(lstart,lend)]

    # --- define labels
    if labels is None:
      label = var
    else:
      label = labels[mm]

    # --- finally plotting
    hl, = ax.plot(times_plot, data, label=label)

    if adjust_xylim:
      ax.set_xlim([times_plot.min(), times_plot.max()])
      ax.set_ylim([data.min(), data.max()])

    if save_data:
      # --- add data array to dataset
      ds[var] = xr.DataArray(data, dims=coords.keys(), coords=coords, attrs={'units': units_ncout, 'long_name': long_name_ncout})

  # --- grid
  ax.grid(True)

  # --- title
  if len(vars_plot)==1:
    f = Dataset(fpath, 'r')
    if units=='':
      units = f.variables[var].units
    units = f' [{units}]'
    if title=='':
      long_name = f.variables[var].long_name
      title = long_name+units
    f.close()
  ax.set_title(title)

  # --- legend
  if len(vars_plot)>1:
    ax.legend()

  # --- vertical lines indicating time frame
  if not (isinstance(t1,str) and t1=='none'):
    hlt1 = ax.axvline(t1, color='k')
  if not (isinstance(t2,str) and t2=='none'):
    hlt2 = ax.axvline(t2, color='k')

  # --- data range information below figure
  if do_write_data_range:
    ind = (times_plot>=t1) & (times_plot<=t2)
    #data_mean = data[ind].mean()  # inaccurate if data consists of monthly means
    data_mean = (data[ind]*dtsum[ind]).sum()/dtsum[ind].sum()
    #print(f'old: {data[ind].mean()}') 
    #print(f'new: {(data[ind]*dtsum[ind]).sum()/dtsum[ind].sum()}') 
    try:
      info_str = 'in timeframe: min: %.4g;        mean: %.4g;        std: %.4g;        max: %.4g' % (data[ind].min(), data_mean, data[ind].std(), data[ind].max())
      ax.text(0.5, -0.18, info_str, ha='center', va='top', transform=ax.transAxes)
    except:
      pass

  if save_data:
    # --- write netcdf file
    print(f'Writing data file {fpath_nc}.')
    ds.to_netcdf(fpath_nc)

  FigInf = dict()
  Dhandles = dict()
  Dhandles['ax'] = ax
  Dhandles['hl'] = hl
  Dhandles['hlt1'] = hlt1
  Dhandles['hlt2'] = hlt2
  return FigInf, Dhandles

def write_table_html(data, leftcol=[], toprow=[], prec='.1f', width='80%'):
  data = np.array(data)
  toprow = np.array(toprow)
  leftcol = np.array(leftcol)
  if toprow.size!=0 and toprow.size!=data.shape[1]:
    raise ValueError('::: Error: Wrong number of entries in toprow! :::')
  if leftcol.size!=0 and leftcol.size!=data.shape[0]:
    raise ValueError('::: Error: Wrong number of entries in leftcol! :::')
  text = ""
  text += '<p></p>\n'
  text += f'<table style="width:{width}">\n'
  # header title row toprow
  if toprow.size!=0:
    text += '  <tr>'
    # upper left empy entry
    if leftcol.size!=0:
      text += '    <th></th>'
    for el in toprow:
      text += f'    <th>{el}</th>'
    text += '  </tr>\n'
  # table body
  for jj in range(data.shape[0]):
    text += '  <tr>'
    # left title column leftcol
    if leftcol.size!=0:
      text += f'    <th>{leftcol[jj]}</th>'
    for ii in range(data.shape[1]):
      # main entries
      if data.dtype.type is np.str_:
        text += f'    <td>{data[jj,ii]}</td>'
      else:
        text += f'    <td>{data[jj,ii]:{prec}}</td>'
    text += '  </tr>\n'
  text += '</table>\n'
  text += '<p></p>\n'
  return text

##def qp_hor_plot(fpath, var, IC='none', iz=0, it=0,
##              grid='orig', 
##              path_rgrid="",
##              clim='auto', cincr='auto', cmap='auto',
##              xlim=[-180,180], ylim=[-90,90], projection='none',
##              title='auto', xlabel='', ylabel='',
##              verbose=1,
##              ax='auto', cax=1,
##              ):
##
##
##  # --- load data
##  fi = Dataset(fpath, 'r')
##  data = fi.variables[var][it,iz,:]
##  if verbose>0:
##    print('Plotting variable: %s: %s' % (var, IC.long_name)) 
##
##  # --- set-up grid and region if not given to function
##  if isinstance(IC,str) and clim=='none':
##    pass
##  else:
##    IC = IconDataFile(fpath, path_grid='/pool/data/ICON/oes/input/r0003/')
##    IC.identify_grid()
##    IC.load_tripolar_grid()
##    IC.data = data
##    if grid=='orig':
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      IC.Tri = matplotlib.tri.Triangulation(IC.vlon, IC.vlat, 
##                                            triangles=IC.vertex_of_cell)
##      IC.mask_big_triangles()
##      use_tgrid = True
##    else: 
##      # --- rectangular grid
##      if not os.path.exists(path_rgrid+grid):
##        raise ValueError('::: Error: Cannot find grid file %s! :::' % 
##          (path_rgrid+grid))
##      ddnpz = np.load(path_rgrid+grid)
##      IC.lon, IC.lat = ddnpz['lon'], ddnpz['lat']
##      IC.Lon, IC.Lat = np.meshgrid(IC.lon, IC.lat)
##      IC.data = icon_to_regular_grid(IC.data, IC.Lon.shape, 
##                          distances=ddnpz['dckdtree'], inds=ddnpz['ickdtree'])
##      IC.data[IC.data==0] = np.ma.masked
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      use_tgrid = False
##  IC.data = IC.data[IC.ind_reg]
##
##  IC.long_name = fi.variables[var].long_name
##  IC.units = fi.variables[var].units
##  IC.name = var
##
##  ax, cax, hm = hplot_base(IC, var, clim=clim, title=title, 
##    projection=projection, use_tgrid=use_tgrid)
##
##  fi.close()
##
##  # --- output
##  FigInf = dict()
##  FigInf['fpath'] = fpath
##  FigInf['long_name'] = long_name
##  FigInf['IC'] = IC
##  #ipdb.set_trace()
##  return FigInf

# ================================================================================ 
# ================================================================================ 
class QuickPlotWebsite(object):
  """ Creates a website where the quick plots can be shown.

Minimal example:

# --- setup
qp = QuickPlotWebsite(
  title='pyicon Quick Plot', 
  author='Nils Brueggemann', 
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  fpath_css='./mycss2.css'
  )

# --- content
for i in range(1,11):
  qp.add_section('Section %d' % i)
  for j in range(1,4):
    qp.add_subsection('Subsection %d.%d' % (i,j))
    qp.add_paragraph(('').join(['asdf %d'%(i)]*10))
    qp.add_fig('./pics/','fig_01.png')
qp.write_to_file()
  """

  def __init__(self, title='Quick Plot', author='', date='', 
               info='', path_data='',
               fpath_css='', fpath_html='./qp_index.html', links='auto'):
    self.author = author 
    self.title = title
    self.date = date
    self.info = info
    self.path_data = path_data
    self.fpath_css = fpath_css
    self.fpath_html = fpath_html

    self.first_add_section_call = True

    self.main = ""
    self.toc = ""
    if links=='auto':
      if 'timeaverages' in title:
        links = """
&emsp; <a href="../index.html">list simulations</a>
"""
      elif 'simulations' in title:
        links = ""
      elif path_data.startswith('Compare'):
        links = """
&emsp; <a href="../index.html">list comparisons</a>
"""
      else:
        links = """
&emsp; <a href="../qp_index.html">list time averages</a>
&emsp; <a href="../../index.html">list simulations</a>
&emsp; <a href="../add_info/add_info.html">additional information</a>
"""

    self.header = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">

<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta http-equiv="Content-Style-Type" content="text/css" />
  <meta name="generator" content="pyicon" />
  <meta name="author" content="{author}" />
  <title>{title}</title>
  <style type="text/css">code{{white-space: pre;}}</style>
  <link rel="stylesheet" href="{fpath_css}" type="text/css" />
</head>

<body>

<div id="header">
<h1 class="title">{title}</h1>
<p> {author} | {date} | {path_data} </p>
<p> {info} 
{links}
</p>
</div>

""".format(author=self.author, title=self.title, 
           date=self.date, 
           path_data=self.path_data,
           info=self.info,
           fpath_css=self.fpath_css, 
           links=links,
          )

#<div id="header">
#<h1 class="title">{title}</h1>
#<h2 class="author">{author}</h2>
#<h3 class="date">{date}</h3>
#</div>

    self.footer = """
</body>
</html>
"""
  
  def add_section(self, title='Section'):
    # --- add to main
    href = title.replace(' ', '-')
    self.main += '\n'
    #self.main += f"<h1 id=\"{href}\">{title}</h1>\n"
    self.main += f"  <div id=\"ctn\">"
    self.main += f"    <a name=\"{href}\">&nbsp;</a>"
    self.main += f"    <h1 class=\"target-label\">{title}</h2>"
    self.main += f"  </div>"
    # --- add to toc
    if self.first_add_section_call:
      self.first_add_section_call = False
      self.toc += """
<div id="TOC">
<ul>
"""
    else:
      self.toc += '</ul></li> \n'
    self.toc += f'<li><a href="#{href}">{title}</a><ul>\n'
    return

  def add_subsection(self, title='Subsection'):
    # --- add to main
    href = title.replace(' ', '-')
    #self.main += f"  <h2 id=\"{href}\">{title}</h2>\n"
    self.main += f"  <div id=\"ctn\">"
    self.main += f"    <a name=\"{href}\">&nbsp;</a>"
    self.main += f"    <h2 class=\"target-label\">{title}</h2>"
    self.main += f"  </div>"
    # --- add to toc
    self.toc += f'<li><a href="#{href}">{title}</a></li>\n'
    return

  def add_paragraph(self, text=''):
    self.main += '    <p>'
    self.main += text
    self.main += '    </p>'
    self.main += '\n'
    return
   
  def add_html(self, fpath):
    f = open(fpath, 'r')
    text = f.read()
    f.close()
    self.main += text
    #self.main += f'    <!--#include virtual="{fpath}" -->'
    self.main += '\n'
    return

  def add_fig(self, fpath, width="1000"):
    self.main += f'    <div class="figure"> <img src="{fpath}" width="{width}" /> </div>'
    self.main += '\n'
    return
  
  def close_toc(self):
    # --- close toc
    self.toc += """</ul></li>
</ul>
</div>

"""
    return

  def write_to_file(self):
    # --- close toc
    self.close_toc()

    # --- write to output file
    print(f'Writing file {self.fpath_html}')
    f = open(self.fpath_html, 'w')
    f.write(self.header)
    f.write(self.toc)
    f.write(self.main)
    f.write(self.footer)
    f.close()
    return

# ================================================================================ 
# ================================================================================ 
def link_all(path_quickplots='../../all_qps/', path_search='path_quickplots'):
  """ 
Link all sub pages
  * either to qp_index.html if sub pages for time averages are linked
  * or to index.html if sub pages for simulations are linked  

Example usage:
--------------

To link sub pages of all simulations:
  pyicqp.link_all(path_quickplots='../../all_qps/')

To link sub pages for time averages for a specific simulation:
  pyicqp.link_all(path_quickplots='../../all_qps/', path_search='../../all_qps/qp-slo1284/')

  """
  #print('la: search path')

  path_qp_driver = os.path.dirname(__file__)+'/'
  
  if path_search=='path_quickplots':
    path_search = path_quickplots
    top_level = True
    title='List of all simulations'
    fname_html = 'index.html'
  else:
    if not path_search.endswith('/'):
      path_search += '/'
    run = path_search.split('/')[-2][3:]
    top_level = False
    title = f'List of timeaverages for {run}'
    fname_html = 'qp_index.html'
  
  #print('la: find all pages')
  # --- find all pages that should be linked
  flist = glob.glob(path_search+'*/qp_index.html')
  flist.sort()
  
  #print('qp_link_all: ',flist)
  
  #print('la: make header')
  # --- make header
  qp = QuickPlotWebsite(
    title=title,
    author=os.environ.get('USER'),
    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #path_data=path_quickplots,
    info='ICON data plotted by pyicon.quickplots',
    fpath_css='./qp_css.css',
    fpath_html=path_search+fname_html
    )
  
  # --- copy css file
  #print('la: copy css')
  shutil.copyfile(path_qp_driver+'qp_css.css', path_search+'qp_css.css')
  
  #print('la: do content')
  # --- start with content
  text = ''
  # --- add link to pyicon docu
  if top_level:
    text += '<p><li><a href="pyicon_doc/html/index.html">pyicon documentation</a></>\n'
  else:
    text += '<p><li><a href="add_info/add_info.html">additional information</a></>\n'
  # --- add link to experiments / timeaverages
  for fpath in flist:
    name = fpath.split('/')[-2]#[3:]
    name = name.replace('qp-','')
    rpath = fpath.replace(path_search,'')
    print(rpath, name)
    text += '<p>'
    text += '<li><a href=\"'+rpath+'\">'+name+'</a>'
    text += '</>\n'
  qp.main = text
  
  # --- finally put everything together
  if True:
    #print('la: write to file')
    qp.write_to_file()
  # --- for diagnostics
  else:
    print(qp.header)
    print(qp.toc)
    print(qp.main)
    print(qp.footer)
  return

def add_info(run, path_data, path_qp_sim, verbose=True):
  path_add_info = path_qp_sim+'add_info/'
  if not os.path.exists(path_add_info):
    os.makedirs(path_add_info)

  # --- make header
  qp = QuickPlotWebsite(
    title='Additional information',
    author=os.environ.get('USER'),
    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #path_data=path_quickplots,
    info='ICON data plotted by pyicon.quickplots',
    fpath_css='../qp_css.css',
    fpath_html=path_add_info+'add_info.html'
    )
  
  """
<p><li><a href="README">README</a></>
<p><li>Run script: <a href="exp.slo1284.run">exp.slo1284.run</a></>
<p><li><a href="icon_ruby_na_circ_mov_dap7023-r0.mov">Movie</a></>
  """
  text = ""
  #f = open(f'{path_data}/../../run/README', 'r')
  #text += f.read()
  #f.close()
  #text += '\n'
  flist = [] 
  flist += [f'{path_data}/README']
  #flist += [f'{path_data}/../../run/exp.{run}.run']
  flist += glob.glob(f'{path_data}/../../run/*{run}*.run')
  namelist = glob.glob(f'{path_data}/NAMELIST*')
  namelist.sort()
  flist += namelist
  #flist += [f'{path_data}/NAMELIST_ICON_output_atm']
  #flist += [f'{path_data}/NAMELIST_{run}_atm']
  #flist += [f'{path_data}/NAMELIST_{run}_lnd']
  #flist += [f'{path_data}/NAMELIST_{run}_oce']
  #flist += [f'{path_data}/NAMELIST_{run}_oce.log']
  #flist += [f'{path_data}/NAMELIST_{run}_oce_output']
  for fpath in flist:
    try:
      shutil.copy(fpath, f'{path_add_info}')
      fname = fpath.split('/')[-1]
      text += f'<p><li><a href="{fname}">{fname}</a></>'
    except:
      if verbose:
        print(f'::: Warning: Cannot find {fpath}! :::')

  qp.main = text
  qp.write_to_file()
  return
