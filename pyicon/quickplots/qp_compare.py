import sys, glob, os
import argparse
from ipdb import set_trace as mybreak
import matplotlib
#if iopts.batch or iopts.slurm:
#  matplotlib.use('Agg')
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import json
#sys.path.append('/home/mpim/m300602/pyicon')
import pyicon as pyic
import pyicon.quickplots as pyicqp
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from qp_cmaps import PyicCmaps
import xarray as xr
import pandas as pd
import f90nml
import seawater as sw

class Settings(object):
  def __init__(self):
    return

class Simulation(object):
  def __init__(self, name='', path_data=''):
    self.name = name
    self.path_data = path_data
    return

def save_fig(title, Set):
  fpath = Set.path_pics+title+Set.fig_sfx+'.png'
  print(f'Saving {fpath}')
  plt.savefig(fpath, dpi=300)
  plt.close()
  return

def save_tab(txt, title, Set):
  fpath = Set.path_pics+title+Set.fig_sfx+'.html'
  print(f'Saving {fpath}')
  with open(fpath, 'w') as f:
    f.write(txt)
  return

def do_clabels(mappable, ax):
  Cl = ax.clabel(mappable, colors='k', fontsize=6, fmt='%.1f', inline=False)
  for txt in Cl:
    txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))
  return

def time_ave(da, S, Set):
  if Set.do_timeave:
    da_tave = da.sel(time=slice(S.t1, S.t2)).mean(dim='time')
  else:
    da_tave = da
  return da_tave

# Settings default
# ----------------
Set = Settings()

Set.path_base = './'
Set.path_pics = Set.path_base+'pics/'

Set.do_diff = True
Set.compare_with_reference = False
Set.omit_last_file = False
Set.do_timeave = True

Set.tstr = '????????????????'
Set.tstr_ts = '????????????????'
Set.prfx_3d      = '_P1Y_3d'
Set.prfx_2d      = '_P1M_2d'
Set.prfx_monitor = '_P1M_mon'
Set.prfx_moc     = '_P1M_moc'
Set.prfx_cvmix   = '_P1Y_tke'

Set.mfdset_kwargs = dict(combine='nested', concat_dim='time')

Set.ndo_twice = 1
xticks = np.arange(-90, 100, 30)

# Read input arguments
# --------------------
help_text = ""
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

# --- necessary arguments
parser.add_argument('fpath_config', metavar='fpath_config', type=str,
                    help='path to quickplot configure file')
# --- optional arguments

# --- read input
iopts = parser.parse_args()

# Settings config file
# --------------------
#fpath_config = './config_02.py'
#fpath_config = './config_r2b4_diff_GM_02.py'
fpath_config = iopts.fpath_config
exec(open(fpath_config).read())

#Set.path_base = '/mnt/lustre01/pf/zmaw/m300602/qp_compare_test/'
#Set.path_pics = Set.path_base+'pics/'
#
#Sims = []
##path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/old_experiments/'
#path_base = '/home/mpim/m300602/work/proj_vmix/icon/icon_17/icon-oes/experiments/'
#
## Simulations
## -----------
#S = Simulation()
#S.run = 'nib0001'
#Sims.append(S)
#
##S = Simulation()
##S.run = 'nib0002'
##Sims.append(S)
##
##S = Simulation()
##S.run = 'nib0003'
##Sims.append(S)
##
##S = Simulation()
##S.run = 'nib0004'
##Sims.append(S)
#
#S = Simulation()
#S.run = 'nib0005'
#Sims.append(S)
#
#for nn, S in enumerate(Sims):
#  gname = 'r2b4_oce_r0004'
#  lev = 'L40'
#  path_grid = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'
#  S.path_data = f'{path_base}/{S.run}/'
#  #S.tave_int = ['2090', '2100']
#  S.tave_int = ['2040', '2050']
#  S.name = S.run
#  S.fpath_ckdtree = f'{path_grid}/{gname}/ckdtree/rectgrids/{gname}_res1.00_180W-180E_90S-90N.npz' 
#  S.fpath_tgrid   = f'{path_grid}/{gname}/{gname}_tgrid.nc'
#  #S.fpath_fx      = f'{path_grid}/{gname}/{gname}_{lev}_fx.nc')
#  S.fpath_fx      = f'{S.path_data}{S.run}_fx.nc'

# --- list of runs
runs = []
for nn, S in enumerate(Sims): 
  runs.append(S.run)

# --- read namelists
dfs = []
for nn, S in enumerate(Sims): 
    run = S.run
    # ------ open namelist
    nml = f90nml.read(S.namelist_oce)
    # ------ go through all sections (sub-namelists)
    df = pd.DataFrame() 
    df.loc[0, 'run'] = run
    for nml_sec in nml.keys():
      if nml_sec=='output_nml':
        continue
      for entr in nml[nml_sec]:
        if entr=='dzlev_m':
          continue
        try:
          df.loc[0, entr] = nml[nml_sec][entr]
        except:
          print(f'xx  {entr}')
    dfs.append(df)
# ----- only keep parameters which have changed
dfm = pd.concat(dfs)
ll = list(dfm.columns)
cols_unique = []
for col in list(dfm.columns):
  #print(f'col={col}, {np.unique(dfm.loc[:,col]).size}')
  if np.unique(dfm.loc[:,col]).size>1:
    #print(col)
    cols_unique.append(col)
dfu = dfm.loc[:,cols_unique]
dfu = dfu.drop(['restart_filename'], axis=1)
dfu.reset_index(drop=True, inplace=True) 
tstr_list = []
for S in Sims:
  tstr_list.append(str(S.tave_int[0])+' - '+str(S.tave_int[1]))
dfu.insert(loc=1, column='time_average', value=tstr_list)
dfu = dfu.set_index('run') 
tab_parameters_html = dfu.transpose().to_html()

Set.plot_names = []
Set.plot_names += ['sec:Information']
Set.plot_names += ['tab_parameters']
Set.plot_names += ['sec:Upper ocean']
Set.plot_names += ['ssh', 'mld_mar', 'mld_sep']
Set.plot_names += ['ice_concentration_nh_mar', 'ice_concentration_nh_sep']
Set.plot_names += ['ice_thickness_nh_mar', 'ice_thickness_nh_mar']
#Set.plot_names += ['ice_concentration_sh_mar', 'ice_concentration_sh_sep']
#Set.plot_names += ['ice_thickness_sh_mar', 'ice_thickness_sh_mar']
Set.plot_names += ['sec:Water masses']
Set.plot_names += ['temp_depths_intervals']
Set.plot_names += ['temp_gzave', 'temp_azave', 'temp_ipzave']
Set.plot_names += ['salt_depths_intervals']
Set.plot_names += ['salt_gzave', 'salt_azave', 'salt_ipzave']
Set.plot_names += ['temp_salt_profiles', 'temp_salt_hor_ave']
Set.plot_names += ['density_gzave']
Set.plot_names += ['sec:Transports']
Set.plot_names += ['amoc', 'pmoc', 'gmoc']
Set.plot_names += ['moc_profile']
Set.plot_names += ['ts_amoc']
Set.plot_names += ['htr']
Set.plot_names += ['ke100m', 'ke2000m']
Set.plot_names += ['tab_transport_sections']
Set.plot_names += ['bstr']
#Set.plot_names += ['sec:Surface fluxes']
#Set.plot_names += ['hfbasin']
#Set.plot_names += ['ffl']
#Set.plot_names += ['taux']
#Set.plot_names += ['tauy']
Set.plot_names += ['sec:Vertical mixing']
Set.plot_names += ['kv_depths_intervals', 'kv_gzave', 'tke_gzave']
Set.plot_names += ['sec:Time series']
Set.plot_names += ['ts_kin', 'ts_global_heat_content']
Set.fig_names = Set.plot_names

# --- for debugging
#Set.fig_names = []
#Set.fig_names += ['ts_amoc']
#Set.fig_names += ['tab_parameters']
#Set.fig_names += ['ts_kin']
#Set.fig_names += ['temp_salt_profiles']
#Set.fig_names += ['temp_depths_intervals']
#Set.fig_names += ['temp_salt_hor_ave']
#Set.fig_names += ['bstr']
#Set.fig_names += ['ke100m', 'ke2000m']
#Set.fig_names = ['hfbasin']
#Set.fig_names += ['moc_profile']
#Set.fig_names = ['ice_concentration_nh']
#Set.fig_names += ['kv_depths_intervals']
#Set.fig_names += ['kv_gzave', 'tke_gzave']
#Set.fig_names += ['density_gzave']
#Set.fig_names += ['tab_transport_sections']

# Prepare paths
# -------------
if os.path.exists(Set.path_base):
  print(f'::: Warning: path {Set.path_base} does already exist. :::') 
else:
  os.mkdir(Set.path_base)
  os.mkdir(Set.path_base+'/pics/')

def get_flist(searchstr):
  flist = glob.glob(searchstr)
  flist.sort()
  if Set.omit_last_file:
    flist = flist[:-1]
  return flist

if True:
  for nn, S in enumerate(Sims):
    print(f'Loading {S.run}')

    flist = get_flist(f'{S.path_data}{S.run}{Set.prfx_monitor}_{Set.tstr_ts}.nc')
    S.ds_ts = xr.open_mfdataset(flist,
      **Set.mfdset_kwargs
      )
  
    flist = get_flist(f'{S.path_data}{S.run}{Set.prfx_moc}_{Set.tstr}.nc')
    S.ds_moc = xr.open_mfdataset(flist,
      **Set.mfdset_kwargs
      )
  
    flist = get_flist(f'{S.path_data}{S.run}{Set.prfx_3d}_{Set.tstr}.nc')
    S.ds_3d = xr.open_mfdataset(flist,
      **Set.mfdset_kwargs
      )

    flist = get_flist(f'{S.path_data}{S.run}{Set.prfx_2d}_{Set.tstr}.nc')
    S.ds_2d = xr.open_mfdataset(flist,
      **Set.mfdset_kwargs
      )

    flist = get_flist(f'{S.path_data}{S.run}{Set.prfx_cvmix}_{Set.tstr}.nc')
    S.ds_cvmix = xr.open_mfdataset(flist,
      **Set.mfdset_kwargs
      )

    print(f'--- done with xr.open_mfdataset')
    #if use_num2date:
    #  time = S.ds_ts.time
    #  time = num2date(time.data, units=time.units, calendar=time.calendar).astype("datetime64[s]")

    # --- time averaging
    S.t1, S.t2 = S.tave_int
    #mask_int = (S.ds_2d.time>=np.datetime64(S.t1)) & (S.ds_2d.time<=np.datetime64(S.t2))
    #months = S.ds_2d.time.data.astype('datetime64[M]').astype(int) % 12 + 1
    #S.it_ave_mar = np.where( mask_int & (months==4)  )[0]
    #S.it_ave_sep = np.where( mask_int & (months==10) )[0]
    try:
      # for numpy.datetime64 (only works for some time periods)
      month_list = S.ds_2d.time.dt.month
      S.it_mar = S.ds_2d.groupby(month_list==4).groups[1]    
      S.it_sep = S.ds_2d.groupby(month_list==10).groups[1]    
    except:
      # for pandas / cftime
      month_list = [item.month for item in S.ds_2d.time.data]
      S.it_mar = month_list==4
      S.it_sep = month_list==10
    #S.it_mar = S.ds_2d.groupby(S.ds_2d.time.dt.month==4).groups[1]
    #S.it_sep = S.ds_2d.groupby(S.ds_2d.time.dt.month==10).groups[1]

    # --- data set of interpolated data
    S.dsi = xr.Dataset()
  
    # --- loading / interpolating grid files
    S.fx = xr.open_dataset(f'{S.fpath_fx}')
    S.tgrid = xr.open_dataset(f'{S.fpath_tgrid}')
    S.tgrid['dtype'] = 'float32'

    lon, lat, S.basin_ci = pyic.interp_to_rectgrid(S.fx.basin_c, fpath_ckdtree=S.fpath_ckdtree)
    #S.basin_ci = S.basin_ci[np.newaxis,:,:]
    S.dsi['basin_ci'] = xr.DataArray(S.basin_ci, dims=['lat', 'lon'], coords=dict(lat=lat, lon=lon))

    S.depth = S.ds_3d.depth
    S.depthi = S.fx.depth_2
    S.depthi = S.depthi.rename(dict(depth_2='depthi'))
    S.dzw = S.fx.prism_thick_c
    _, _, S.dzwi = pyic.interp_to_rectgrid(S.dzw, fpath_ckdtree=S.fpath_ckdtree)
    S.dzt = S.fx.constantPrismCenters_Zdistance
    _, _, S.dzti = pyic.interp_to_rectgrid(S.dzt, fpath_ckdtree=S.fpath_ckdtree)
  
    # --- interpolating/integrating 3D data 
    print('--- loading to')
    to = S.ds_3d.to.sel(time=slice(S.t1, S.t2)).mean(dim='time') 
    to = time_ave(S.ds_3d.to, S, Set)
    to = to.where(S.fx.wet_c==1.)

    print('--- loading so')
    so = time_ave(S.ds_3d.so, S, Set)
    so = so.where(S.fx.wet_c==1.)

    _, _, soi = pyic.interp_to_rectgrid(so.copy(), fpath_ckdtree=S.fpath_ckdtree)
    _, _, toi = pyic.interp_to_rectgrid(to.copy(), fpath_ckdtree=S.fpath_ckdtree)
    S.densi = sw.dens(soi, toi, 0)-1000. 

    if Set.compare_with_reference:
      print('--- loading ref data and deriving bias')
      ds_ref = xr.open_mfdataset(f'{S.fpath_ref}',
        decode_times=False, **Set.mfdset_kwargs
        )
      ds_ref = ds_ref.rename(dict(level='depth'))
      to_ref = ds_ref['T'].isel(time=0)
      to_ref = to_ref.where(S.fx.wet_c==1.)
      so_ref = ds_ref['S'].isel(time=0)
      so_ref = so_ref.where(S.fx.wet_c==1.)
      to += -to_ref
      so += -so_ref

    if nn==0:
      _, _, so_refi = pyic.interp_to_rectgrid(so_ref, fpath_ckdtree=S.fpath_ckdtree)
      _, _, to_refi = pyic.interp_to_rectgrid(to_ref, fpath_ckdtree=S.fpath_ckdtree)
      dens_refi = sw.dens(so_refi, to_refi, 0)-1000. 

    print('--- interpolating to, so fields')
    _, _, S.toi = pyic.interp_to_rectgrid(to, fpath_ckdtree=S.fpath_ckdtree)
    S.dsi['toi'] = xr.DataArray(S.toi, dims=['depth', 'lat', 'lon'], coords=dict(depth=S.depth, lat=lat, lon=lon))
    _, _, S.soi = pyic.interp_to_rectgrid(so, fpath_ckdtree=S.fpath_ckdtree)
    S.dsi['soi'] = xr.DataArray(S.soi, dims=['depth', 'lat', 'lon'], coords=dict(depth=S.depth, lat=lat, lon=lon))

    print('--- loading kv')
    kv = time_ave(S.ds_3d.A_tracer_v_to, S, Set)
    kv = kv.where(kv!=0.)
    _, _, S.kvi = pyic.interp_to_rectgrid(kv, fpath_ckdtree=S.fpath_ckdtree)
    S.dsi['kvi'] = xr.DataArray(S.kvi, dims=['depthi', 'lat', 'lon'], coords=dict(depth=S.depthi, lat=lat, lon=lon))

    print('--- loading tke')
    try:
      tke = time_ave(S.ds_3d.tke, S, Set)
    except:
      tke = time_ave(S.ds_cvmix.tke, S, Set)
    tke = tke.where(tke!=0.)
    _, _, S.tkei = pyic.interp_to_rectgrid(tke, fpath_ckdtree=S.fpath_ckdtree)
    S.dsi['tkei'] = xr.DataArray(S.tkei, dims=['depthi', 'lat', 'lon'], coords=dict(depth=S.depthi, lat=lat, lon=lon))
    
    print('--- mass_flux_vint')
    mass_flux = time_ave(S.ds_3d.mass_flux, S, Set)
    mass_flux = mass_flux.rename(dict(ncells_2='edge'))
    S.mass_flux_vint = mass_flux.sum(axis=0).compute()

    # --- IconData is necessary for bstr
    if 'bstr' in Set.fig_names:
      verbose = True
      fname = f'{S.run}{Set.prfx_monitor}_{Set.tstr}.nc'
      S.IcD = pyic.IconData(
        fname        = fname, 
        path_data    = S.path_data,
        path_grid    = path_grid+S.gname+'/',
        path_ckdtree = path_grid+S.gname+'/ckdtree/',
        fpath_fx     = S.fpath_fx,
        gname        = S.gname,
        lev          = S.lev,
        rgrid_name   = 'global_0.3',
        do_triangulation       = False,
        omit_last_file         = False,
        load_vertical_grid     = False,
        load_triangular_grid   = True, # needed for bstr
        load_rectangular_grid  = False,
        calc_coeff             = False,
        #load_xarray_dset       = load_xarray_dset,
        #xr_chunks              = xr_chunks,
        verbose                = verbose,
      )

# start here for debugging
#if True:
for do_twice in range(Set.ndo_twice):
  if Set.ndo_twice==2:
    if do_twice==0:
      Set.do_diff = False
      Set.fig_sfx = ''
    elif do_twice==1:
      Set.do_diff = True
      Set.fig_sfx = '_diff'
  else: # old behaviour
    Set.fig_sfx = ''

  plt.close("all")
  Title = dict()
  nrow = int(np.ceil(len(Sims)/2)) 

  # ---
  fig_name = 'tab_parameters'
  Title[fig_name] = 'Simulation parameters'
  if fig_name in Set.fig_names:
    txt = tab_parameters_html
    save_tab(txt, fig_name, Set)

  # ---
  fig_name = 'ssh'
  Title[fig_name] = 'Sea surface height'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 #sharex=True, sharey=True,
                                 #xlabel='latitude', ylabel='depth [m]',
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = time_ave(S.ds_2d.zos, S, Set)
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = 2.
        cincr = 0.2
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  SSH [m]', loc='left')
      else:
        data += -data_ref
        clim = 0.2
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'mld_mar'
  Title[fig_name] = 'Mixed layer depth March'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      #data = S.ds_2d.mld.isel(time=S.it_ave_mar).mean(dim='time')
      data = S.ds_2d.mld.isel(time=S.it_mar).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [0, 5000]
        clevs=[0,25,50,100,150,200,300,400,500,750,1000,1500,2000,2500,3000,3500,4000,5000]
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  MLD in Mar [m]', loc='left')
      else:
        data += - data_ref
        clim = 200.
        clevs = np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs, projection=ccrs.PlateCarree())
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'mld_sep'
  Title[fig_name] = 'Mixed layer depth September'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      #data = S.ds_2d.mld.isel(time=S.it_ave_sep).mean(dim='time')
      data = S.ds_2d.mld.isel(time=S.it_sep).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [0, 5000]
        clevs=[0,25,50,100,150,200,300,400,500,750,1000,1500,2000,2500,3000,3500,4000,5000]
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  MLD in Sep [m]', loc='left')
      else:
        data += - data_ref
        clim = 200.
        clevs = np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs, projection=ccrs.PlateCarree())
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ice_concentration_nh_mar'
  Title[fig_name] = 'Sea ice concentration NH Mar'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2,nrow, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1
    for nn, S in enumerate(Sims):
      #conc_mar = S.ds_2d.conc.isel(time=S.it_ave_mar).mean(dim='time')
      conc_mar = S.ds_2d.conc.isel(time=S.it_mar).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      _, _, conc_mar = pyic.interp_to_rectgrid(conc_mar, fpath_ckdtree=S.fpath_ckdtree)
      data = conc_mar[0,:,:]
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim=[0,1]
        clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.
        ax.set_title('  sea ice conc. Mar', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data = data - data_ref
        clim=0.2
        clevs=np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs,
        projection=ccrs.PlateCarree()
      )
    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ice_concentration_nh_sep'
  Title[fig_name] = 'Sea ice concentration NH Sep'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2,nrow, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1
    for nn, S in enumerate(Sims):
      #conc_sep = S.ds_2d.conc.isel(time=S.it_ave_sep).mean(dim='time')
      conc_sep = S.ds_2d.conc.isel(time=S.it_sep).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      _, _, conc_sep = pyic.interp_to_rectgrid(conc_sep, fpath_ckdtree=S.fpath_ckdtree)
      data = conc_sep[0,:,:]
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim=[0,1]
        clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.
        ax.set_title('  sea ice conc. Mar', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data = data - data_ref
        clim=0.2
        clevs=np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs,
        projection=ccrs.PlateCarree()
      )
    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ice_thickness_nh_mar'
  Title[fig_name] = 'Sea ice thickness NH Mar'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2,nrow, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1
    for nn, S in enumerate(Sims):
      #conc_mar = S.ds_2d.conc.isel(time=S.it_ave_mar).mean(dim='time')
      #hi_mar = S.ds_2d.hi.isel(time=S.it_ave_mar).mean(dim='time')
      conc_mar = S.ds_2d.conc.isel(time=S.it_mar).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      hi_mar = S.ds_2d.hi.isel(time=S.it_mar).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      hiconc_mar = (hi_mar*conc_mar)
      _, _, hiconc_mar = pyic.interp_to_rectgrid(hiconc_mar, fpath_ckdtree=S.fpath_ckdtree)
      data = hiconc_mar[0,:,:]
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim=[0,6]
        clevs=[0, 0.01, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
        ax.set_title('  sea ice thickn. Mar', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data = data - data_ref
        clim=1
        clevs=np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs,
        projection=ccrs.PlateCarree()
      )
    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ice_thickness_nh_sep'
  Title[fig_name] = 'Sea ice thickness NH Sep'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2,nrow, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1
    for nn, S in enumerate(Sims):
      #conc_sep = S.ds_2d.conc.isel(time=S.it_ave_sep).mean(dim='time')
      #hi_sep = S.ds_2d.hi.isel(time=S.it_ave_sep).mean(dim='time')
      conc_sep = S.ds_2d.conc.isel(time=S.it_sep).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      hi_sep = S.ds_2d.hi.isel(time=S.it_sep).sel(time=slice(S.t1, S.t2)).mean(dim='time')
      hiconc_sep = (hi_sep*conc_sep)
      _, _, hiconc_sep = pyic.interp_to_rectgrid(hiconc_sep, fpath_ckdtree=S.fpath_ckdtree)
      data = hiconc_sep[0,:,:]
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim=[0,6]
        clevs=[0, 0.01, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6]
        ax.set_title('  sea ice thickn. Mar', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data = data - data_ref
        clim=1.
        clevs=np.linspace(-clim, clim, 21)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs,
        projection=ccrs.PlateCarree()
      )
    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()
    save_fig(fig_name, Set)

  # -------------------------------------------------------------------------------- 
  # Water masses
  # -------------------------------------------------------------------------------- 

  # ---
  fig_name = 'temp_depths_intervals'
  Title[fig_name] = 'Temp. diff. depths'
  if fig_name in Set.fig_names:
    dlevs = np.array([0,250,500,1000,2000,4000])
    a = np.zeros((len(Sims)))
    a[0] = 1; a[-1] = 1
    plot_cb = np.tile(a, (1,dlevs.size-1)).flatten()
    hca, hcb = pyic.arrange_axes(len(Sims), dlevs.size-1, asp=0.5, fig_size_fac=1.5, plot_cb=plot_cb,
                                 sharex=True, sharey=True, projection=ccrs.PlateCarree(),
                                )
    ii=-1
    clims_a = [[-2,32],[-2,16], [-2,10], [-2,8], [-2,4]]
    clims_d = [3, 2, 2, 1, 0.5]
    for kk in range(dlevs.size-1):
      for nn, S in enumerate(Sims):
        ind = (S.depth>=dlevs[kk]) & (S.depth<dlevs[kk+1])
        data = (S.toi*S.dzwi)[ind,:,:].sum(axis=0) / S.dzwi[ind,:,:].sum(axis=0)
        ii+=1; ax=hca[ii]; cax=hcb[ii]
        if nn==0 or Set.do_diff==False:
          data_ref = data
          if Set.compare_with_reference==False:
            clim = clims_a[kk]
            cincr = (clims_a[kk][1]-clims_a[kk][0])/20.
            ax.set_title(f'  temperature [$^o$C]', loc='left')
          else:
            clim = clims_d[kk]
            cincr = clims_d[kk]/10.
            ax.set_title(f'  temperature bias [$^o$C]', loc='left')
          ax.set_title(f'ref: {S.name}', loc='right')
        else:
          data += -data_ref
          clim = clims_d[kk]
          cincr = clims_d[kk]/10.
          ax.set_title(f'{S.name} - ref', loc='right')
        if nn==0:
          ax.set_ylabel(f'{dlevs[kk]}m - {dlevs[kk+1]}m')
        pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr, 
          projection=ccrs.PlateCarree())
    for ax in hca:
      pyic.plot_settings(ax, template='global')
      ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'temp_gzave'
  Title[fig_name] = 'Temp. glob. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = S.toi.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [-2, 30]
          cincr = 2.
          ax.set_title(f'  temp. zon. ave. [$^o$C]', loc='left')
        else:
          clim = 2
          cincr = 0.2
          ax.set_title(f'  temp. bias zon. ave. [$^o$C]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 2
        cincr = 0.2
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xlim(-80, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
      ax.set_xticks(xticks)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'temp_azave'
  Title[fig_name] = 'Temp. Atl. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = 1.*S.toi
      data[:,S.basin_ci!=1] = np.ma.masked
      data = data.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [-2, 30]
          cincr = 2.
          ax.set_title(f'  temp. zon. ave. Atl. [$^o$C]', loc='left')
        else:
          clim = 2
          cincr = 0.2
          ax.set_title(f'  temp. bias zon. ave. Atl. [$^o$C]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 2
        cincr = 0.2
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xticks(np.arange(-90, 100, 30))
      ax.set_xlim(-30, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'temp_ipzave'
  Title[fig_name] = 'Temp. Indo.-Pac. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = 1.*S.toi
      data[:,(S.basin_ci!=3)&(S.basin_ci!=7)] = np.ma.masked
      data = data.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [-2, 30]
          cincr = 2.
          ax.set_title(f'  temp. zon. ave. Indo.-Pac. [$^o$C]', loc='left')
        else:
          clim = 2
          cincr = 0.2
          ax.set_title(f'  temp. bias zon. ave. Indo.-Pac. [$^o$C]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 2
        cincr = 0.2
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xticks(np.arange(-90, 100, 30))
      ax.set_xlim(-30, 65)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'salt_depths_intervals'
  Title[fig_name] = 'Salt. diff. depths'
  if fig_name in Set.fig_names:
    dlevs = np.array([0,250,500,1000,2000,4000])
    a = np.zeros((len(Sims)))
    a[0] = 1; a[-1] = 1
    plot_cb = np.tile(a, (1,dlevs.size-1)).flatten()
    hca, hcb = pyic.arrange_axes(len(Sims), dlevs.size-1, asp=0.5, fig_size_fac=1.5, plot_cb=plot_cb,
                                 sharex=True, sharey=True, projection=ccrs.PlateCarree(),
                                )
    ii=-1
    clims_a = [[33,37],[33,37], [33,37], [34,36], [34,36]]
    clims_d = [1.5, 1, 1, 0.5, 0.25]
    for kk in range(dlevs.size-1):
      for nn, S in enumerate(Sims):
        ind = (S.depth>=dlevs[kk]) & (S.depth<dlevs[kk+1])
        data = (S.soi*S.dzwi)[ind,:,:].sum(axis=0) / S.dzwi[ind,:,:].sum(axis=0)
        ii+=1; ax=hca[ii]; cax=hcb[ii]
        if nn==0 or Set.do_diff==False:
          data_ref = data
          if Set.compare_with_reference==False:
            clim = clims_a[kk]
            cincr = (clims_a[kk][1]-clims_a[kk][0])/20.
            ax.set_title(f'  salinity [psu]', loc='left')
          else:
            clim = clims_d[kk]
            cincr = clims_d[kk]/10.
            ax.set_title(f'  salinity bias [psu]', loc='left')
          ax.set_title(f'ref: {S.name}', loc='right')
        else:
          data += -data_ref
          clim = clims_d[kk]
          cincr = clims_d[kk]/10.
          ax.set_title(f'{S.name} - ref', loc='right')
        if nn==0:
          ax.set_ylabel(f'{dlevs[kk]}m - {dlevs[kk+1]}m')
        pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr, 
          projection=ccrs.PlateCarree())
    for ax in hca:
      pyic.plot_settings(ax, template='global')
      ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'salt_gzave'
  Title[fig_name] = 'Salt. glob. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = S.soi.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [33, 37]
          cincr = 0.5
          ax.set_title(f'  salinity zon. ave. [psu]', loc='left')
        else:
          clim = 0.5
          cincr = clim/10.
          ax.set_title(f'  salinity bias zon. ave. [psu]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 0.5
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xlim(-80, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'salt_azave'
  Title[fig_name] = 'Salt. Atl. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = 1.*S.soi
      data[:,S.basin_ci!=1] = np.ma.masked
      data = data.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [33, 37]
          cincr = 0.5
          ax.set_title(f'  salinity zon. ave. Atl. [psu]', loc='left')
        else:
          clim = 0.5
          cincr = clim/10.
          ax.set_title(f'  salinity bias zon. ave. Atl. [psu]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 0.5
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xticks(np.arange(-90, 100, 30))
      ax.set_xlim(-30, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'salt_ipzave'
  Title[fig_name] = 'Salt. Indo.-Pac. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = 1.*S.soi
      data[:,(S.basin_ci!=3)&(S.basin_ci!=7)] = np.ma.masked
      data = data.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        if Set.compare_with_reference==False:
          clim = [33, 37]
          cincr = 0.5
          ax.set_title(f'  salinity zon. ave. Indo.-Pac. [psu]', loc='left')
        else:
          clim = 0.5
          cincr = clim/10.
          ax.set_title(f'  salinity bias zon. ave. Indo.-Pac. [psu]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 0.5
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
    for ax in hca:
      ax.set_xticks(np.arange(-90, 100, 30))
      ax.set_xlim(-30, 65)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'temp_salt_profiles'
  Title[fig_name] = 'Temp. and salt. profiles'
  lonps = [-30., -30., -30.]
  latps = [-30., 26., 50.]
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(len(lonps), 2, asp=1.2, fig_size_fac=1.5, plot_cb=False,
                                 sharex=False, sharey=False,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    ii=-1

    for kk in range(len(lonps)):
      il = ((lon-lonps[kk])**2).argmin()
      jl = ((lat-latps[kk])**2).argmin()
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        ax.plot(S.toi[:,jl,il], S.depth, label=S.run)
      ax.set_title('temp', loc='left')
      ax.set_title(f'(lon, lat) = ({lonps[kk]},{latps[kk]})', loc='right')

    for kk in range(len(lonps)):
      il = ((lon-lonps[kk])**2).argmin()
      jl = ((lat-latps[kk])**2).argmin()
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        ax.plot(S.soi[:,jl,il], S.depth, label=S.run)
      ax.set_title('salt', loc='left')
      ax.set_title(f'(lon, lat) = ({lonps[kk]},{latps[kk]})', loc='right')

    for ax in hca:
      ax.set_ylim(5000,0)
      ax.grid(True)
      ax.legend()
    save_fig(fig_name, Set)

  # ---
  fig_name = 'temp_salt_hor_ave'
  Title[fig_name] = 'Temp. and salt. hor. averages'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(4, 2, asp=1.2, fig_size_fac=1.5, plot_cb=False,
                                 sharex=False, sharey=False,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    ii=-1

    for nn in range(2):
      if nn==0:
        key = 'toi'
        if Set.compare_with_reference==False:
          vname = 'temp'
        else:
          vname = 'temp bias'
        units = '[$^o$C]'
      elif nn==1:
        key = 'soi'
        if Set.compare_with_reference==False:
          vname = 'salt'
        else:
          vname = 'salt bias'
        units = '[psu]'

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        mask = xr.DataArray(S.basin_ci!=0, dims=['lat', 'lon'])
        data = S.dsi[key].where(mask).mean(dim=['lat', 'lon'])
        ax.plot(data, S.depth, label=S.run)
      ax.set_title(f'{vname} global {units}', loc='left')

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        mask = xr.DataArray(S.basin_ci==1, dims=['lat', 'lon'])
        data = S.dsi[key].where(mask).mean(dim=['lat', 'lon'])
        ax.plot(data, S.depth, label=S.run)
      ax.set_title(f'{vname} Atlantic {units}', loc='left')

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        mask = xr.DataArray(S.basin_ci==3, dims=['lat', 'lon'])
        data = S.dsi[key].where(mask).mean(dim=['lat', 'lon'])
        ax.plot(data, S.depth, label=S.run)
      ax.set_title(f'{vname} Pacific {units}', loc='left')

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      for nn, S in enumerate(Sims):
        mask = xr.DataArray(S.basin_ci==6, dims=['lat', 'lon'])
        data = S.dsi[key].where(mask).mean(dim=['lat', 'lon'])
        ax.plot(data, S.depth, label=S.run)
      ax.set_title(f'{vname} Southern Ocean {units}', loc='left')

    for ax in hca:
      ax.set_ylim(5000,0)
      ax.grid(True)
      ax.legend()
    save_fig(fig_name, Set)

  # ---
  fig_name = 'density_gzave'
  Title[fig_name] = 'Density. glob. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=False,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    data_ref = dens_refi.mean(axis=2)
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      #dens = sw.dens(S.soi, S.toi, 0.)-1000.
      data = S.densi.mean(axis=2)
      #print(S.run)
      #print(data[:,20])
      clev = [26.6, 26.8, 27.0, 27.2, 27.4, 27.6, 27.7, 27.75, 27.8, 27.82, 27.84, 27.86, 27.88, 28.0, 28.2]
      ax.set_title(f'  density zon. ave. [kg/m^3]', loc='left')
      ax.set_title(f'ref: {S.name}', loc='right')
      #pyic.shade(lat, depth, data, ax=ax, cax=cax, clim=[clev[0], clev[-1]], cincr=cincr, colors='k')
      hm = ax.contour(lat, S.depth, data, clev, colors='k')
      Cl = ax.clabel(hm, colors='k', fontsize=6, fmt='%.2f', inline=False)
      for txt in Cl:
        txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))
      hm = ax.contour(lat, Sims[0].depth, data_ref, clev, colors='0.7', linestyles='--')
      Cl = ax.clabel(hm, colors='0.7', fontsize=6, fmt='%.2f', inline=False)
      for txt in Cl:
        txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))
    for ax in hca:
      ax.set_xticks(xticks)
      ax.set_xlim(-80, 90)
      ax.set_ylim(6000, 0)
      #ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # -------------------------------------------------------------------------------- 
  # Vertical mixing
  # -------------------------------------------------------------------------------- 
  # ---
  fig_name = 'kv_depths_intervals'
  Title[fig_name] = 'Vert. diffusivities diff. depths'
  if fig_name in Set.fig_names:
    dlevs = np.array([0,250,500,1000,2000,4000])
    a = np.zeros((len(Sims)))
    a[0] = 1; a[-1] = 1
    plot_cb = np.tile(a, (1,dlevs.size-1)).flatten()
    hca, hcb = pyic.arrange_axes(len(Sims), dlevs.size-1, asp=0.5, fig_size_fac=1.5, plot_cb=plot_cb,
                                 sharex=True, sharey=True, projection=ccrs.PlateCarree(),
                                )
    ii=-1
    clims_a = [[-6,1]]*5
    clims_d = [1, 0.1, 0.01, 0.001, 0.0001]
    for kk in range(dlevs.size-1):
      for nn, S in enumerate(Sims):
        ind = (S.depthi>=dlevs[kk]) & (S.depthi<dlevs[kk+1])
        data = (S.kvi*S.dzti)[ind,:,:].sum(axis=0) / S.dzti[ind,:,:].sum(axis=0)
        ii+=1; ax=hca[ii]; cax=hcb[ii]
        if nn==0 or Set.do_diff==False:
          data_ref = data
          clim = clims_a[kk]
          cincr = (clims_a[kk][1]-clims_a[kk][0])/20.
          ax.set_title(f'  $k_v$ [m$^2$/s$^2$]', loc='left')
          ax.set_title(f'ref: {S.name}', loc='right')
          logplot = True
        else:
          data += -data_ref
          clim = clims_d[kk]
          cincr = clims_d[kk]/10.
          ax.set_title(f'{S.name} - ref', loc='right')
          logplot = False
        if nn==0:
          ax.set_ylabel(f'{dlevs[kk]}m - {dlevs[kk+1]}m')
        pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr, 
          projection=ccrs.PlateCarree(), logplot=logplot)
    for ax in hca:
      pyic.plot_settings(ax, template='global')
      ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'kv_gzave'
  Title[fig_name] = 'Vert. diffusivity glob. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = S.kvi.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [-6,1]
        cincr = 0.2
        logplot = True
        ax.set_title(f'  $k_v$ zon. ave. [m$^2$/s]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 1e-4
        cincr = clim/10.
        logplot = False
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depthi, data, ax=ax, cax=cax, clim=clim, cincr=cincr, logplot=logplot)
    for ax in hca:
      ax.set_xticks(xticks) 
      ax.set_xlim(-80, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'tke_gzave'
  Title[fig_name] = 'TKE glob. zon. ave.'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      data = S.tkei.mean(axis=2)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [-6,-3]
        cincr = 0.15
        logplot = True
        ax.set_title(f'  TKE zon. ave. [m$^2$/s$^2$]', loc='left')
        ax.set_title(f'ref: {S.name}', loc='right')
      else:
        data += - data_ref
        clim = 1e-4
        cincr = clim/10.
        logplot = False
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lat, S.depthi, data, ax=ax, cax=cax, clim=clim, cincr=cincr, logplot=logplot)
    for ax in hca:
      ax.set_xticks(xticks) 
      ax.set_xlim(-80, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

#  # ---
#  fig_name = 'temp_azave'
#  Title[fig_name] = 'Temp. Atl. zon. ave.'
#  if fig_name in Set.fig_names:
#    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
#                                 sharex=True, sharey=True,
#                                 xlabel='latitude', ylabel='depth [m]',
#                                )
#    for nn, S in enumerate(Sims):
#      ax=hca[nn]; cax=hcb[nn]
#      data = 1.*S.toi
#      data[:,S.basin_ci!=1] = np.ma.masked
#      data = data.mean(axis=2)
#      if nn==0 or Set.do_diff==False:
#        data_ref = data
#        if Set.compare_with_reference==False:
#          clim = [-2, 30]
#          cincr = 2.
#          ax.set_title(f'  temp. zon. ave. Atl. [$^o$C]', loc='left')
#        else:
#          clim = 2
#          cincr = 0.2
#          ax.set_title(f'  temp. bias zon. ave. Atl. [$^o$C]', loc='left')
#        ax.set_title(f'ref: {S.name}', loc='right')
#      else:
#        data += - data_ref
#        clim = 2
#        cincr = 0.2
#        ax.set_title(f'{S.name} - ref', loc='right')
#      pyic.shade(lat, depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
#    for ax in hca:
#      ax.set_xticks(np.arange(-90, 90, 30))
#      ax.set_xlim(-30, 90)
#      ax.set_ylim(6000, 0)
#      ax.set_facecolor('0.7')
#    save_fig(fig_name, Set)
#
#  # ---
#  fig_name = 'temp_ipzave'
#  Title[fig_name] = 'Temp. Indo.-Pac. zon. ave.'
#  if fig_name in Set.fig_names:
#    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
#                                 sharex=True, sharey=True,
#                                 xlabel='latitude', ylabel='depth [m]',
#                                )
#    for nn, S in enumerate(Sims):
#      ax=hca[nn]; cax=hcb[nn]
#      data = 1.*S.toi
#      data[:,(S.basin_ci!=3)&(S.basin_ci!=7)] = np.ma.masked
#      data = data.mean(axis=2)
#      if nn==0 or Set.do_diff==False:
#        data_ref = data
#        if Set.compare_with_reference==False:
#          clim = [-2, 30]
#          cincr = 2.
#          ax.set_title(f'  temp. zon. ave. Indo.-Pac. [$^o$C]', loc='left')
#        else:
#          clim = 2
#          cincr = 0.2
#          ax.set_title(f'  temp. bias zon. ave. Indo.-Pac. [$^o$C]', loc='left')
#        ax.set_title(f'ref: {S.name}', loc='right')
#      else:
#        data += - data_ref
#        clim = 2
#        cincr = 0.2
#        ax.set_title(f'{S.name} - ref', loc='right')
#      pyic.shade(lat, depth, data, ax=ax, cax=cax, clim=clim, cincr=cincr)
#    for ax in hca:
#      ax.set_xticks(np.arange(-90, 90, 30))
#      ax.set_xlim(-30, 65)
#      ax.set_ylim(6000, 0)
#      ax.set_facecolor('0.7')
#    save_fig(fig_name, Set)

  # -------------------------------------------------------------------------------- 
  # Transports
  # -------------------------------------------------------------------------------- 

  # ---
  fig_name = 'amoc'
  Title[fig_name] = 'Atlantic MOC'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      moc = S.ds_moc.atlantic_moc
      moc = moc.sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      data = moc.data[:,:,0]*1e-9
      ax=hca[nn]; cax=hcb[nn]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = 24.
        cincr = 2.
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  Atlantic MOC [Sv]', loc='left')
      else:
        data = data-data_ref
        clim = 4
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      hm = pyic.shade(moc.lat, moc.depth, data, ax=ax, cax=cax,
                 clim=clim, cincr=cincr, conts='auto')
      do_clabels(hm[1], ax)
    for ax in hca:
      ax.set_xticks(np.arange(-90,100,30))
      ax.set_xlim(-30, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'pmoc'
  Title[fig_name] = 'Pacific MOC'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      moc = S.ds_moc.pacific_moc
      moc = moc.sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      data = moc.data[:,:,0]*1e-9
      ax=hca[nn]; cax=hcb[nn]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = 24.
        cincr = 2.
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  Pacific MOC [Sv]', loc='left')
      else:
        data = data-data_ref
        clim = 4
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      hm = pyic.shade(moc.lat, moc.depth, data, ax=ax, cax=cax,
                 clim=clim, cincr=cincr, conts='auto')
      do_clabels(hm[1], ax)
    for ax in hca:
      ax.set_xticks(np.arange(-90,100,30))
      ax.set_xlim(-30, 65)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'gmoc'
  Title[fig_name] = 'Global MOC'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 sharex=True, sharey=True,
                                 xlabel='latitude', ylabel='depth [m]',
                                )
    for nn, S in enumerate(Sims):
      moc = S.ds_moc.global_moc
      moc = moc.sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      data = moc.data[:,:,0]*1e-9
      ax=hca[nn]; cax=hcb[nn]
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = 24.
        cincr = 2.
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  Global MOC [Sv]', loc='left')
      else:
        data = data-data_ref
        clim = 4
        cincr = clim/10.
        ax.set_title(f'{S.name} - ref', loc='right')
      hm = pyic.shade(moc.lat, moc.depth, data, ax=ax, cax=cax,
                 clim=clim, cincr=cincr, conts='auto')
      do_clabels(hm[1], ax)
    for ax in hca:
      ax.set_xticks(np.arange(-90,100,30))
      ax.set_xlim(-90, 90)
      ax.set_ylim(6000, 0)
      ax.set_facecolor('0.7')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'moc_profile'
  Title[fig_name] = 'MOC profiles'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(3, 1, asp=1.5, fig_size_fac=3, plot_cb=False, sharex=False)
    ax = hca[0]
    for nn, S in enumerate(Sims):
      moc_prof = S.ds_moc.atlantic_moc.sel(lat=26, method='nearest').sel(time=slice(S.t1, S.t2)).mean(dim='time')*1e-9
      ax.plot(moc_prof.data, moc_prof.depth, label=S.name)
    ax.legend()
    ax.set_title('  AMOC', loc='left')
    ax.set_title(f'y = {moc_prof.lat.data}', loc='right')
    ax = hca[1]
    for nn, S in enumerate(Sims):
      moc_prof = S.ds_moc.atlantic_moc.sel(lat=50, method='nearest').sel(time=slice(S.t1, S.t2)).mean(dim='time')*1e-9
      ax.plot(moc_prof.data, moc_prof.depth, label=S.name)
    ax.legend()
    ax.set_title('  AMOC', loc='left')
    ax.set_title(f'y = {moc_prof.lat.data}', loc='right')
    ax = hca[2]
    for nn, S in enumerate(Sims):
      moc_prof = S.ds_moc.pacific_moc.sel(lat=-10, method='nearest').sel(time=slice(S.t1, S.t2)).mean(dim='time')*1e-9
      ax.plot(moc_prof.data, moc_prof.depth, label=S.name)
    ax.legend()
    ax.set_title('  PMOC', loc='left')
    ax.set_title(f'y = {moc_prof.lat.data}', loc='right')
    for ax in hca:
      ax.set_ylim(6000,0)
      ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ts_amoc'
  Title[fig_name] = 'TS: AMOC 26N'
  var = 'amoc26n'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(1, 1, asp=0.4, fig_size_fac=1.5, plot_cb=False)
    ax = hca[0]
    for nn, S in enumerate(Sims):
      data = S.ds_ts[var].groupby('time.year').mean()
      data.plot(ax=ax, label=S.name, marker='.')
    ax.legend()
    ax.set_xlabel('time [years]')
    ax.set_ylabel('')
    ax.set_title(f'{Title[fig_name]} [{S.ds_ts[var].units}]')
    ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ts_kin'
  Title[fig_name] = 'TS: kin. energy'
  var = 'kin_energy_global'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(1, 1, asp=0.4, fig_size_fac=1.5, plot_cb=False)
    ax = hca[0]
    for nn, S in enumerate(Sims):
      data = S.ds_ts[var].groupby('time.year').mean()
      data.plot(ax=ax, label=S.name, marker='.')
    ax.legend()
    ax.set_xlabel('time [years]')
    ax.set_ylabel('')
    ax.set_title(f'{Title[fig_name]} [{S.ds_ts[var].units}]')
    ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ts_global_heat_content'
  Title[fig_name] = 'TS: global heat content'
  var = 'global_heat_content'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(1, 1, asp=0.4, fig_size_fac=1.5, plot_cb=False)
    ax = hca[0]
    for nn, S in enumerate(Sims):
      data = S.ds_ts[var].groupby('time.year').mean()
      data.plot(ax=ax, label=S.name, marker='.')
    ax.legend()
    ax.set_xlabel('time [years]')
    ax.set_ylabel('')
    ax.set_title(f'{Title[fig_name]} [{S.ds_ts[var].units}]')
    ax.grid(True)
    save_fig(fig_name, Set)

  # ---
  fig_name = 'bstr'
  Title[fig_name] = 'Barotropic streamfunction'
  if fig_name in Set.fig_names:
    print('start bstr')
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      print(f'--- {S.run}')
      data = pyic.calc_bstr_vgrid(S.IcD, S.mass_flux_vint.compute().data, 
                                  lon_start=0., lat_start=90.,
                                  verbose=True, old_version=True)
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree, 
                                           coordinates='vlat vlon')
      if nn==0 or Set.do_diff==False:
        data_ref = data
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  barotr. streamf. [Sv]', loc='left')
        clim = 200
        clevs = [-200,-160,-120,-80,-40,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,40,80,120,160,200]
      else:
        clim = 50
        clevs = np.arange(-clim, clim+5., 5.)
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, clevs=clevs)
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'htr'
  Title[fig_name] = 'Heat transport'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(1, 3, asp=0.4, fig_size_fac=1.5, plot_cb=False, sharex=False)
    ax = hca[0]
    for nn, S in enumerate(Sims):
      hfbasin = S.ds_moc.atlantic_hfbasin.sel(time=slice(S.t1, S.t2)).mean(dim='time')/1e15
      hfbasin.plot(label=S.name, ax=ax)
    ax.legend()
    ax.set_title('Atlantic heat transport [PW]')
    ax = hca[1]
    for nn, S in enumerate(Sims):
      hfbasin = S.ds_moc.pacific_hfbasin.sel(time=slice(S.t1, S.t2)).mean(dim='time')/1e15
      hfbasin.plot(label=S.name, ax=ax)
    ax.legend()
    ax.set_title('Pacific heat transport [PW]')
    ax = hca[2]
    for nn, S in enumerate(Sims):
      hfbasin = S.ds_moc.global_hfbasin.sel(time=slice(S.t1, S.t2)).mean(dim='time')/1e15
      hfbasin.plot(label=S.name, ax=ax)
    ax.legend()
    ax.set_title('Global heat transport [PW]')
    for ax in hca:
      ax.set_xticks(xticks)
      ax.set_xlim(-90, 90)
      ax.grid(True)
      ax.set_xlabel('latitude')
      ax.set_ylabel('')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ke100m'
  Title[fig_name] = 'Kin. energy 100m'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 #sharex=True, sharey=True,
                                 #xlabel='latitude', ylabel='depth [m]',
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      iz = ((S.depth-100)**2).argmin()
      u = S.ds_3d.u.isel(depth=iz).sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      v = S.ds_3d.v.isel(depth=iz).sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      data = 0.5*(u**2+v**2)
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [-7, 0]
        cincr = 0.5
        logplot = True
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  kin. energy 100m [m$^2$/s$^2$]', loc='left')
      else:
        data += -data_ref
        clim = 0.1
        cincr = clim/10.
        logplot = False
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr, logplot=logplot)
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'ke2000m'
  Title[fig_name] = 'Kin. energy 2000m'
  if fig_name in Set.fig_names:
    hca, hcb = pyic.arrange_axes(2, nrow, asp=0.5, fig_size_fac=1.5, plot_cb=True,
                                 #sharex=True, sharey=True,
                                 #xlabel='latitude', ylabel='depth [m]',
                                 projection=ccrs.PlateCarree(),
                                )
    for nn, S in enumerate(Sims):
      ax=hca[nn]; cax=hcb[nn]
      iz = ((S.depth-2000)**2).argmin()
      u = S.ds_3d.u.isel(depth=iz).sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      v = S.ds_3d.v.isel(depth=iz).sel(time=slice(S.t1, S.t2)).mean(dim='time') 
      data = 0.5*(u**2+v**2)
      _, _, data = pyic.interp_to_rectgrid(data, fpath_ckdtree=S.fpath_ckdtree)
      if nn==0 or Set.do_diff==False:
        data_ref = data
        clim = [-7, -2]
        cincr = 0.5
        logplot = True
        ax.set_title(f'ref: {S.name}', loc='right')
        ax.set_title(f'  kin. energy 2000m [m$^2$/s$^2$]', loc='left')
      else:
        data += -data_ref
        clim = 0.1
        cincr = clim/10.
        logplot = False
        ax.set_title(f'{S.name} - ref', loc='right')
      pyic.shade(lon, lat, data, ax=ax, cax=cax, clim=clim, cincr=cincr, logplot=logplot)
    for ax in hca:
      pyic.plot_settings(ax=ax, template='global')
    save_fig(fig_name, Set)

  # ---
  fig_name = 'tab_transport_sections'
  Title[fig_name] = 'Transport through sections'
  if fig_name in Set.fig_names:
    f = Dataset(f'{path_grid}/{S.gname}/section_mask_{S.gname}.nc', 'r')
    snames = []
    for var in f.variables.keys():
      if var.startswith('mask'):
        snames += [var[5:]]
    f.close()
    df_transp = pd.DataFrame(index=snames, columns=runs)
    for nn, S in enumerate(Sims):
      f = Dataset(f'{path_grid}/{S.gname}/section_mask_{S.gname}.nc', 'r')
      edge_length = S.tgrid.edge_length.compute()
      for var in snames:
        #print(f'{S.run}: {var}')
        mask = f.variables['mask_'+var][:]
        df_transp.loc[var, S.run] = ((S.mass_flux_vint*edge_length*mask).sum()/1e6).compute().data
      f.close()
    txt = df_transp.to_html()
    save_tab(txt, fig_name, Set)

# --- path to module
path_qp_driver = os.path.dirname(pyicqp.__file__)+'/'

# --- copy css style file
shutil.copyfile(path_qp_driver+'qp_css.css', Set.path_base+'qp_css.css')

# --- backup qp_driver (this script)
fname_this_script = __file__.split('/')[-1]
shutil.copyfile(path_qp_driver+fname_this_script, Set.path_base+'bcp_'+fname_this_script)

# --- make web page
string = 'Compare: '
for S in Sims:
  string += f'{S.name}, '
string = string[:-2]
qp = pyicqp.QuickPlotWebsite(
  title=f'{Set.name}', 
  author=os.environ.get('USER'),
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  path_data=string,
  #info=f'time average from {t1} to {t2}',
  fpath_css='./qp_css.css',
  fpath_html=Set.path_base+'index.html'
  )

for fig_name in Set.plot_names:
  print(f'Linking {fig_name}')
  #fname = fig_name+'.png'
  #fpath_fig = Set.path_pics+fname
  #tname = fig_name+'.html'
  #fpath_tab = Set.path_pics+tname
  if fig_name.startswith('sec'):
    qp.add_section(fig_name.split(':')[1])
  elif os.path.exists(Set.path_pics+fig_name+'.html'): 
     qp.add_subsection(Title[fig_name])
     qp.add_html(f'{Set.path_base}pics/{fig_name}.html')
  elif os.path.exists(Set.path_pics+fig_name+'.png'): 
     qp.add_subsection(Title[fig_name])
     qp.add_fig(f'pics/{fig_name}.png')
  else:
    print(f'::: Warning: {fig_name} could not be found.:::')
qp.write_to_file()

#plt.show()
