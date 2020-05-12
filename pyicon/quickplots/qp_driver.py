import sys, glob, os
if len(sys.argv)==2 and sys.argv=='--slurm':
  matplotlib.use('Agg')
import shutil
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from ipdb import set_trace as mybreak
#from importlib import reload
import json
#sys.path.append('/home/mpim/m300602/pyicon')
import pyicon as pyic
import pyicon.quickplots as pyicqp
#reload(pyic)
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from qp_cmaps import PyicCmaps

#cm_wbgyr = PyicCmaps().WhiteBlueGreenYellowRed

runname = ''
oce_def = ''
oce_moc = '_MOC'
oce_mon = '_oceanMonitor'
oce_monthly = oce_def
do_atmosphere_plots = False
do_ocean_plots = True

t1 = 'auto'
t2 = 'auto'

# ---
#runname = 'icon_08'
#run = 'nib0002'
#tstep = '20020101T000000Z'
#
#path_data    = '/Users/nbruggemann/work/icon_playground/icon_r2b4_test_data/icon_08/icon-oes/experiments/nib0002/'
#path_ckdtree = '/Users/nbruggemann/work/icon_playground/icon_ckdtree/'

# ------ r2b4
#exec(open("../../config_qp/conf-icon_08-nib002.py").read())
#exec(open("../../config_qp/conf-mac-icon_08-nib0002.py").read())

# ------ r2b6
#exec(open("../../config_qp/conf-icon_08-nib003.py").read())
# does not work so far. diff. grid file
#exec(open("../../config_qp/conf-ocean_omip_long_r2b6_19224-YVF.py").read()) 

# ------ r2b8
#exec(open("../../config_qp/conf-ocean_era51h_r2b8_19074-AMK.py").read())
#exec(open("../../config_qp/conf-exp.ocean_ncep6h_r2b8_20096-FZA.py").read())

# ------ r2b6
#exec(open("../../config_qp/conf-icon_08-nib0004.py").read())
#exec(open("../../config_qp/conf-icon_08-nib0006.py").read())
exec(open("../../config_qp/conf-jkr0042.py").read())

fpath_ref_atm = '/mnt/lustre01/work/mh0033/m300602/icon/era/pyicon_prepare_era.nc'

# -------------------------------------------------------------------------------- 
# Settings
# -------------------------------------------------------------------------------- 
projection = 'PlateCarree'
#projection = 'none'

# --- structure of web page
fig_names = []
fig_names += ['sec:Upper ocean']
fig_names += ['ssh', 'ssh_variance', 'sst', 'sss', 'mld_mar', 'mld_sep'] 
fig_names += ['sec:Ice']
fig_names += ['ice_concentration_nh', 'ice_thickness_nh', 'snow_thickness_nh',] 
fig_names += ['ice_concentration_sh', 'ice_thickness_sh', 'snow_thickness_sh',]
fig_names += ['sec:Sections']
fig_names += ['temp30w', 'salt30w', 'dens30w']
fig_names += ['sec:Zonal averages']
fig_names += ['temp_gzave', 'temp_azave', 'temp_ipzave']
fig_names += ['salt_gzave', 'salt_azave', 'salt_ipzave']
fig_names += ['sec:Transports']
fig_names += ['bstr', 'amoc', 'pmoc', 'gmoc']
fig_names += ['ke_100m', 'ke_2000m']
fig_names += ['heat_flux', 'freshwater_flux']
fig_names += ['sec:Biases']
fig_names += ['sst_bias', 'temp_bias_gzave', 'temp_bias_azave', 'temp_bias_ipzave']
fig_names += ['sss_bias', 'salt_bias_gzave', 'salt_bias_azave', 'salt_bias_ipzave']
fig_names += ['sec:Time series']
fig_names += ['ts_amoc', 'ts_heat_content', 'ts_ssh', 'ts_sst', 'ts_sss', 'ts_hfl', 'ts_wfl', 'ts_ice_volume_nh', 'ts_ice_volume_sh', 'ts_ice_extent',]
fig_names += ['ts_tas_gmean', 'ts_radtop_gmean']
fig_names += ['sec:Atmosphere surface']
fig_names += ['atm_2m_temp','atm_surface_temp','atm_sea_level_pressure',]
fig_names += ['atm_zonal_wind_stress', 'atm_meridional_wind_stress']
fig_names += ['atm_column_water_vapour', 'atm_total_precipitation', 'atm_total_cloud_cover', 'atm_p_e', 'atm_10m_wind']
fig_names += ['sec:Atmosphere surface bias']
fig_names += ['atm_tas_bias']
fig_names += ['atm_prw_bias']
fig_names += ['atm_psl_bias']
fig_names += ['atm_tauu_bias']
fig_names += ['atm_tauv_bias']
fig_names += ['sec:Atmosphere zonal averages']
fig_names += ['atm_temp_zave', 'atm_u_zave', 'atm_v_zave', 'atm_rel_hum_zave']
fig_names += ['atm_cloud_cover_zave', 'atm_cloud_water_zave', 'atm_cloud_ice_zave', 'atm_cloud_water_ice_zave', 'atm_psi']
#fig_names += ['sec:TKE and IDEMIX']
#fig_names += ['tke30w', 'iwe30w', 'kv30w']

plist = fig_names
fig_names = []
for pitem in plist:
  if not pitem.startswith('sec:'):
    fig_names += [pitem]

# --- for debugging
#fig_names = []
#fig_names += ['atm_psi']
#fig_names += ['ts_tas_gmean']
#fig_names += ['sst']
#fig_names += ['ts_amoc']
#fig_names += ['ts_amoc', 'ts_heat_content', 'ts_ssh', 'ts_sst', 'ts_sss', 'ts_hfl', 'ts_wfl', 'ts_ice_volume_nh', 'ts_ice_volume_sh', 'ts_ice_extent',]
#fig_names += ['mld_mar', 'mld_sep']
#fig_names = ['temp_bias_gzave']
#fig_names = ['sss']
#fig_names = ['ssh_variance']
#fig_names += ['amoc']
#fig_names += ['sst_bias', 'temp_bias_gzave', 'temp_bias_azave', 'temp_bias_ipzave']
#fig_names += ['sss_bias', 'salt_bias_gzave', 'salt_bias_azave', 'salt_bias_ipzave']
#fig_names += ['ice_concentration_nh', 'ice_thickness_nh', 'snow_thickness_nh',] 
#fig_names += ['ice_concentration_sh', 'ice_thickness_sh', 'snow_thickness_sh',]
#fig_names += ['bstr']
#fig_names += ['ke_100m', 'ke_2000m']
#fig_names += ['atm_sea_level_pressure']
#fig_names += ['atm_2m_temp','atm_sea_level_pressure',]
#fig_names += ['atm_zonal_wind_stress',]
#fig_names += ['atm_column_water_vapour', 'atm_total_precipitation', 'atm_total_cloud_cover']
#fig_names += ['atm_tas_bias']
#fig_names += ['atm_temp_zave', 'atm_u_zave', 'atm_v_zave', 'atm_rel_hum_zave']
#fig_names += ['atm_cloud_cover_zave', 'atm_cloud_water_zave', 'atm_cloud_ice_zave', 'atm_cloud_water_ice_zave']
#fig_names += ['atm_cloud_water_zave', 'atm_cloud_ice_zave', 'atm_cloud_water_ice_zave']
#fig_names += ['vort']
#fig_names += ['np_zonal_wind_stress']


fig_names = np.array(fig_names)

# -------------------------------------------------------------------------------- 
# Function to save figures
# -------------------------------------------------------------------------------- 
close_figs = True
def save_fig(title, path_pics, fig_name, FigInf=dict()):
  FigInf['name']  = fig_name+'.png'
  FigInf['fpath'] = path_pics+FigInf['name']
  FigInf['title'] = title
  print('Saving {:<40} {:<40}'.format(FigInf['fpath'], FigInf['title']))
  plt.savefig(FigInf['fpath'], dpi=300)
  with open(path_pics+fig_name+'.json', 'w') as fj:
    json.dump(FigInf, fj, sort_keys=True, indent=4)
  #if close_figs:
  #  plt.close('all')
  return
plt.close('all')

#rgrid_name = 'global_1.0'
rgrid_name = 'global_0.3'
rgrid_name_atm = 'global_1.0_era'

def indfind(array, vals):
  vals = np.array(vals)
  inds = np.zeros(vals.size, dtype=int)
  for nn, val in enumerate(vals):
    inds[nn] = np.argmin(np.abs(array-val))
  return inds

def load_era3d_var(fpath, var, isort):
  f = Dataset(fpath, 'r')
  data = f.variables[var][:].mean(axis=0)
  f.close()
  data = data[:,:,isort]
  return data

# -------------------------------------------------------------------------------- 
# Load all necessary data sets (can take a while)
# -------------------------------------------------------------------------------- 
if do_ocean_plots:
  fname = '%s%s_%s.nc' % (run, oce_def, tstep)
  print('Dataset %s' % (fname))
  IcD = pyic.IconData(
                 fname        = fname,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 #path_ckdtree = path_ckdtree,
                 #path_tgrid   = path_tgrid,
                 #fpath_tgrid  = fpath_tgrid,
                 #fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name,
                 do_triangulation = False,
                 omit_last_file = False,
                )
  fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
  [k100, k500, k800, k1000, k2000, k3000] = indfind(IcD.depthc, [100., 500., 800., 1000., 2000., 3000.])
  
  fname_moc = '%s%s_%s.nc' % (run, oce_moc, tstep)
  print('Dataset %s' % (fname_moc))
  IcD_moc = pyic.IconData(
                 fname        = fname_moc,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 #path_ckdtree = path_ckdtree,
                 #path_tgrid   = path_tgrid,
                 #fpath_tgrid  = fpath_tgrid,
                 #fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name,
                 do_triangulation = False,
                 omit_last_file = False,
                )

  fname_monthly = '%s%s_%s.nc' % (run, oce_monthly, tstep)
  print('Dataset %s' % (fname_monthly))
  IcD_monthly = pyic.IconData(
                 fname        = fname_monthly,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 #path_ckdtree = path_ckdtree,
                 #path_tgrid   = path_tgrid,
                 #fpath_tgrid  = fpath_tgrid,
                 #fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name,
                 do_triangulation = True,
                 omit_last_file = False,
                )

if do_atmosphere_plots:
  fname = '%s%s_%s.nc' % (run, atm_2d, tstep)
  print('Dataset %s' % (fname))
  IcD_atm2d = pyic.IconData(
                 fname        = fname,
                 path_data    = path_data,
                 path_grid    = path_grid_atm,
                 gname        = gname_atm,
                 lev          = lev_atm,
                 #path_ckdtree = path_ckdtree,
                 #path_tgrid   = path_tgrid,
                 #fpath_tgrid  = fpath_tgrid,
                 #fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name_atm,
                 do_triangulation = True,
                 omit_last_file = False,
                 load_vertical_grid = False,
                 time_mode = 'float2date',
                )

  fname = '%s%s_%s.nc' % (run, atm_3d, tstep)
  print('Dataset %s' % (fname))
  IcD_atm3d = pyic.IconData(
                 fname        = fname,
                 path_data    = path_data,
                 path_grid    = path_grid_atm,
                 gname        = gname_atm,
                 lev          = lev_atm,
                 #path_ckdtree = path_ckdtree,
                 #path_tgrid   = path_tgrid,
                 #fpath_tgrid  = fpath_tgrid,
                 #fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name_atm,
                 do_triangulation = True,
                 omit_last_file = False,
                 load_vertical_grid = False,
                 model_type   = 'atm',
                 time_mode = 'float2date',
                )
  fpath_ckdtree_atm = IcD_atm3d.rgrid_fpath_dict[rgrid_name_atm]
  
print('Done reading datasets')

# -------------------------------------------------------------------------------- 
# timing
# -------------------------------------------------------------------------------- 
for tave_int in tave_ints:
  t1 = tave_int[0]
  t2 = tave_int[1]

  if isinstance(t1,str) and t1=='auto':
    t1 = IcD.times[0]
  else:
    t1 = np.datetime64(t1)
  if isinstance(t2,str) and t2=='auto':
    t2 = IcD.times[-1]
  else:
    t2 = np.datetime64(t2)

  it_ave = np.where( (IcD_monthly.times>=t1) & (IcD_monthly.times<=t2) )[0]
  it_ave_mar = it_ave[2::12]
  it_ave_sep = it_ave[8::12]
  print('ave_mar: ', IcD_monthly.times[it_ave_mar])
  print('ave_sep: ', IcD_monthly.times[it_ave_sep])
  # -------------------------------------------------------------------------------- 
  # making new directories and copying files
  # -------------------------------------------------------------------------------- 
  if runname=='':
    dirname = f'qp-{run}'
  else:
    dirname = f'qp-{runname}-{run}'
  path_base = f'../../all_qps/{dirname}/'
  tave_str = f'tave_{t1}-{t2}/'
  path_qp = f'{path_base}/{tave_str}/'
  if not os.path.exists(path_qp):
    os.makedirs(path_qp)
  
  rpath_pics = './pics/'
  path_pics = path_qp+rpath_pics
  if not os.path.exists(path_pics):
    os.makedirs(path_pics)
  
  shutil.copyfile('./qp_css.css', path_qp+'qp_css.css')
  
  if runname=='':
    tstr = run
  else:
    tstr = '%s | %s' % (runname, run)
  qp = pyicqp.QuickPlotWebsite(
    title=tstr, 
    author=os.environ.get('USER'),
    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    path_data=path_data,
    info=f'time average from {t1} to {t2}',
    fpath_css='./qp_css.css',
    fpath_html=path_qp+'qp_index.html'
    )
  
  fname_this_script = __file__.split('/')[-1]
  shutil.copyfile(fname_this_script, path_qp+'bcp_'+fname_this_script)
  
  # -------------------------------------------------------------------------------- 
  # upper ocean
  # -------------------------------------------------------------------------------- 
  fname = '%s%s_%s.nc' % (run, oce_def, tstep)
  Ddict = dict(
    xlim=[-180.,180.], ylim=[-90.,90.],
    rgrid_name=rgrid_name,
    path_ckdtree=path_ckdtree,
    projection=projection,
              )
  
  # ---
  fig_name = 'mld_mar'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='mld', depth=0,
                             it_ave=it_ave_mar,
                             title='mixed layer depth March [m]',
                             #clim=[0,5000.], cincr=250., cmap=PyicCmaps().WhiteBlueGreenYellowRed,
                             clim=[0,5000.], cincr=250., cmap='RdYlBu_r',
                             do_mask=True,
                             IcD=IcD_monthly, **Ddict)
    save_fig('Mixed layer depth March', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'mld_sep'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='mld', depth=0,
                             it_ave=it_ave_sep,
                             title='mixed layer depth September [m]',
                             clim=[0,5000.], cincr=250., cmap='RdYlBu_r',
                             do_mask=True,
                             IcD=IcD_monthly, **Ddict)
    save_fig('Mixed layer depth September', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'sst'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='to', depth=0, it=0,
                             t1=t1, t2=t2,
                             clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                             IcD=IcD, **Ddict)
    #save_fig('Sea surface temperature', path_pics, fig_name, FigInf)
    save_fig('SST', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'sss'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='so', depth=0, it=0,
                             t1=t1, t2=t2,
                             #clim=[32.,37], cincr=0.25, cmap='cmo.haline',
                             clim=[25.,40.], clevs=[25, 28, 30, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 37, 38, 40], cmap='cmo.haline',
                             IcD=IcD, **Ddict)
    #save_fig('Sea surface salinity', path_pics, fig_name, FigInf)
    save_fig('SSS', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'ssh'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='zos', depth=0, it=0,
                             t1=t1, t2=t2,
                             clim=2, cincr=0.2, cmap='RdBu_r',
                             IcD=IcD, **Ddict)
    #save_fig('Sea surface height', path_pics, fig_name, FigInf)
    save_fig('SSH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'ssh_variance'
  if fig_name in fig_names:
    zos, it_ave          = pyic.time_average(IcD, 'zos',        t1=t1, t2=t2)
    zos_square, it_ave   = pyic.time_average(IcD, 'zos_square', t1=t1, t2=t2)
    zos_var = np.sqrt(zos_square-zos**2)
    IaV = pyic.IconVariable('zos_var', 'm', 'ssh variance')
    IaV.data = zos_var
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD, IaV, clim=[-2,0], cincr=0.2, cmap='RdYlBu_r',
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                    logplot=True,
                    title='log$_{10}$(ssh variance) [m]', do_write_data_range=True,
                    )
    save_fig('SSH variance', path_pics, fig_name)
  
  # ---
  fig_name = 'ice_concentration_nh'
  if fig_name in fig_names:
    #proj_1 = 'NorthPolarStereo'
    #proj_2 = 'SouthPolarStereo'

    conc_mar, it_ave = pyic.time_average(IcD_monthly, 'conc', it_ave=it_ave_mar, iz='all')
    hi_mar, it_ave   = pyic.time_average(IcD_monthly, 'hi', it_ave=it_ave_mar, iz='all')
    hs_mar, it_ave   = pyic.time_average(IcD_monthly, 'hs', it_ave=it_ave_mar, iz='all')
    hiconc_mar = (hi_mar*conc_mar)[0,:]
    hsconc_mar = (hs_mar*conc_mar)[0,:]

    conc_sep, it_ave = pyic.time_average(IcD_monthly, 'conc', it_ave=it_ave_sep, iz='all')
    hi_sep, it_ave   = pyic.time_average(IcD_monthly, 'hi', it_ave=it_ave_sep, iz='all')
    hs_sep, it_ave   = pyic.time_average(IcD_monthly, 'hs', it_ave=it_ave_sep, iz='all')
    hiconc_sep = (hi_sep*conc_sep)[0,:]
    hsconc_sep = (hs_sep*conc_sep)[0,:]

  # ---
  fig_name = 'ice_concentration_nh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('conc_mar', '', 'sea ice concentration March')
    IaV.data = conc_mar[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('conc_sep', 'm', 'sea ice concentration September')
    IaV.data = conc_sep[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sea ice equiv. thickness NH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'ice_thickness_nh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hiconc_mar', 'm', 'ice equiv. thickness March')
    IaV.data = hiconc_mar
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], cincr=0.5, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hiconc_sep', 'm', 'ice equiv. thickness September')
    IaV.data = hiconc_sep
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], cincr=0.5, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sea ice equiv. thickness NH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'snow_thickness_nh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.NorthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hsconc_mar', 'm', 'snow equiv. thickness March')
    IaV.data = hsconc_mar
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hsconc_sep', 'm', 'snow equiv. thickness September')
    IaV.data = hsconc_sep
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Snow equiv. thickness NH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'ice_concentration_sh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.SouthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('conc_mar', '', 'sea ice concentration March')
    IaV.data = conc_mar[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('conc_sep', 'm', 'sea ice concentration September')
    IaV.data = conc_sep[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Ice concentration SH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'ice_thickness_sh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.SouthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hiconc_mar', 'm', 'ice equiv. thickness March')
    IaV.data = hiconc_mar
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], clevs=[0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hiconc_sep', 'm', 'ice equiv. thickness September')
    IaV.data = hiconc_sep
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], clevs=[0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sea ice equiv. thickness SH', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'snow_thickness_sh'
  if fig_name in fig_names:
    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="", ylabel="",
                 projection=ccrs.SouthPolarStereo(),
                                )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hsconc_mar', 'm', 'snow equiv. thickness March')
    IaV.data = hsconc_mar
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IaV = pyic.IconVariable('hsconc_sep', 'm', 'snow equiv. thickness September')
    IaV.data = hsconc_sep
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='cmo.ice_r',
                    projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                    ax=ax, cax=cax,
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )

    for ax in hca:
      ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
      ax.gridlines()
      ax.add_feature(cartopy.feature.LAND)
      ax.coastlines()

    FigInf = dict(long_name=IaV.long_name)
    save_fig('Snow equiv. thickness SH', path_pics, fig_name, FigInf)

#  # ---
#  fig_name = 'ice_thickness_sh'
#  if fig_name in fig_names:
#    hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
#                 sharex=True, sharey=True, xlabel="", ylabel="",
#                 projection=ccrs.SouthPolarStereo()
#                                )
#    ii=-1
#
#    ii+=1; ax=hca[ii]; cax=hcb[ii]
#    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='hi', depth=0, it=0,
#                           t1=t1, t2=t2,
#                           it_ave=it_ave_mar,
#                           title='sea ice thickness March [m]',
#                           clim=[0,6], cincr=0.5, cmap='cmo.ice_r',
#                           xlim=[-180, 180], ylim=[-90,-50],
#                           #use_tgrid=True,
#                           ax=ax, cax=cax,
#                           IcD=IcD_monthly,
#                           rgrid_name=rgrid_name,
#                           path_ckdtree=path_ckdtree,
#                           projection='PlateCarree',
#                           crs_features=False,
#                           do_plot_settings=False,
#                          )
#  
#    ii+=1; ax=hca[ii]; cax=hcb[ii]
#    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='hi', depth=0, it=0,
#                           t1=t1, t2=t2,
#                           it_ave=it_ave_sep,
#                           title='sea ice thickness September [m]',
#                           clim=[0,3], cincr=0.25, cmap='cmo.ice_r',
#                           xlim=[-180, 180], ylim=[-90,-50],
#                           #use_tgrid=True,
#                           ax=ax, cax=cax,
#                           IcD=IcD_monthly,
#                           rgrid_name=rgrid_name,
#                           path_ckdtree=path_ckdtree,
#                           projection='PlateCarree',
#                           crs_features=False,
#                           do_plot_settings=False,
#                          )
#  
#    for ax in hca:
#      ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())
#      ax.gridlines()
#      ax.add_feature(cartopy.feature.LAND)
#      ax.coastlines()
#    
#    save_fig('Sea ice thickness SH', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # biases
  # -------------------------------------------------------------------------------- 
  fname = '%s%s_%s.nc' % (run, oce_def, tstep)
  calc_bias = False
  for fig_name in fig_names:
    if '_bias' in fig_name: 
      calc_bias = True
  if calc_bias:
    fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
  
    f = Dataset(fpath_initial_state, 'r')
    temp_ref = f.variables['T'][0,:,:]
    salt_ref = f.variables['S'][0,:,:]
    f.close()
    temp_ref[temp_ref==0.]=np.ma.masked
    salt_ref[salt_ref==0.]=np.ma.masked
  
    #f = Dataset(path_data+fname, 'r')
    #temp = f.variables['to'][:,:,:].mean(axis=0)
    #salt = f.variables['so'][:,:,:].mean(axis=0)
    #f.close()
    #temp[temp==0.]=np.ma.masked
    #salt[salt==0.]=np.ma.masked
    temp, it_ave = pyic.time_average(IcD, 'to', t1, t2, iz='all')
    salt, it_ave = pyic.time_average(IcD, 'so', t1, t2, iz='all')
    temp[temp==0.]=np.ma.masked
    salt[salt==0.]=np.ma.masked
  
    tbias = temp-temp_ref
    sbias = salt-salt_ref
  
  # ---
  fig_name = 'sst_bias'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
    IaV.data = tbias[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD, IaV, cmap='RdBu_r',
                    clim=10, clevs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10], do_write_data_range=True,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Tbias: surface', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'temp_bias_gzave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='global', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                    clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                    asp=0.5, do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Tbias: global zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'temp_bias_azave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='atl', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                    clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                    asp=0.5, xlim=[-30,90], do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Tbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'temp_bias_ipzave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='indopac', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                    clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                    asp=0.5, xlim=[-30,65], do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Tbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'sss_bias'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
    IaV.data = sbias[0,:]
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD, IaV, clim=3., cmap='RdBu_r', cincr=0.2, do_write_data_range=True,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sbias: surface', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_bias_gzave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='global', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, clim=2., cmap='RdBu_r', cincr=0.2, contfs='auto',
                    asp=0.5, do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sbias: global zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_bias_azave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='atl', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, clim=2., cmap='RdBu_r', cincr=0.2, contfs='auto',
                    asp=0.5, xlim=[-30,90], do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_bias_ipzave'
  if fig_name in fig_names:
    IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='indopac', 
                               fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
    pyic.vplot_base(IcD, IaV, clim=2., cmap='RdBu_r', cincr=0.2, contfs='auto',
                    asp=0.5, xlim=[-30,65], do_write_data_range=True)
    FigInf = dict(long_name=IaV.long_name)
    save_fig('Sbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
  
  ## ---
  #fname = '%s_restart_oce_%s.nc' % (run, tstep)
  ## ---
  #
  ## ---
  #fig_name = 'vort'
  #if fig_name in fig_names:
  #  FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='vort', depth=0, it=0,
  #                      clim=1e-4, cincr=-1., cmap='RdBu_r',
  #                      **Ddict)
  #  FigInf['fpath'] = path_pics+fig_name+'.png'
  #  FigInf['title'] = 'Surface vorticity'
  #  plt.savefig(FigInf['fpath'])
  #  qp.add_subsection(FigInf['title'])
  #  qp.add_fig(FigInf['fpath'])
  
  # -------------------------------------------------------------------------------- 
  # Sections
  # -------------------------------------------------------------------------------- 
  fname = '%s%s_%s.nc' % (run, oce_def, tstep)
  Ddict = dict(
    #xlim=[-180.,180.], ylim=[-90.,90.],
    sec_name='30W_100pts',
    path_ckdtree=path_ckdtree,
              )
  
  # ---
  fig_name = 'temp30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
                        t1=t1, t2=t2,
                        clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                        IcD=IcD, **Ddict)
    save_fig('Temperature at 30W', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
                        t1=t1, t2=t2,
                        clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                        IcD=IcD, **Ddict)
    save_fig('Salinity at 30W', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'dens30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='rhopot', it=0,
                        t1=t1, t2=t2,
                        clim=[1024., 1029.], cincr=0.2, cmap='cmo.dense',
                        IcD=IcD, **Ddict)
    save_fig('Density at 30W', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # zonal averages
  # -------------------------------------------------------------------------------- 
  Ddict = dict(
    path_ckdtree=path_ckdtree,
              )
  
  # ---
  fig_name = 'temp_gzave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
                        t1=t1, t2=t2,
                        clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                        sec_name='zave:glob:%s'%rgrid_name,
                        IcD=IcD, **Ddict)
    save_fig('Temperature global zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'temp_azave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
                        t1=t1, t2=t2,
                        clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                        sec_name='zave:atl:%s'%rgrid_name,
                        IcD=IcD, xlim=[-30,90], **Ddict)
    save_fig('Temperature Atlantic zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'temp_ipzave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
                        t1=t1, t2=t2,
                        clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                        sec_name='zave:indopac:%s'%rgrid_name,
                        IcD=IcD, xlim=[-30,65], **Ddict)
    save_fig('Temperature Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_gzave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
                        t1=t1, t2=t2,
                        clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                        sec_name='zave:glob:%s'%rgrid_name,
                        IcD=IcD, **Ddict)
    save_fig('Salinity global zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_azave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
                        t1=t1, t2=t2,
                        clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                        sec_name='zave:atl:%s'%rgrid_name,
                        IcD=IcD, xlim=[-30,90], **Ddict)
    save_fig('Salinity Atlantic zon. ave.', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'salt_ipzave'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
                        t1=t1, t2=t2,
                        clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                        sec_name='zave:indopac:%s'%rgrid_name,
                        IcD=IcD, xlim=[-30,65], **Ddict)
    save_fig('Salinity Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # Transports
  # -------------------------------------------------------------------------------- 
  fig_name = 'ke_100m'
  if fig_name in fig_names:
    u, it_ave   = pyic.time_average(IcD, 'u', t1=t1, t2=t2, iz=k100)
    v, it_ave   = pyic.time_average(IcD, 'v', t1=t1, t2=t2, iz=k100)
    ke = 0.5*(u**2+v**2)
    IaV = pyic.IconVariable('ke', 'm^2/s^2', 'kinetic energy')
    IaV.data = ke
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD, IaV, clim=[-7,0], cincr=0.5, cmap='RdYlBu_r',
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                    logplot=True, do_write_data_range=True,
                    title='log$_{10}$(kinetic energy) at %dm [m$^2$/s$^2$]'%(IcD.depthc[k100])
                    )
    save_fig('Kinetic energy 100m', path_pics, fig_name)

  fig_name = 'ke_2000m'
  if fig_name in fig_names:
    u, it_ave   = pyic.time_average(IcD, 'u', t1=t1, t2=t2, iz=k2000)
    v, it_ave   = pyic.time_average(IcD, 'v', t1=t1, t2=t2, iz=k2000)
    ke = 0.5*(u**2+v**2)
    IaV = pyic.IconVariable('ke', 'm^2/s^2', 'kinetic energy')
    IaV.data = ke
    IaV.interp_to_rectgrid(fpath_ckdtree)
    pyic.hplot_base(IcD, IaV, clim=[-7,0], cincr=0.5, cmap='RdYlBu_r',
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                    logplot=True, do_write_data_range=True,
                    title='log$_{10}$(kinetic energy) at %dm [m$^2$/s$^2$]'%(IcD.depthc[k2000])
                    )
    save_fig('Kinetic energy 2000m', path_pics, fig_name)
  
  # ---
  fig_name = 'bstr'
  
  if fig_name in fig_names:
    fname = '%s%s_%s.nc' % (run, oce_def, tstep)
    #f = Dataset(IcD.path_data+fname, 'r')
    ## FIXME: where to deal with time averaging?
    #mass_flux = f.variables['mass_flux'][:].mean(axis=0)
    #mass_flux_vint = mass_flux.sum(axis=0)
    #f.close()
    mass_flux, it_ave = pyic.time_average(IcD, 'mass_flux', t1, t2, iz='all')
    mass_flux_vint = mass_flux.sum(axis=0)
  
    # --- derive and interp bstr
    bstr = pyic.calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.)
    IaV = pyic.IconVariable('bstr', units='Sv', long_name='barotropic streamfunction',
                            coordinates='vlat vlon', is3d=False)
    IaV.data = bstr
    IaV.interp_to_rectgrid(fpath_ckdtree=fpath_ckdtree)
  
    # --- plot bstr
    ax, cax, mappable, Dstr = pyic.hplot_base(
      IcD, IaV, cmap='RdBu_r',
      clim=200, clevs=[-200,-160,-120,-80,-40,-30,-20,-10,10,20,30,40,80,120,160,200], do_write_data_range=True,
      use_tgrid=False, projection=projection,)
    FigInf = dict(name=fig_name, fpath=path_pics+fig_name,
                  long_name=IaV.long_name)
    save_fig('Barotropic streamfunction', path_pics, fig_name, FigInf)
  
  # --- 
  Ddict = dict(
    #xlim=[-180.,180.], ylim=[-90.,90.],
    sec_name='moc',
    path_ckdtree=path_ckdtree,
              )
  
  # ---
  fig_name = 'amoc'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='atlantic_moc', it=0,
                        t1=t1, t2=t2,
                        var_fac=1e-9,
                        clim=24, cincr=2., cmap='RdBu_r',
                        IcD=IcD_moc, xlim=[-30,90], **Ddict)
    save_fig('Atlantic MOC', path_pics, fig_name, FigInf)
  # ---
  fig_name = 'pmoc'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='pacific_moc', it=0,
                        t1=t1, t2=t2,
                        var_fac=1e-9,
                        clim=24, cincr=2., cmap='RdBu_r',
                        IcD=IcD_moc, xlim=[-30,65], **Ddict)
    save_fig('Pacific MOC', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'gmoc'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='global_moc', it=0,
                        t1=t1, t2=t2,
                        var_fac=1e-9,
                        clim=24, cincr=2., cmap='RdBu_r',
                        IcD=IcD_moc, **Ddict)
    save_fig('Global MOC', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # heat flux
  # -------------------------------------------------------------------------------- 
  fig_name = 'heat_flux'
  if fig_name in fig_names:
    global_hfl,   it_ave = pyic.time_average(IcD_moc, 'global_hfl',   t1, t2, iz='all')
    atlantic_hfl, it_ave = pyic.time_average(IcD_moc, 'atlantic_hfl', t1, t2, iz='all')
    pacific_hfl,  it_ave = pyic.time_average(IcD_moc, 'pacific_hfl',  t1, t2, iz='all')
  
    f = Dataset(IcD_moc.flist_ts[0], 'r')
    lat_hlf = f.variables['lat'][:]
    #ncv = f.variables['global_hfl']
    #long_name = ncv.long_name 
    #units = ncv.units
    f.close()
  
    hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="latitude", ylabel="",)
    ii=-1
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    ax.plot(lat_hlf, global_hfl/1e15, label='global_hfl')
    ax.plot(lat_hlf, atlantic_hfl/1e15, label='atlantic_hfl')
    ax.plot(lat_hlf, pacific_hfl/1e15, label='pacific_hfl')
    ax.grid(True)
    ax.legend()
    ax.set_title(f'heat transport [PW]')
  
    FigInf = dict()
    #FigInf['fpath'] = fpath
    #FigInf['long_name'] = long_name
  
    save_fig('Heat transport', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # freshwater flux
  # -------------------------------------------------------------------------------- 
  fig_name = 'freshwater_flux'
  if fig_name in fig_names:
    global_wfl,   it_ave = pyic.time_average(IcD_moc, 'global_wfl',   t1, t2, iz='all')
    atlantic_wfl, it_ave = pyic.time_average(IcD_moc, 'atlantic_wfl', t1, t2, iz='all')
    pacific_wfl,  it_ave = pyic.time_average(IcD_moc, 'pacific_wfl',  t1, t2, iz='all')
  
    f = Dataset(IcD_moc.flist_ts[0], 'r')
    lat_hlf = f.variables['lat'][:]
    #ncv = f.variables['global_wfl']
    #long_name = ncv.long_name 
    #units = ncv.units
    f.close()
  
    hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                 sharex=True, sharey=True, xlabel="latitude", ylabel="",)
    ii=-1
    
    ii+=1; ax=hca[ii]; cax=hcb[ii]
    ax.plot(lat_hlf, global_wfl/1e6, label='global')
    ax.plot(lat_hlf, atlantic_wfl/1e6, label='Atlantic')
    ax.plot(lat_hlf, pacific_wfl/1e6, label='Pacific')
    ax.grid(True)
    ax.legend()
    ax.set_title(f'freshwater transport [10^6 m^3 s^-1]')
  
    FigInf = dict()
    #FigInf['fpath'] = fpath
    #FigInf['long_name'] = long_name
  
    save_fig('Freshwater transport', path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # time series
  # -------------------------------------------------------------------------------- 
  fname = f'{run}{oce_mon}_????????????????.nc'
  
  fig_name = 'ts_amoc'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['amoc26n'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_heat_content'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['global_heat_content'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_ssh'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['ssh_global'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_sst'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['sst_global'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_sss'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['sss_global'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_hfl'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['HeatFlux_Total_global'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_wfl'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, 
      ['FrshFlux_Precipitation_global', 'FrshFlux_SnowFall_global', 'FrshFlux_Evaporation_global', 'FrshFlux_Runoff_global', 'FrshFlux_VolumeIce_global', 'FrshFlux_TotalOcean_global', 'FrshFlux_TotalIce_global', 'FrshFlux_VolumeTotal_global'], 
      title='Fresh water flux [m/s]',
      t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_ice_volume_nh'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, 
      ['ice_volume_nh', 'ice_volume_nh'], 
      title='sea ice volume Northern hemisphere [km^3]',
      t1=t1, t2=t2, ave_freq=12, mode_ave=['max', 'min'], labels=['max', 'min'])
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_ice_volume_sh'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, 
      ['ice_volume_sh', 'ice_volume_sh'], 
      title='sea ice volume Southern hemisphere [km^3]',
      t1=t1, t2=t2, ave_freq=12, mode_ave=['max', 'min'], labels=['max', 'min'])
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_ice_extent'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, 
      ['ice_extent_nh', 'ice_extent_sh'], 
      title='sea ice extent [km^2]',
      t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)

  fname = f'{run}{atm_mon}_????????????????.nc'
  
  fig_name = 'ts_tas_gmean'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['tas_gmean'], 
      t1=t1, t2=t2, ave_freq=12,
      var_add=-273.15, units=' [$^o$C]')
    save_fig(fig_name, path_pics, fig_name, FigInf)
  fig_name = 'ts_radtop_gmean'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_timeseries(IcD, fname, ['radtop_gmean'], t1=t1, t2=t2, ave_freq=12)
    save_fig(fig_name, path_pics, fig_name, FigInf)
  
  # -------------------------------------------------------------------------------- 
  # Additional plots
  # -------------------------------------------------------------------------------- 
  fname = '%s_idemix_%s.nc' % (run, tstep)
  Ddict = dict(
    #xlim=[-180.,180.], ylim=[-90.,90.],
    sec_name='30W_100pts',
    path_ckdtree=path_ckdtree,
              )
  
  # ---
  fig_name = 'tke30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='tke', it=0,
                        t1=t1, t2=t2,
                        clim=[-8,0], cincr=-1., cmap='plasma',
                        logplot=True,
                        **Ddict)
    save_fig('TKE at 30W', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'iwe30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='iwe', it=0,
                        t1=t1, t2=t2,
                        clim=[-8,0], cincr=-1., cmap='plasma',
                        logplot=True,
                        **Ddict)
    save_fig('IWE at 30W', path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'kv30w'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='K_tracer_h_to', it=0,
                        t1=t1, t2=t2,
                        clim=[-8,0], cincr=-1., cmap='plasma',
                        logplot=True,
                        **Ddict)
    save_fig('k_v at 30W', path_pics, fig_name, FigInf)

  # -------------------------------------------------------------------------------- 
  # Atmosphere 2D
  # -------------------------------------------------------------------------------- 
  Ddict = dict(
    xlim=[-180.,180.], ylim=[-90.,90.],
    rgrid_name=rgrid_name,
    path_ckdtree=path_ckdtree,
    projection=projection,
              )

  ## ---
  if do_atmosphere_plots:
    IcD_atm3d.plevc = np.array([100000,92500,85000,77500,70000,60000,50000,40000,30000,25000,20000,15000,10000,7000,5000,3000,1000])
    pfull, it_ave = pyic.time_average(IcD_atm3d, 'pfull', t1, t2, iz='all')
    icall, ind_lev, fac = pyic.calc_vertical_interp_weights(pfull, IcD_atm3d.plevc)

  # ---
  fig_name = 'atm_2m_temp'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='tas', it=0,
                             t1=t1, t2=t2,
                             var_add=-273.15,
                             units = '$^o$C',
                             clim=[-30.,30.], cincr=5.0, cmap='cmo.thermal',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('2m temperature', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_surface_temp'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='ts', it=0,
                             t1=t1, t2=t2,
                             var_add=-273.15,
                             units = '$^o$C',
                             clim=[-30.,30.], cincr=5.0, cmap='cmo.thermal',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('surface temperature', path_pics, fig_name, FigInf)

  #calc_bias = False
  #for fig_name in fig_names:
  #  if 'bias' in fig_name: 
  #    calc_bias = True
  #calc_bias=True
  #if calc_bias:
#    fpath_ckdtree = IcD_atm2d.rgrid_fpath_dict[rgrid_name]
#  
#    fpath_atm_ref = '/pool/data/ICON/post/r2b4_amip/ERA-Interim/ERAin_r2b4_atm_phy_1979_2008_MAM.nc'
#    f = Dataset(fpath_atm_ref, 'r')
#    tas_ref = f.variables['tas'][0,:]
#    clon = f.variables['clon'][:] * 180./np.pi
#    clat = f.variables['clat'][:] * 180./np.pi
#    f.close()
#  

    ##var = 'ta'
    ### --- interpolate data
    ##data3d, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all')
    ##datavi = data3d[ind_lev,icall]*fac+data3d[ind_lev-1,icall]*(1.-fac)
    ##lon, lat, datavihi = pyic.interp_to_rectgrid(datavi, fpath_ckdtree_atm, coordinates='clat clon')
    ##tasi = datavihi
  
  # ---
  fig_name = 'atm_tas_bias'
  if fig_name in fig_names:
    var = 'tas'
    # --- interpolate data
    data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz='all')
    lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
    datai = data2di
    # --- reference
    f = Dataset(fpath_ref_atm, 'r')
    data_ref = f.variables[var][:]
    f.close()
    # --- calculate bias
    data_bias = datai-data_ref
    IaV = pyic.IconVariable('data_bias', 'deg C', 'surface temperature bias')
    IaV.data = data_bias
    pyic.hplot_base(IcD_atm2d, IaV, clim=10., contfs='auto', cmap='RdBu_r', 
                    use_tgrid=False,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                    land_facecolor='none', do_write_data_range=True,
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
  # ---
  fig_name = 'atm_prw_bias'
  if fig_name in fig_names:
    var = 'prw'
    # --- interpolate data
    data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz='all')
    lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
    datai = data2di
    # --- reference
    f = Dataset(fpath_ref_atm, 'r')
    data_ref = f.variables[var][:]
    f.close()
    # --- calculate bias
    data_bias = datai-data_ref
    IaV = pyic.IconVariable('data_bias', 'kg m-2', 'integrated water vapor')
    IaV.data = data_bias
    pyic.hplot_base(IcD_atm2d, IaV, clim='sym', contfs='auto', cmap='RdBu_r', 
                    use_tgrid=False,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                    land_facecolor='none', do_write_data_range=True,
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_psl_bias'
  if fig_name in fig_names:
    var = 'psl'
    # --- interpolate data
    data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz='all')
    lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
    datai = data2di
    # --- reference
    f = Dataset(fpath_ref_atm, 'r')
    data_ref = f.variables[var][:]
    f.close()
    # --- calculate bias
    data_bias = datai-data_ref
    IaV = pyic.IconVariable('data_bias', 'hPa', 'sea level pressure')
    IaV.data = data_bias/100.
    pyic.hplot_base(IcD_atm2d, IaV, clim=10., contfs='auto', cmap='RdBu_r', 
                    use_tgrid=False,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                    land_facecolor='none', do_write_data_range=True,
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_tauu_bias'
  if fig_name in fig_names:
    var = 'tauu'
    # --- interpolate data
    data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz='all')
    lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
    datai = data2di
    # --- reference
    f = Dataset(fpath_ref_atm, 'r')
    data_ref = f.variables[var][:]
    f.close()
    # --- calculate bias
    data_bias = datai-data_ref
    IaV = pyic.IconVariable('data_bias', 'mN m-2', 'zonal wind stress')
    IaV.data = data_bias*1e3
    pyic.hplot_base(IcD_atm2d, IaV, clim=100, contfs='auto', cmap='RdBu_r', 
                    use_tgrid=False,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],  do_write_data_range=True,
                    #land_facecolor='none',
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_tauv_bias'
  if fig_name in fig_names:
    var = 'tauv'
    # --- interpolate data
    data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz='all')
    lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
    datai = data2di
    # --- reference
    f = Dataset(fpath_ref_atm, 'r')
    data_ref = f.variables[var][:]
    f.close()
    # --- calculate bias
    data_bias = datai-data_ref
    IaV = pyic.IconVariable('data_bias', 'mN m-2', 'meridional wind stress')
    IaV.data = data_bias*1e3
    pyic.hplot_base(IcD_atm2d, IaV, clim=100, contfs='auto', cmap='RdBu_r', 
                    use_tgrid=False,
                    projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],  do_write_data_range=True,
                    #land_facecolor='none',
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_sea_level_pressure'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='psl', it=0,
                             t1=t1, t2=t2,
                             var_add=-1000.,
                             var_fac=0.01,
                             units='hPa-1000',
                             clim=[-25.,25.], cincr=5.0, cmap='RdYlBu_r',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('sea level pressure', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_zonal_wind_stress'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='tauu', it=0,
                             t1=t1, t2=t2,
                             var_fac=1e3,
                             units='mN/m$^2$',
                             clim=[-200.,200.], clevs=[-200,-100,-50,-20,0,20,50,100,200], cmap='RdYlBu_r',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('zonal wind stress', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_meridional_wind_stress'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='tauv', it=0,
                             t1=t1, t2=t2,
                             var_fac=1e3,
                             units='mN/m$^2$',
                             clim=[-200.,200.], clevs=[-200,-100,-50,-20,0,20,50,100,200], cmap='RdYlBu_r',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('meridional wind stress', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_column_water_vapour'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='prw', it=0,
                             t1=t1, t2=t2,
                             clim=[10.,50.], cincr=10.0, cmap='RdYlBu_r',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('column water vapour', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_cllvi_clivi'
  if fig_name in fig_names:
    cllvi, it_ave = pyic.time_average(IcD_atm2d, 'cllvi', t1, t2, iz='all')
    clivi, it_ave = pyic.time_average(IcD_atm2d, 'clivi', t1, t2, iz='all')
    IaV = pyic.IconVariable('cclvi_clivi', 'g/m$^2', 'Liquid water + ice content')
    IaV.data = (cllvi+clivi)*1e3
    pyic.hplot_base(IcD_atm2d, IaV, clim=[10,300], clevs=[10,50,100,200,300], cmap='RdYlBu_r', 
                    use_tgrid=False,
                    projection=projection, do_write_data_range=True,
                    land_facecolor='none',
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_total_precipitation'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='pr', it=0,
                             t1=t1, t2=t2,
                             var_fac=86400.,
                             units='mm/d',
                             clim=[1,12], clevs=[1,2,4,8,12], cmap='RdYlBu',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('total precipitation', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_total_cloud_cover'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='clt', it=0,
                             t1=t1, t2=t2,
                             var_fac=100,
                             units='%',
                             clim=[10.,90.], cincr=20.0, cmap='RdYlBu_r',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('total cloud cover', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_p_e'
  if fig_name in fig_names:
    pr, it_ave = pyic.time_average(IcD_atm2d, 'pr', t1, t2, iz='all')
    evspsbl, it_ave = pyic.time_average(IcD_atm2d, 'evspsbl', t1, t2, iz='all')
    IaV = pyic.IconVariable('p_e', 'mm/day', 'P-E')
    IaV.data = (pr+evspsbl)*86400
    IaV.interp_to_rectgrid(fpath_ckdtree_atm)
    pyic.hplot_base(IcD_atm2d, IaV, clim=5, cincr=1, cmap='RdBu', 
                    use_tgrid=False,
                    projection=projection, do_write_data_range=True,
                    land_facecolor='none',
                    asp=0.5)
    FigInf = dict(long_name=IaV.long_name)
    save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_10m_wind'
  if fig_name in fig_names:
    FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='sfcwind', it=0,
                             t1=t1, t2=t2,
                             clim=[0.5, 10.], clevs=[0.5,1.,2.,3.,5.,7.,10.], cmap='RdYlBu_r',
                             land_facecolor='none',
                             IcD=IcD_atm2d, **Ddict)
    save_fig('surface temperature', path_pics, fig_name, FigInf)

  # ---
  fig_name = 'atm_temp_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'ta', t1, t2, iz='all')
    IaV = pyic.IconVariable('temp', 'deg C', 'zonally averaged temperature')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data += -273.15
    pyic.vplot_base(IcD_atm3d, IaV, clim=[-80., 30.], cincr=5, contfs='auto',
                    cmap='RdYlBu_r',
                    asp=0.5, do_write_data_range=True)
    save_fig('temperature', path_pics, fig_name)

  # ---
  fig_name = 'atm_u_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'ua', t1, t2, iz='all')
    IaV = pyic.IconVariable('ua', 'm/s', 'zon. ave. zonal velocity')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    pyic.vplot_base(IcD_atm3d, IaV, clim=30., cincr=5, contfs='auto', cmap='RdBu_r',
                    asp=0.5, do_write_data_range=True)
    save_fig('zonal velocity', path_pics, fig_name)

  # ---
  fig_name = 'atm_v_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'va', t1, t2, iz='all')
    IaV = pyic.IconVariable('va', 'm/s', 'zon. ave. meridional velocity')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    pyic.vplot_base(IcD_atm3d, IaV, clim=4., cincr=0.4, contfs='auto', cmap='RdBu_r',
                    asp=0.5, do_write_data_range=True)
    save_fig('meridional velocity', path_pics, fig_name)

  # ---
  fig_name = 'atm_spec_hum_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'hus', t1, t2, iz='all')
    IaV = pyic.IconVariable('hus', 'g/kg', 'zon. ave. specific humidity')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data *= 1000.
    pyic.vplot_base(IcD_atm3d, IaV, clim=0.5, contfs=[0.005,0.01,0.03,0.05,0.1,0.3,0.5,1.,2.,5.,8.],
                    asp=0.5, do_write_data_range=True)
    save_fig('specific humidity', path_pics, fig_name)

  # ---
  fig_name = 'atm_rel_hum_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'hur', t1, t2, iz='all')
    IaV = pyic.IconVariable('hur', '%', 'zon. ave. relative humidity')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data *= 100.
    pyic.vplot_base(IcD_atm3d, IaV, clim=[0,100.], cincr=10., contfs='auto',
                    asp=0.5, do_write_data_range=True)
    save_fig('relative humidity', path_pics, fig_name)

  # ---
  fig_name = 'atm_cloud_cover_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'cl', t1, t2, iz='all')
    IaV = pyic.IconVariable('cl', '%', 'zon. ave. cloud cover')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data *= 100.
    pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2.5, contfs='auto',
                    asp=0.5, do_write_data_range=True)
    save_fig('cloud cover', path_pics, fig_name)

  # ---
  fig_name = 'atm_cloud_water_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'clw', t1, t2, iz='all')
    IaV = pyic.IconVariable('clw', 'mg/kg', 'zon. ave. cloud water')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data *= 1e6
    clw = IaV.data
    pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2., contfs='auto',
                    asp=0.5, do_write_data_range=True)
    save_fig('cloud water', path_pics, fig_name)

  # ---
  fig_name = 'atm_cloud_ice_zave'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'cli', t1, t2, iz='all')
    IaV = pyic.IconVariable('cli', 'mg/kg', 'zon. ave. cloud ice')
    IaV.lat_sec, IaV.data = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    IaV.data *= 1e6
    cli = IaV.data
    pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2., contfs='auto',
                    asp=0.5, do_write_data_range=True)
    save_fig('cloud ice', path_pics, fig_name)

  # ---
  fig_name = 'atm_cloud_water_ice_zave'
  if fig_name in fig_names:
    lat_sec = IaV.lat_sec
    IaV = pyic.IconVariable('clw_cli', 'mg/kg', 'zon. ave. cloud water+ice')
    IaV.data = clw+cli
    IaV.lat_sec = lat_sec 
    pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2., contfs='auto',
                    asp=0.5, do_write_data_range=True)
    save_fig('cloud water+ice', path_pics, fig_name)

  # ---
  fig_name = 'atm_psi'
  if fig_name in fig_names:
    data, it_ave = pyic.time_average(IcD_atm3d, 'va', t1, t2, iz='all')
    IaV = pyic.IconVariable('psi', '10$^9$kg/s', 'Psi')
    IaV.lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
    plevi = np.concatenate(([107500],0.5*(IcD_atm3d.plevc[1:]+IcD_atm3d.plevc[:-1]),[0.]))
    dp = np.diff(plevi)
    IaV.data = (2.*np.pi*6.371e6/9.81)*np.cos(IaV.lat_sec*np.pi/180.)* ( (data_zave[::-1,:]*dp[::-1,np.newaxis]).cumsum(axis=0) )[::-1,:]*1e-9
    pyic.vplot_base(IcD_atm3d, IaV, clim=[-80,80], cincr=10., contfs='auto', cmap='RdBu_r',
                    asp=0.5, do_write_data_range=True)
    save_fig('Psi', path_pics, fig_name)

  # --- North Pole

  # ---
  fig_name = 'np_zonal_wind_stress'
  if fig_name in fig_names:
    #ax = plt.axes(projection=ccrs.NorthPolarStereo())
    tauu, it_ave   = pyic.time_average(IcD_atm2d, 'tauu', it_ave=it_ave_mar, iz='all')
    IaV = pyic.IconVariable('tauu', 'mN/m$^2$', 'zonal wind stress')
    IaV.data = tauu*1e3
    IaV.interp_to_rectgrid(fpath_ckdtree_atm)
    ax, cax, mappable, Dstr = pyic.hplot_base(IcD_atm2d, IaV, clim=200, cincr=25, cmap='RdYlBu_r',
                    projection='NorthPolarStereo', xlim=[-180.,180.], ylim=[60.,90.],
                    crs_features=False, do_plot_settings=False, do_write_data_range=True,
                    )
    pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
    #FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='tauu', it=0,
    #                         t1=t1, t2=t2,
    #                         var_fac=1e3,
    #                         units='mN/m$^2$',
    #                         clim=[-200.,200.], cincr=25.0, cmap='RdYlBu_r',
    #                         IcD=IcD_atm2d, **Ddict)
    plt.show()
    sys.exit()
    save_fig('zonal wind stress', path_pics, fig_name, FigInf)

  # --------------------------------------------------------------------------------
  # Website
  # --------------------------------------------------------------------------------
  flist_all = glob.glob(path_pics+'*.json')
  
  print('Make QP website for the following figures:')
  for plot in plist:
    fpath = path_pics+plot+'.json'
    if plot.startswith('sec'):
      qp.add_section(plot.split(':')[1])
    else:
      if os.path.exists(fpath):
        print(fpath)
        flist_all.remove(fpath)
        with open(fpath) as file_json:
          FigInf = json.load(file_json)
        qp.add_subsection(FigInf['title'])
        rfpath_pics = rpath_pics+FigInf['name']
        qp.add_fig(rfpath_pics)
      else:
        raise ValueError('::: Error: file does not exist: %s!:::' %fpath)
  
  # --- all residual figs that can be found
  qp.add_section('More plots')
  for fpath in flist_all:
    print(fpath)
    with open(fpath) as file_json:
      FigInf = json.load(file_json)
    qp.add_subsection(FigInf['title'])
    rfpath_pics = rpath_pics+FigInf['name']
    qp.add_fig(rfpath_pics)
  
  qp.write_to_file()

# --- add link of this time average
print("Executing qp_link_all.py")
os.system(f"python qp_link_all.py {path_qp}/../")

### --------------------------------------------------------------------------------
### show figures
### --------------------------------------------------------------------------------
##if fig_names.size<3:
##  plt.show()
print('All done!')
