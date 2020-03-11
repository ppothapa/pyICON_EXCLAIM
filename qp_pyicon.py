import sys, glob, os
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from ipdb import set_trace as mybreak
from importlib import reload
import json
#sys.path.append('/home/mpim/m300602/pyicon')
import pyicon as pyic
reload(pyic)
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ---
#runname = 'icon_08'
#run = 'nib0002'
#tstep = '20020101T000000Z'
#
#path_data    = '/Users/nbruggemann/work/icon_playground/icon_r2b4_test_data/icon_08/icon-oes/experiments/nib0002/'
#path_ckdtree = '/Users/nbruggemann/work/icon_playground/icon_ckdtree/'

# ------ r2b4
#exec(open("./conf-icon_08-nib002.py").read())
#exec(open("./conf-mac-icon_08-nib0002.py").read())

# ------ r2b6
#exec(open("./conf-icon_08-nib003.py").read())
# does not work so far. diff. grid file
#exec(open("./conf-ocean_omip_long_r2b6_19224-YVF.py").read()) 

# ------ r2b8
#exec(open("./conf-ocean_era51h_r2b8_19074-AMK.py").read())

# ------ r2b6
exec(open("./conf-icon_08-nib0004.py").read())
#exec(open("./conf-icon_08-nib0006.py").read())

path_qp = './all_qps/qp-'+runname+'-'+run+'/'
if not os.path.exists(path_qp):
  os.makedirs(path_qp)

rpath_pics = './pics/'
path_pics = path_qp+rpath_pics
if not os.path.exists(path_pics):
  os.makedirs(path_pics)

shutil.copyfile('./qp_css.css', path_qp+'qp_css.css')

qp = pyic.QuickPlotWebsite(
  title='%s | %s' % (runname, run), 
  author=os.environ.get('USER'),
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  path_data=path_data,
  info='time: '+tstep,
  fpath_css='./qp_css.css',
  fpath_html=path_qp+'qp_index.html'
  )

fname_this_script = __file__.split('/')[-1]
shutil.copyfile(fname_this_script, path_qp+'bcp_'+fname_this_script)

# -------------------------------------------------------------------------------- 
# Settings
# -------------------------------------------------------------------------------- 
projection = 'PlateCarree'
#projection = 'none'

fig_names = []
#fig_names += ['sst']
#fig_names += ['mld_jan', 'mld_jul', 'sst', 'sss', 'ssh']
fig_names += ['mld_jan', 'sst', 'sss', 'ssh']
fig_names += ['ice']
fig_names += ['temp30w', 'salt30w', 'dens30w']
fig_names += ['sst_bias', 'temp_bias_gzave', 'temp_bias_azave', 'temp_bias_ipzave']
fig_names += ['sss_bias', 'salt_bias_gzave', 'salt_bias_azave', 'salt_bias_ipzave']
fig_names += ['temp_gzave', 'temp_azave', 'temp_ipzave']
fig_names += ['salt_gzave', 'salt_azave', 'salt_ipzave']
fig_names += ['salt_gzave', 'salt_azave', 'salt_ipzave']
fig_names += ['bstr', 'amoc', 'pmoc', 'gmoc']
#fig_names += ['tke30w', 'iwe30w', 'kv30w']

#fig_names = []
#fig_names += ['ice']
#fig_names += ['bstr']

#fig_names += ['vort']

# -------------------------------------------------------------------------------- 
# Function to save figures
# -------------------------------------------------------------------------------- 
close_figs = True
def save_fig(title, path_pics, fig_name, FigInf):
  FigInf['name']  = fig_name+'.png'
  FigInf['fpath'] = path_pics+FigInf['name']
  FigInf['title'] = title
  print('Saving {:<40} {:<40}'.format(FigInf['fpath'], FigInf['title']))
  plt.savefig(FigInf['fpath'], dpi=300)
  with open(path_pics+fig_name+'.json', 'w') as fj:
    json.dump(FigInf, fj, sort_keys=True, indent=4)
  if close_figs:
    plt.close()
  return
plt.close('all')

# -------------------------------------------------------------------------------- 
# Load all necessary data sets (can take a while)
# -------------------------------------------------------------------------------- 
if True:
  fname = '%s_%s.nc' % (run, tstep)
  print('Dataset %s' % (fname))
  #rgrid_name = 'global_1.0'
  rgrid_name = 'global_0.3'
  IcD = pyic.IconData(
                 search_str   = fname,
                 path_data    = path_data,
                 path_ckdtree = path_ckdtree,
                 path_tgrid   = path_tgrid,
                 fpath_tgrid  = fpath_tgrid,
                 fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name,
                 do_triangulation = False,
                 omit_last_file = False,
                )
  fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
  
  fname_moc = '%s_MOC_%s.nc' % (run, tstep)
  print('Dataset %s' % (fname))
  IcD_moc = pyic.IconData(
                 search_str   = fname_moc,
                 path_data    = path_data,
                 path_ckdtree = path_ckdtree,
                 path_tgrid   = path_tgrid,
                 fpath_tgrid  = fpath_tgrid,
                 fpath_fx     = fpath_fx,
                 rgrid_name   = rgrid_name,
                 do_triangulation = False,
                 omit_last_file = False,
                )
  
print('Done reading datasets')

# -------------------------------------------------------------------------------- 
# upper ocean
# -------------------------------------------------------------------------------- 
fname = '%s_%s.nc' % (run, tstep)
Ddict = dict(
  xlim=[-180.,180.], ylim=[-90.,90.],
  rgrid_name=rgrid_name,
  path_ckdtree=path_ckdtree,
  projection=projection,
            )

# ---
fig_name = 'mld_jan'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='mld', depth=0, it=0,
                      clim=[0,5000.], cincr=250., cmap='cmo.deep',
                      IcD=IcD, **Ddict)
  save_fig('Mixed layer depth January', path_pics, fig_name, FigInf)

# ---
fig_name = 'mld_jul'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='mld', depth=0, it=6,
                      clim=[0,5000.], cincr=250., cmap='cmo.deep',
                      IcD=IcD, **Ddict)
  save_fig('Mixed layer depth July', path_pics, fig_name, FigInf)

# ---
fig_name = 'sst'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='to', depth=0, it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      IcD=IcD, **Ddict)
  #save_fig('Sea surface temperature', path_pics, fig_name, FigInf)
  save_fig('SST', path_pics, fig_name, FigInf)

# ---
fig_name = 'sss'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='so', depth=0, it=0,
                      clim=[32.,37], cincr=0.25, cmap='cmo.haline',
                      IcD=IcD, **Ddict)
  #save_fig('Sea surface salinity', path_pics, fig_name, FigInf)
  save_fig('SSS', path_pics, fig_name, FigInf)

# ---
fig_name = 'ssh'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='zos', depth=0, it=0,
                      clim=2, cincr=0.2, cmap='RdBu_r',
                      IcD=IcD, **Ddict)
  #save_fig('Sea surface height', path_pics, fig_name, FigInf)
  save_fig('SSH', path_pics, fig_name, FigInf)

# ---
fig_name = 'ice'
if fig_name in fig_names:
  proj_1 = 'NorthPolarStereo'
  proj_2 = 'SouthPolarStereo'
  hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, sasp=1., fig_size_fac=2.,
               sharex=True, sharey=True, xlabel="", ylabel="",
               projection=[getattr(ccrs,proj_1)(), getattr(ccrs,proj_2)()],
                              )
  ii=-1
  
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='hi', depth=0, it=0,
                         clim=[0,6], cincr=0.5, cmap='cmo.ice_r',
                         xlim=[-180, 180], ylim=[60,90],
                         ax=ax, cax=cax,
                         IcD=IcD,
                         rgrid_name=rgrid_name,
                         path_ckdtree=path_ckdtree,
                         projection='PlateCarree',
                         crs_features=False,
                        )
  ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

  ii+=1; ax=hca[ii]; cax=hcb[ii]
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='hi', depth=0, it=0,
                         clim=[0,3], cincr=0.25, cmap='cmo.ice_r',
                         xlim=[-180, 180], ylim=[-90,-50],
                         ax=ax, cax=cax,
                         IcD=IcD,
                         rgrid_name=rgrid_name,
                         path_ckdtree=path_ckdtree,
                         projection='PlateCarree',
                         crs_features=False,
                        )
  ax.set_extent([-180, 180, -90, -50], ccrs.PlateCarree())

  for ax in hca:
    ax.gridlines()
    ax.add_feature(cartopy.feature.LAND)
    ax.coastlines()
  
  save_fig('Ice thickness', path_pics, fig_name, FigInf)

# -------------------------------------------------------------------------------- 
# biases
# -------------------------------------------------------------------------------- 
fname = '%s_%s.nc' % (run, tstep)
if True:
  #IcD = pyic.IconData(
  #               search_str   = fname,
  #               path_data    = path_data,
  #               path_ckdtree = path_ckdtree,
  #               rgrid_name   = rgrid_name,
  #              )
  fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]

  #fpath_initial_state = path_data+'initial_state.nc'
  f = Dataset(fpath_initial_state, 'r')
  temp_ref = f.variables['T'][0,:,:]
  salt_ref = f.variables['S'][0,:,:]
  f.close()
  temp_ref[temp_ref==0.]=np.ma.masked
  salt_ref[salt_ref==0.]=np.ma.masked

  f = Dataset(path_data+fname, 'r')
  temp = f.variables['to'][:,:,:].mean(axis=0)
  salt = f.variables['so'][:,:,:].mean(axis=0)
  f.close()
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
  pyic.hplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Tbias: surface', path_pics, fig_name, FigInf)

# ---
fig_name = 'temp_bias_gzave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='global', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Tbias: global zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'temp_bias_azave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='atl', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Tbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'temp_bias_ipzave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='indopac', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Tbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'sss_bias'
if fig_name in fig_names:
  IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
  IaV.data = sbias[0,:]
  IaV.interp_to_rectgrid(fpath_ckdtree)
  pyic.hplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Sbias: surface', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_bias_gzave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='global', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Sbias: global zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_bias_azave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='atl', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Sbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_bias_ipzave'
if fig_name in fig_names:
  IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
  IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='indopac', 
                             fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
  pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.2,
                  sasp=0.5)
  FigInf = dict(long_name=IaV.long_name)
  save_fig('Sbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)

## ---
#fname = '%s_restart_oce_%s.nc' % (run, tstep)
## ---
#
## ---
#fig_name = 'vort'
#if fig_name in fig_names:
#  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='vort', depth=0, it=0,
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
fname = '%s_%s.nc' % (run, tstep)
Ddict = dict(
  xlim=[-180.,180.], ylim=[-90.,90.],
  sec_name='30W_100pts',
  path_ckdtree=path_ckdtree,
            )

# ---
fig_name = 'temp30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='to', it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      IcD=IcD, **Ddict)
  save_fig('Temperature at 30W', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='so', it=0,
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      IcD=IcD, **Ddict)
  save_fig('Salinity at 30W', path_pics, fig_name, FigInf)

# ---
fig_name = 'dens30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='rhopot', it=0,
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
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='to', it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      sec_name='zave:glob:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Temperature global zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'temp_azave'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='to', it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      sec_name='zave:atl:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Temperature Atlantic zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'temp_ipzave'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='to', it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      sec_name='zave:indopac:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Temperature Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_gzave'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='so', it=0,
                      clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                      sec_name='zave:glob:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Salinity global zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_azave'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='so', it=0,
                      clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                      sec_name='zave:atl:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Salinity Atlantic zon. ave.', path_pics, fig_name, FigInf)

# ---
fig_name = 'salt_ipzave'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='so', it=0,
                      clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
                      sec_name='zave:indopac:%s'%rgrid_name,
                      IcD=IcD, **Ddict)
  save_fig('Salinity Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)

# -------------------------------------------------------------------------------- 
# Transports
# -------------------------------------------------------------------------------- 

# ---
fig_name = 'bstr'

if fig_name in fig_names:
  fname = '%s_%s.nc' % (run, tstep)
  f = Dataset(IcD.path_data+fname, 'r')
  # FIXME: where to deal with time averaging?
  mass_flux = f.variables['mass_flux'][:].mean(axis=0)
  mass_flux_vint = mass_flux.sum(axis=0)
  f.close()

  # --- derive and interp bstr
  bstr = pyic.calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.)
  IaV = pyic.IconVariable('bstr', units='Sv', long_name='barotropic streamfunction',
                          coordinates='vlat vlon', is3d=False)
  IaV.data = bstr
  IaV.interp_to_rectgrid(fpath_ckdtree=fpath_ckdtree)

  # --- plot bstr
  ax, cax, mappable, Dstr = pyic.hplot_base(
    IcD, IaV, clim=60, cmap='RdBu_r', cincr=5.,
    use_tgrid=False, projection=projection,)
  FigInf = dict(name=fig_name, fpath=path_pics+fig_name,
                long_name=IaV.long_name)
  save_fig('Barotropic streamfunction', path_pics, fig_name, FigInf)

# --- 
Ddict = dict(
  xlim=[-180.,180.], ylim=[-90.,90.],
  sec_name='moc',
  path_ckdtree=path_ckdtree,
            )

# ---
fig_name = 'amoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname_moc, var='atlantic_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      IcD=IcD_moc, **Ddict)
  save_fig('Atlantic MOC', path_pics, fig_name, FigInf)

# ---
fig_name = 'pmoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname_moc, var='pacific_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      IcD=IcD_moc, **Ddict)
  save_fig('Pacific MOC', path_pics, fig_name, FigInf)

# ---
fig_name = 'gmoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname_moc, var='global_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      IcD=IcD_moc, **Ddict)
  save_fig('Global MOC', path_pics, fig_name, FigInf)

# -------------------------------------------------------------------------------- 
# Additional plots
# -------------------------------------------------------------------------------- 
fname = '%s_idemix_%s.nc' % (run, tstep)
Ddict = dict(
  xlim=[-180.,180.], ylim=[-90.,90.],
  sec_name='30W_100pts',
  path_ckdtree=path_ckdtree,
            )

# ---
fig_name = 'tke30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='tke', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  save_fig('TKE at 30W', path_pics, fig_name, FigInf)

# ---
fig_name = 'iwe30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='iwe', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  save_fig('IWE at 30W', path_pics, fig_name, FigInf)

# ---
fig_name = 'kv30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='K_tracer_h_to', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  save_fig('k_v at 30W', path_pics, fig_name, FigInf)

# --------------------------------------------------------------------------------
# Website
# --------------------------------------------------------------------------------
flist_all = glob.glob(path_pics+'*.json')

plist = []
plist += ['sec:Upper ocean']
#plist += ['ssh', 'sst', 'sss', 'mld_jan', 'mld_jul'] 
plist += ['ssh', 'sst', 'sss', 'mld_jan'] 
plist += ['ice']
plist += ['sec:Sections']
plist += ['temp30w', 'salt30w', 'dens30w']
plist += ['sec:Zonal averages']
plist += ['temp_gzave', 'salt_gzave', 'temp_azave', 'salt_azave', 
          'temp_ipzave', 'salt_ipzave']
plist += ['sec:Transports']
plist += ['bstr', 'amoc', 'pmoc', 'gmoc']
plist += ['sec:Biases']
plist += ['sst_bias', 'sss_bias']
plist += ['temp_bias_gzave', 'temp_bias_azave', 'temp_bias_ipzave']
plist += ['salt_bias_gzave', 'salt_bias_azave', 'salt_bias_ipzave']

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

# --------------------------------------------------------------------------------
# show figures
# --------------------------------------------------------------------------------
#IC = FigInf['IC']
if len(fig_names)<3:
  plt.show()
print('All done!')
