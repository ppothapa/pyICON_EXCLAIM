import sys, glob, os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import my_toolbox as my
from ipdb import set_trace as mybreak
from importlib import reload

#sys.path.append('/home/mpim/m300602/pyicon')
import pyicon as pyic
reload(pyic)

# ---
runname = 'icon_08'
#run = 'nib0001'
#tstep = '20090101T000000Z'
run = 'nib0002'
tstep = '20020101T000000Z'
iz = 0
it = 0

path_data = '/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/%s/icon-oes/experiments/%s/' % (runname, run)

path_pics = './pics/'
if not os.path.exists(path_pics):
  os.makedirs(path_pics)

qp = pyic.QuickPlotWebsite(
  title='pyicon Quick Plot', 
  author='Nils Brueggemann', 
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  fpath_css='./mycss2.css'
  )

plt.close('all')

# ================================================================================ 
qp.add_section('Basics')
# ================================================================================ 

# -------------------------------------------------------------------------------- 
#qp.add_subsection('Surface dynamics')
# -------------------------------------------------------------------------------- 

#lon_reg = (-66, -40)
#lat_reg = (50, 70)
lon_reg = [-180, 180]
lat_reg = [-90, 90]
projection = 'PlateCarree'
#projection = 'none'

path_data    = '/Users/nbruggemann/work/icon_playground/icon_r2b4_test_data/icon_08/icon-oes/experiments/nib0002/'
path_ckdtree = '/Users/nbruggemann/work/icon_playground/icon_ckdtree/'

fig_names = []
#fig_names += ['mld', 'sst', 'sss', 'ssh']
#fig_names += ['temp30w', 'salt30w', 'dens30w']
#fig_names += ['tke30w', 'iwe30w', 'kv30w']
#fig_names += ['vort']
fig_names += ['amoc', 'pmoc', 'gmoc']
rgrid_name='orig'

Ddict = dict(
  xlim=lon_reg, ylim=lat_reg,
  rgrid_name=rgrid_name,
  path_ckdtree=path_ckdtree,
  projection=projection,
            )

def add_fig(FigInf):
  fpath = FigInf['fpath']
  print('Saving %s...' % (fpath))
  #qp.add_subsection(FigInf['long_name'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(fpath)
  return

# ---
fname = '%s_%s.nc' % (run, tstep)
# ---

# ---
fig_name = 'mld'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='mld', depth=0, it=0,
                      clim=[0,5000.], cincr=250., cmap='cmo.deep',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Mixed layer depth'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'sst'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='to', depth=0, it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Sea surface temperature'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'sss'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='so', depth=0, it=0,
                      clim=[32.,37], cincr=0.25, cmap='cmo.haline',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Sea surface salinity'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'ssh'
if fig_name in fig_names:
  FigInf = pyic.qp_hplot(fpath=path_data+fname, var='zos', depth=0, it=0,
                      clim=2, cincr=0.2, cmap='RdBu_r',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Sea surface height'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

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


# ---
# ---
# ---
Ddict = dict(
  xlim=lon_reg, ylim=lat_reg,
  sec_name='30W_100pts',
  path_ckdtree=path_ckdtree,
            )

# ---
fig_name = 'temp30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='to', it=0,
                      clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Temperature at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'salt30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='so', it=0,
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Salinity at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'dens30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='rhopot', it=0,
                      clim=[1024., 1029.], cincr=0.2, cmap='cmo.dense',
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Salinity at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fname = '%s_idemix_%s.nc' % (run, tstep)
# ---

# ---
fig_name = 'tke30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='tke', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'TKE at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'iwe30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='iwe', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'IWE at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'kv30w'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='K_tracer_h_to', it=0,
                      clim=[-8,0], cincr=-1., cmap='plasma',
                      logplot=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'k_v at 30W'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fname = '%s_MOC_%s.nc' % (run, tstep)
# ---

# ---
fig_name = 'amoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='atlantic_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      do_load_moc=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Atlantic MOC'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'pmoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='pacific_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      do_load_moc=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Pacific MOC'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])

# ---
fig_name = 'gmoc'
if fig_name in fig_names:
  FigInf = pyic.qp_vplot(fpath=path_data+fname, var='global_moc', it=0,
                      var_fac=1e-9,
                      clim=24, cincr=2., cmap='RdBu_r',
                      do_load_moc=True,
                      **Ddict)
  FigInf['fpath'] = path_pics+fig_name+'.png'
  FigInf['title'] = 'Global MOC'
  plt.savefig(FigInf['fpath'])
  qp.add_subsection(FigInf['title'])
  qp.add_fig(FigInf['fpath'])


# --------------------------------------------------------------------------------
#qp.add_subsection('Vertical sections')
# --------------------------------------------------------------------------------

qp.write_to_file()

#IC = FigInf['IC']
if len(fig_names)<3:
  plt.show()
print('All done!')
