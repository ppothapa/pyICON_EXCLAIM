import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
from importlib import reload
import pyicon as pyic
reload(pyic)
import cartopy.crs as ccrs

run = 'nib0004'
runname = 'icon_08'
gname = 'r2b6'
lev = 'L64'

path_data     = f'/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/{runname}/icon-oes/experiments/{run}/'
path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
fpath_ckdtree = f'{path_grid}ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz'
fpath_ckdtree_zave = fpath_ckdtree

IcD = pyic.IconData(
               fname        = run+'_????????????????.nc', 
               path_data    = path_data,
               path_grid    = path_grid,
               gname        = gname,
               lev          = lev,
               do_triangulation = True,
               omit_last_file   = False
              )


# --- specify time step
it = np.argmin(np.abs(IcD.times-np.datetime64('2295-01-01T00:00:00')))
# --- specify depth level
iz = np.argmin(np.abs(IcD.depthc-1000.))

# --- load data
f = Dataset(IcD.flist_ts[it], 'r')
to = f.variables['to'][IcD.its[it],:,:]
so = f.variables['so'][IcD.its[it],:,:]
to[IcD.wet_c==0.]=np.ma.masked
so[IcD.wet_c==0.]=np.ma.masked
f.close()

# --- load reference data
fpath_initial_state = path_data+'initial_state.nc'
f = Dataset(fpath_initial_state, 'r')
temp_ref = f.variables['T'][0,:,:]
salt_ref = f.variables['S'][0,:,:]
temp_ref[IcD.wet_c==0.]=np.ma.masked
salt_ref[IcD.wet_c==0.]=np.ma.masked
f.close()

# --- calculate bias
tbias = to-temp_ref
sbias = so-salt_ref

# --- horizontally averaged bias
print('calculate horizontal averages')
total_area = (IcD.wet_c*IcD.cell_area[np.newaxis,:]).sum(axis=1)

tbias_have = (tbias*IcD.cell_area[np.newaxis,:]).sum(axis=1)/total_area
sbias_have = (sbias*IcD.cell_area[np.newaxis,:]).sum(axis=1)/total_area

# --- zonally averaged bias
print('calculate zonal averages')
lat_sec, tbias_gzave = pyic.zonal_average_3d_data(tbias, basin='global', 
                         fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree_zave)
lat_sec, sbias_gzave = pyic.zonal_average_3d_data(sbias, basin='global', 
                         fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree_zave)

# --- vertically averaged bias
print('calculate vertial averages')
# ------ define depth levels in between which biases should be averaged
levs = np.array([0,250,500,3000,6000])
# ------ allocate variables
ddnpz = np.load(fpath_ckdtree)
tbias_vavei = np.zeros((levs.size, ddnpz['lat'].size, ddnpz['lon'].size))
sbias_vavei = np.zeros((levs.size, ddnpz['lat'].size, ddnpz['lon'].size))
# ------ loop over layers, average bias and interpolate to rectgrid
for nn in range(len(levs)-1):
  iz1 = np.argmin((IcD.depthc-levs[nn])**2)
  iz2 = np.argmin((IcD.depthc-levs[nn+1])**2)
  # ------ temperature
  bias_vave = ((tbias*IcD.dzw)[iz1:iz2,:]).sum(axis=0)/(IcD.dzw[iz1:iz2,:].sum(axis=0)+1e-33)
  lon, lat, tbias_vavei[nn,:,:] = pyic.interp_to_rectgrid(bias_vave, fpath_ckdtree)
  # ------ salinity
  bias_vave = ((sbias*IcD.dzw)[iz1:iz2,:]).sum(axis=0)/(IcD.dzw[iz1:iz2,:].sum(axis=0)+1e-33)
  lon, lat, sbias_vavei[nn,:,:] = pyic.interp_to_rectgrid(bias_vave, fpath_ckdtree)

# ================================================================================ 
# Here starts plotting
# ================================================================================ 
print('start plotting')
plt.close('all')

# --- horizontal avarages
hca, hcb = pyic.arrange_axes(2,1, plot_cb=False, asp=1.2, fig_size_fac=1.5,
                           sharex=False, sharey=True, 
                           xlabel="", ylabel="depth [m]",
                           dfigt=0.5,
                          )
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(tbias_have, IcD.depthc)
ax.set_title('hor. ave.\ntemp. bias [$^o$C]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
ax.plot(sbias_have, IcD.depthc)
ax.set_title('hor. ave.\nsal. bias [kg/m$^3$]')

for ax in hca:
  ax.set_ylim(6000,0)

# --- zonal averages
hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=0.5, fig_size_fac=1.5,
                           sharex=True, sharey=True, 
                           xlabel="latitude", ylabel="depth [m]",
                          )
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(lat_sec, IcD.depthc, tbias_gzave, ax=ax, cax=cax, 
                  clim=1.6, cincr=0.16, contfs='auto')
ax.set_title('zon. averaged temp. bias [$^o$C]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(lat_sec, IcD.depthc, sbias_gzave, ax=ax, cax=cax, 
                  clim=0.25, cincr=0.02, contfs='auto')
ax.set_title('zon. averaged sal. bias [kg/m$^3$]')

for ax in hca:
  ax.set_facecolor('0.7')
  ax.set_ylim(6000,0)

# --- vertical averages
ccrs_proj = ccrs.PlateCarree()
hca, hcb = pyic.arrange_axes(4,2, plot_cb=True, asp=0.5, #fig_size_fac=2.,
                           sharex=True, sharey=True, 
                           xlabel="", ylabel="",
                           projection=ccrs_proj,
                          )
ii=-1

nn = 0

climT = 'sym'
climS = 'sym'

# --- temperature
for nn in range(levs.size-1):
  iz1 = np.argmin((IcD.depthc-levs[nn])**2)
  iz2 = np.argmin((IcD.depthc-levs[nn+1])**2)

  ii+=1; ax=hca[ii]; cax=hcb[ii]
  hm = pyic.shade(lon, lat, tbias_vavei[nn,:,:], ax=ax, cax=cax, clim=climT,
                  transform=ccrs_proj)
  ax.set_title('temp. bias %.fm - %.fm [$^o$C]'%(IcD.depthc[iz1],IcD.depthc[iz2]))
  pyic.plot_settings(ax, projection=ccrs_proj, template='global')

# --- salinity
for nn in range(levs.size-1):
  iz1 = np.argmin((IcD.depthc-levs[nn])**2)
  iz2 = np.argmin((IcD.depthc-levs[nn+1])**2)

  ii+=1; ax=hca[ii]; cax=hcb[ii]
  hm = pyic.shade(lon, lat, sbias_vavei[nn,:,:], ax=ax, cax=cax, clim=climS,
                  transform=ccrs_proj)
  ax.set_title('sal. bias %.fm - %.fm [kg/m$^3$]'%(IcD.depthc[iz1],IcD.depthc[iz2]))
  pyic.plot_settings(ax, projection=ccrs_proj, template='global')

plt.show()
