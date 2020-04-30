import sys, glob, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pyicon as pyic
import cartopy.crs as ccrs

run       = 'jkr0042'
gname_atm = 'r2b4a'
lev_atm   = 'L84'
rgrid_name = 'global_0.3'
t1 = np.datetime64('1780-02-01')
t2 = np.datetime64('1790-01-01')

path_data     = '/work/mh0287/users/juergen/icon-oes/experiments/'+run+'/'
path_grid_atm = '/mnt/lustre01/work/mh0033/m300602/icon/grids/'+gname_atm+'/'

tstep     = '????????????????'

fname = '%s_atm_2d_ml_%s.nc' % (run, tstep)
print('Dataset %s' % (fname))
IcD_atm2d = pyic.IconData(
               fname        = fname,
               path_data    = path_data,
               path_grid    = path_grid_atm,
               gname        = gname_atm,
               lev          = lev_atm,
               rgrid_name   = rgrid_name,
               do_triangulation = True,
               omit_last_file = False,
               load_vertical_grid = False,
               time_mode = 'float2date',
              )
fpath_ckdtree_atm = IcD_atm2d.rgrid_fpath_dict[rgrid_name]

tauu, it_ave   = pyic.time_average(IcD_atm2d, 'tauu', t1=t1, t2=t2, iz='all')
tauv, it_ave   = pyic.time_average(IcD_atm2d, 'tauv', t1=t1, t2=t2, iz='all')

lon, lat, tauui = pyic.interp_to_rectgrid(tauu, fpath_ckdtree_atm, coordinates='clat clon')
lon, lat, tauvi = pyic.interp_to_rectgrid(tauv, fpath_ckdtree_atm, coordinates='clat clon')

# ================================================================================ 
# Here starts plotting
# ================================================================================ 
plt.close('all')

# --- PlateCarree projection
ccrs_proj = ccrs.PlateCarree()
hca, hcb = pyic.arrange_axes(2,1, asp=0.5, projection=ccrs_proj, fig_size_fac=1.5)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, tauui, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax, template='global', land_facecolor='none')
ax.set_title('zonal wind stress [N/m$^2$]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, tauvi, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax, template='global', land_facecolor='none')
ax.set_title('meridional wind stress [N/m$^2$]')

# --- NorthPolarStereo projection
ccrs_proj = ccrs.NorthPolarStereo()
hca, hcb = pyic.arrange_axes(2,1, asp=1.0, projection=ccrs_proj, fig_size_fac=1.5,
                             sharex=True, sharey=True)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, tauui, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
ax.set_title('zonal wind stress [N/m$^2$]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, tauvi, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
ax.set_title('meridional wind stress [N/m$^2$]')

# --- NorthPolarStereo projection orig grid
ccrs_proj = ccrs.NorthPolarStereo()
hca, hcb = pyic.arrange_axes(2,1, asp=1.0, projection=ccrs_proj, fig_size_fac=1.5,
                             sharex=True, sharey=True)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(IcD_atm2d.Tri, tauu, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
ax.set_title('zonal wind stress [N/m$^2$]')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(IcD_atm2d.Tri, tauv, ax=ax, cax=cax, clim=0.5, projection=ccrs.PlateCarree())
pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
ax.set_title('meridional wind stress [N/m$^2$]')


plt.show()
