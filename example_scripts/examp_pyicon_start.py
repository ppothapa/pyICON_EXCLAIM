import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import pyicon as pyic
import cartopy

run = 'nib0004'
runname = 'icon_08'
gname = 'r2b6_oce_r0004'
lev = 'L64'

path_data     = f'/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/{runname}/icon-oes/experiments/{run}/'
path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
path_ckdtree  = f'{path_grid}ckdtree/'
fpath_ckdtree = f'{path_grid}ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz'
fpath_ckdtree_sec = f'{path_grid}ckdtree/sections/{gname}_nps300_30W80S_30W80N.npz'
fpath_fx      = f'{path_grid}{gname}_{lev}_fx.nc'

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
iz = np.argmin(np.abs(IcD.depthc-100.))

# --- load data
f = Dataset(IcD.flist_ts[it], 'r')
to = f.variables['to'][IcD.its[it],:,:]
f.close()

# --- mask land values for tgrid plot
mask_land = to[iz,:]==0.0
IcD.Tri.set_mask(IcD.mask_bt+mask_land)

# --- interpolate to rectangular grid
lon, lat, toi = pyic.interp_to_rectgrid(to, fpath_ckdtree, coordinates='clat clon')

# --- extract a section
lon_sec, lat_sec, dist_sec, to_sec = pyic.interp_to_section(to, fpath_ckdtree_sec, coordinates='clat clon')

# --- here starts plotting
plt.close('all')
projection = None
#projection = cartopy.crs.PlateCarree()

# --- horizontal plots
hca, hcb = pyic.arrange_axes(1,2, plot_cb=True, asp=0.5, fig_size_fac=1.5, 
                             xlabel='longitude', ylabel='latitude', projection=projection)
ii=-1

print('rectangular grid plot')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, toi[iz,:], ax=ax, cax=cax, clim='auto', projection=projection)
ax.set_title('temperature at %s and %.1fm depth'%(IcD.times[it], IcD.depthc[iz]))
cax.set_title('$^o$C')

print('triangular grid plot')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(IcD.Tri, to[iz,:], ax=ax, cax=cax, clim='auto', projection=projection)
ax.set_title('temperature original grid')
cax.set_title('$^o$C')

#for ax in hca:
#  pyic.plot_settings(ax=ax, projection=projection, template='global')

# --- horizontal plots
hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=0.5, fig_size_fac=1.5, 
                             xlabel='latitude', ylabel='depth [m]')
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lat_sec, IcD.depthc, to_sec, ax=ax, cax=cax, clim='auto')
ax.set_title('temperature at 30$^o$W')
cax.set_title('$^o$C')
ax.set_ylim(5500,0)

plt.show()
