import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
from importlib import reload
import pyicon as pyic
reload(pyic)
#import cartopy.crs as ccrs
#import cartopy
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

run = 'nib0004'
runname = 'icon_08'
gname = 'r2b6'
lev = 'L64'

path_data     = f'/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/{runname}/icon-oes/experiments/{run}/'
path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
path_ckdtree  = f'{path_grid}ckdtree/'
fpath_ckdtree = f'{path_grid}ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz'
fpath_fx      = f'{path_grid}{gname}_{lev}_fx.nc'

IcD = pyic.IconData(
               fname        = run+'_????????????????.nc', 
               path_data    = path_data,
               path_grid    = path_grid,
               gname        = gname,
               lev          = lev,
               #path_ckdtree = path_ckdtree,
               #section_name = '',
               #fpath_fx     = fpath_fx,
               do_triangulation = True,
               omit_last_file   = False
              )


# --- specify time step
it = np.argmin(np.abs(IcD.times-np.datetime64('2295-01-01T00:00:00')))
# --- specify depth level
iz = np.argmin(np.abs(IcD.depthc-4000.))

# --- load data
f = Dataset(IcD.flist_ts[it], 'r')
to = f.variables['to'][IcD.its[it],iz,:]
f.close()

# --- mask land values for tgrid plot
mask_land = to==0.0
IcD.Tri.set_mask(IcD.mask_bt+mask_land)

# --- interpolate to rectangular grid
lon, lat, toi = pyic.interp_to_rectgrid(to, fpath_ckdtree, coordinates='clat clon')

# --- here starts plotting
plt.close('all')

hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, sasp=0.5)
ii=-1

print('rectangular grid plot')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.shade(lon, lat, toi, ax=ax, cax=cax, clim='auto')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('temperature at %s and %.1fm depth'%(IcD.times[it], IcD.depthc[iz]))
cax.set_title('$^o$C')

print('triangular grid plot')

ii+=1; ax=hca[ii]; cax=hcb[ii]
pyic.trishade(IcD.Tri, to, ax=ax, cax=cax, clim='auto')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.set_title('temperature original grid')
cax.set_title('$^o$C')

plt.show()

