import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from netCDF4 import Dataset
import sys
from importlib import reload
import pyicon as pyic
reload(pyic)
import cartopy
import cartopy.crs as ccrs
import copy

run = 'nib0004'
runname = 'icon_08'
gname = 'r2b6'
lev = 'L64'

path_data     = f'/mnt/lustre01/work/mh0033/m300602/proj_vmix/icon/{runname}/icon-oes/experiments/{run}/'
path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
fpath_ckdtree = f'{path_grid}ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz'

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
mass_flux = f.variables['mass_flux'][IcD.its[it],:,:]
mass_flux_vint = mass_flux.sum(axis=0)
f.close()

# --- calculate barotropic streamfunction
bstr = pyic.calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.)

# ================================================================================ 
# Option 1: normal interpolation
# ================================================================================ 
# ----- interpolate from vertex grid to regular grid
lon, lat, bstri = pyic.interp_to_rectgrid(bstr, fpath_ckdtree, coordinates='vlat vlon')

# ================================================================================ 
# Option 2: interpolate from vertex to center but otherwise use origional grid 
# (crop region to make it faster)
# ================================================================================ 
if True:
  # --- crop grid to region
  lon_reg = [-181., 181]
  lat_reg = [50., 90.]
  clon, clat, vertex_of_cell, edge_of_cell, ind_reg = \
    pyic.crop_tripolar_grid(lon_reg, lat_reg, IcD.clon, IcD.clat, IcD.vertex_of_cell, IcD.edge_of_cell)
  Tri = matplotlib.tri.Triangulation(IcD.vlon, IcD.vlat, triangles=vertex_of_cell)
  Tri, mask_bt = pyic.mask_big_triangles(IcD.vlon, vertex_of_cell, Tri)
  bstr_cut = bstr[ind_reg] # this is still on vgrid and should not be used
else:
  # --- use uncropped
  Tri = IcD.Tri
  bstr_cut = bstr  # this is still on vgrid and should not be used
# ----- interpolate to cell center
bstr_cent = bstr[vertex_of_cell].sum(axis=1)/3.

# ================================================================================ 
# Option 3: use new grid optimized for North Polar projections
# ================================================================================ 
fpath_ckdtree = IcD.path_ckdtree+'misk/r2b6_np_60N-90N_10km.npz'
ddnpz = np.load(fpath_ckdtree)
Lon_np = ddnpz['Lon_np']
Lat_np = ddnpz['Lat_np']
bstri2 = pyic.apply_ckdtree(bstr, fpath_ckdtree, coordinates='vlat vlon')
bstri2 = bstri2.reshape(Lon_np.shape)

# ================================================================================ 
# Here starts plotting
# ================================================================================ 
plt.close("all")

ccrs_proj = ccrs.NorthPolarStereo()
Lon, Lat = np.meshgrid(lon, lat)

# --- barotropic streamfunction
hca, hcb = pyic.arrange_axes(2,2, plot_cb=True, asp=1.0, fig_size_fac=1.5,
                               sharex=False, sharey=True, xlabel="", ylabel="",
                               projection=ccrs_proj,
                            )
ii=-1

# normal contours
clim = 15
cincr = 1
conts = np.arange(-clim, clim, cincr)
conts = conts[conts!=0.]

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(lon, lat, bstri, ax=ax, cax=cax, transform=ccrs.PlateCarree(),
                  clim=clim, cincr=cincr) 
ax.set_title('Option 1:\nnormal interpolation')

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(Tri, bstr_cent, ax=ax, cax=cax, transform=ccrs.PlateCarree(),
                  clim=clim, cincr=cincr) 
ax.set_title('Option 2:\ninterp. to triang. center')

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(Lon_np, Lat_np, bstri2, ax=ax, cax=cax, transform=ccrs.PlateCarree(),
                  clim=clim, cincr=cincr) 
ax.set_title('Option 3:\ninterp. to NP optimized grid')

ii+=1; ax=hca[ii]; cax=hcb[ii]
hm = pyic.shade(Tri, bstr_cut, ax=ax, cax=cax, transform=ccrs.PlateCarree(),
                  clim=clim, cincr=cincr) 
ax.set_title('no interp:\nnot correct')

for ax in hca:
  pyic.plot_settings(ax=ax, xlocs=np.arange(-180.,181.,45.), ylocs=np.arange(50.,91.,10.), xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True)
  # zoom
  ax.set_xlim((92096.170171821, 985104.9380667121))
  ax.set_ylim((-136218.3778475427, 958437.5311849053))

ccrs_proj = ccrs.PlateCarree()

# --- barotropic streamfunction
hca, hcb = pyic.arrange_axes(1,2, plot_cb=True, asp=0.5, fig_size_fac=1.5,
                               sharex=False, sharey=True, xlabel="", ylabel="",
                               projection=ccrs_proj,
                            )
ii=-1

for kk in range(2):
  ii+=1; ax=hca[ii]; cax=hcb[ii]
  
  # normal contours
  clim = 60
  cincr = 10
  conts = np.arange(-clim, clim, cincr)
  conts = conts[conts!=0.]
  hm = pyic.shade(lon, lat, bstri, ax=ax, cax=cax, transform=ccrs_proj,
                    clim=clim, cincr=cincr, conts=conts) 
  
  # extra contours southern ocean
  conts = np.arange(60,400,40)
  hm = pyic.shade(lon, lat, bstri, ax=ax, cax=cax, transform=ccrs_proj,
                    conts=conts, use_pcol=False, contcolor='0.7') 

# title and limits
ax = hca[0]
ax.set_title('barotropic streamfunction [Sv]')
xlim = [-180,180]
ylim = [-90,90]
pyic.plot_settings(ax, xlim, ylim, projection=ccrs_proj)

ax = hca[1]
ax.set_title('North Atlantic zoom')
xlim = [-80,0]
ylim = [30,70]
pyic.plot_settings(ax, xlim, ylim, projection=ccrs_proj)

plt.show()
