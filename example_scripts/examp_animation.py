import sys
import numpy as np
import matplotlib 
# --- avoid errors with display when executed by sbatch
if len(sys.argv)>1 and sys.argv[1]=='--no_backend':
  print('apply: matplotlib.use(\'Agg\')')
  matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyicon as pyic
import cartopy.crs as ccrs
import glob, os
import xarray as xr

""" This script shows how to make an animation from ICON data.

In this script, a 2D variable which changes in time is plotted as separate 
images for each time instance. Finally, these images can be glued together
to an animation by using ffmpeg.

It is recommended to first run the script in an ipython session and
uncomment the 'plt.draw(); sys.exit()' lines below. The plot can now be
adjusted until it looks like expected. Afterwards the 'plt.draw(); sys.exit()'
lines can be commented out again and the script can be run 1) either in
the ipython session or 2) be submitted to the queue by 
`sbatch run_examp_animation.sh`
to produce all the figures. Finally, the figures can be glued together
into an animation as detailed by the output of the script.
"""

do_ocean_data = True

run = 'dpp0052'
path_data = f'/work/mh0287/k203123/GIT/icon-aes-dyw3/experiments/{run}/'
if do_ocean_data:
  fpath_ckdtree = '/home/mpim/m300602/work/icon/grids/r2b9_oce_r0004/ckdtree/rectgrids/r2b9_oce_r0004_res0.30_180W-180E_90S-90N.nc'
else:
  fpath_ckdtree = '/home/mpim/m300602/work/icon/grids/r2b9_atm_r0015/ckdtree/rectgrids/r2b9_atm_r0015_res0.30_180W-180E_90S-90N.nc'

# --- initialize timer
timi = pyic.timing([0], 'start')

# --- set path where figures are storred 
savefig = True
path_fig = '/home/mpim/m300602/work/movies/%s/' % (__file__.split('/')[-1][:-3])
figure_prefix = __file__.split('/')[-1][:-3]
nnf=0

help_txt = f"""
Your save figures can be found in 
{path_fig} 
to find the saved figures. 

In this directory, you can create an animation from your figures by using ffmpeg. 
You can play around with the frame rate specified by -r to get more or less 
frames per second.

ffmpeg -r 5 -f image2 -i {figure_prefix}_%04d.jpg -c:v mpeg4 -b:v 12000k {figure_prefix}.mp4

If you are missing a proper ffmpeg, you might want to install it by using 
conda.

On mistral you can also use:
ffmpeg="/work/mh0033/m300602/miniconda3/envs/pyicon_py39_exp/bin/ffmpeg"
${{ffmpeg}} -r 5 -f image2 -i {figure_prefix}_%04d.jpg -c:v mpeg4 -b:v 12000k {figure_prefix}.mp4
"""
print(help_txt)

try:
  os.makedirs(path_fig)
except:
  pass

# --- open files with xarray
if do_ocean_data:
  fpath = f'{path_data}/{run}_oce_3dlev_P1D_????????????????.nc'
else:
  fpath = f'{path_data}/{run}_atm_2d_ml_????????????????.nc'
flist = np.array(glob.glob(fpath))
flist.sort()
flist = flist[:10] # if desired, reduce number of files to be read
mfdset_kwargs = dict(
  combine='nested', concat_dim='time', 
  data_vars='minimal', coords='minimal', 
  compat='override', join='override',
  parallel=True,
)
if True:
  timi = pyic.timing(timi, 'open_mfdataset')
  ds = xr.open_mfdataset(flist, **mfdset_kwargs, chunks={'time': 1})
  # --- correct time steps (only for atm data)
  if not do_ocean_data:
    ds['time'] = pyic.nctime_to_datetime64(ds.time.data, time_mode='float2date')

# --- load data
timi = pyic.timing(timi, 'load data')
step = 10 # dummy step
if do_ocean_data:
  data_xr = ds['to'].isel(depth=0, time=step)
else:
  data_xr = ds['ts'].isel(time=step) - 273.15

# --- interpolate to regular grid
timi = pyic.timing(timi, 'interp_to_rectgrid_xr')
data_xr_interp = pyic.interp_to_rectgrid_xr(data_xr, fpath_ckdtree)

# --- specify which steps should be plotted
steps = np.arange(ds.time.size)

# -------------------------------------------------------------------------------- 
# Here starts plotting
# -------------------------------------------------------------------------------- 
timi = pyic.timing(timi, 'initialize plot')

plt.close('all')
ccrs_proj = ccrs.PlateCarree()

hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, fig_size_fac=2., asp=0.5, projection=ccrs_proj)
ii=-1

ii+=1; ax=hca[ii]; cax=hcb[ii]
# --- pcolormesh plot (wrapped by pyic.shade) which is updated later on
hm1 = pyic.shade(data_xr_interp.lon, data_xr_interp.lat, data_xr_interp, 
                 ax=ax, cax=cax, clim=[-2, 32])
ax.set_title('surface temperature [deg C]')
# --- time string (also updated)
bbox=dict(facecolor='w', alpha=1., edgecolor='none')
ht = ax.text(0.02, 0.92, str(data_xr_interp.time.data)[:16], transform=ax.transAxes, bbox=bbox)

for ax in hca:
  pyic.plot_settings(ax, template='global')
 
# --- (optional) verify whether plot looks like you want it to look
#plt.show()
#sys.exit()

def update_fig(step):
  # --- load new data
  if do_ocean_data:
    data_xr = ds['to'].isel(depth=0, time=step)
  else:
    data_xr = ds['ts'].isel(time=step) - 273.15
  # --- interp to rectgrid 
  data_xr_interp = pyic.interp_to_rectgrid_xr(data_xr, fpath_ckdtree)

  # --- update plot
  hm1[0].set_array(data_xr_interp.data.flatten())
  ht.set_text(str(data_xr_interp.time.data)[:16])
  return

# --- parallel execution of script (usually no need to change anything below)
# >>> start parallel (comment out if not wanted) <<<
try:
  # --- load mpi if possible and define `rank`=processor ID and `npro`=number of total processors
  from mpi4py import MPI
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  npro = comm.Get_size()
except:
  print('::: Warning: Proceeding without mpi4py! :::')
  rank = 0
  npro = 1
print('proc %d/%d: Hello world!' % (rank, npro))

# --- modify previously defined variable `steps` such that each processor gets
#     its own `steps` variable containing only the steps which it should work on 
list_all_pros = [0]*npro
for nn in range(npro):
  list_all_pros[nn] = steps[nn::npro]
steps = list_all_pros[rank]
# >>> end parallel (comment out if not wanted) <<<

# --- start loop
for nn, step in enumerate(steps):
  print('proc %d/%d: Step %d/%d' % (rank, npro, nn, len(steps)))
  timi = pyic.timing(timi, 'loop step')

  update_fig(step)

  nnf+=1
  if savefig:
    fpath = '%s%s_%04d.jpg' % (path_fig,figure_prefix, step)
    print('save figure: %s' % (fpath))
    plt.savefig(fpath, dpi=250)

timi = pyic.timing(timi, 'all done')

print(help_txt)
