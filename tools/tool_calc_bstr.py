#!/usr/bin/env python3

print('Starting tool_calc_bstr.py...')

import matplotlib
#matplotlib.use('Agg')
import shutil
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import pyicon as pyic
import mpi4py
import xarray as xr
import sys
import argparse

help_text = """
Derive barotropic streamfunction from variable for all time instances of mass_flux specified in `fpathi` and save results `path_datao`.

Basic usage:
tool_streamfunction.py /path/to/model/output/runname*.nc' '/path/to/output/for/bstr/'
"""

# --- read input arguments
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

# --- necessary arguments
parser.add_argument('path_datai', metavar='path_datai', type=str,
                    help='path of input data')
parser.add_argument('fname', metavar='fname', type=str,
                    help='file name of input data (wildcards can be used)')
parser.add_argument('path_datao', metavar='path_datao', type=str,
                    help='path of output data')

# --- optional arguments
parser.add_argument('--gname', type=str, default='r2b6_oce_r0004', 
                    help='name of grid')
parser.add_argument('--path_grid', type=str, default='/mnt/lustre01/work/mh0033/m300602/icon/grids/', 
                    help='path of grid files')
parser.add_argument('--verbose', type=bool, default=True, 
                    help='give some more information about current status')
parser.add_argument('--omit_last_file', type=bool, default=False, 
                    help='whether to omit last file of list (e.g. for running exps)')
iopts = parser.parse_args()

# --- converting input arguments
# --- necessary arguments
path_datao = iopts.path_datao + '/'
path_datai = iopts.path_datai + '/'
fname = iopts.fname
fpathi = path_datai+fname
# --- optional arguments
gname = iopts.gname
path_grid = iopts.path_grid
verbose = iopts.verbose
omit_last_file = iopts.omit_last_file

run = fpathi.split('/')[-1].split('_')[0]
#path_data = '/'.join(fpathi.split('/')[:-1]) + '/'
#fname = fpathi.split('/')[-1]

path_grid += gname+'/'

### -------------------------------------------------------------------------------- 
### start user input
### -------------------------------------------------------------------------------- 
### path of output file
##path_datao = '/scratch/m/m300602/tmp/bstr/'
##
### input files
##fpathi = '/work/mh0287/m211032/Icon/Git_Icon/icon.oes.20200506/experiments/slo1325/slo1325_oce_def_????????????????.nc'
##
### --- derive from input
##run = fpathi.split('/')[-1].split('_')[0]
##path_datai = '/'.join(fpathi.split('/')[:-1]) + '/'
##fname = fpathi.split('/')[-1]
##
### name of horizontal grid
##gname       = 'r2b6_oce_r0004'
### basic path to grid files and ckdtrees
##path_grid   = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
##omit_last_file = False
##verbose = True
##
### -------------------------------------------------------------------------------- 
### end user input
### -------------------------------------------------------------------------------- 

# --- initialize IconData object
IcD = pyic.IconData(
               fname        = fname,
               path_data    = path_datai,
               path_grid    = path_grid,
               gname        = gname,
               do_triangulation       = False,
               omit_last_file         = omit_last_file,
               load_vertical_grid     = False,
               load_triangular_grid   = True, # needed for bstr
               load_rectangular_grid  = False,
               calc_coeff             = False,
               verbose                = verbose,
              )
naf = IcD.flist.size

if verbose:
  print('Derive streamfunction for the following list of files:')
  print(IcD.flist)

# --- mask for land points
vert_mask =  (IcD.cell_sea_land_mask[IcD.cells_of_vertex]>-2).sum(axis=1)>0

for nf, fpath in enumerate(IcD.flist):
  # --- open input file
  f = Dataset(IcD.flist[nf], 'r')

  # --- initialize DataSet and add coordinates
  ds = xr.Dataset()
  ds['vlon'] = xr.DataArray(IcD.vlon, dims=['vert'])
  ds['vlat'] = xr.DataArray(IcD.vlat, dims=['vert'])

  nt = f.variables['time'].size
  for l in range(nt):
    print(f'nf = {nf+1}/{naf}, l = {l+1}/{nt}:  {fpath}')

    # --- load time
    time = pyic.nctime_to_datetime64(f.variables['time'], time_mode='num2date')[l]

    # --- load mass_flux and derive stream function
    mass_flux = f.variables['mass_flux'][l,:,:]
    mass_flux_vint = mass_flux.sum(axis=0)
    bstr = pyic.calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.)
    bstr[vert_mask] = 0.

    # --- create DataArray
    da = xr.DataArray(bstr[np.newaxis,:], 
      dims=['time', 'vert'], 
      coords=dict(time=[time]),
      attrs={'units': 'Sv', 'long_name': 'barotropic streamfunction'},
    )

    # --- concatenate DataArray
    # (since concatenating a netcdf file seems not possible with xr)
    if l==0:
      da_all = da
    else:
      da_all = xr.concat([da_all, da], dim='time')

  # --- write to netcdf
  ds['bstr'] = da_all
  tstr = fpath.split('/')[-1].split('_')[-1][:-3]
  fpath_o = f'{path_datao}/{run}_bstr_{tstr}.nc'
  print(f'Writing file {fpath_o}.')
  ds.to_netcdf(fpath_o)

  f.close()

print('All done!')
