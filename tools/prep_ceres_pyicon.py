#---------------------------------------------------------------
# Prepare CERES data as reference for pyicon
# G. Boeloeni, 31/01/2023
#
# To run this code:
#
# 1) set 'User settings' below
#
# 2) type:
# conda activate pyicon_py39
# python prep_ceres_pyicon.py
#---------------------------------------------------------------

import sys, glob, os 
import numpy as np
from scipy import interpolate
import xarray as xr

#----------------------- User settings -------------------------

# input path (CERES data 2000-2020)
pathIn = '/hpc/uwork/icon-sml/pyICON/CERES/'

# output path
pathOut = './'

# start and end year for the time mean
yearStart = 2001
yearEnd   = 2010

# prefix
prefix = 'ceres'

# do we need means over djf / jja?
do_djf = False
do_jja = False

# output resolution (deg)
res = 1.5 # can be chosen arbitrarily but CERES' resolution 
          # is 1x1 i.e. a higher resolution than that will 
          # add only noise

#------------------------- Functions ---------------------------

def prepcoords():

  # This function serves just aesthetics. Without calling it 
  # everything should still work but then variables and 
  # coordinates come in a mixed order in the output file.

  print('')
  print('... preparing dimensions & cooridnates ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # load input as xarray.Dataset
  dsIn = xr.open_mfdataset(dataIn, mask_and_scale=False)

  # shift lon by -180 degrees
  mylonshft = mylon - 180

  # create output xarray.Dataset with coordinates
  dsOut = xr.Dataset(
     coords=dict(lon=(['lon'], mylonshft), 
                 lat=(['lat'], mylat),
                 time=(['time'], dsIn.time.values),),
     attrs=dict(description='CERES data for pyicon'),
  )

  # write xarray.Dataset with coordinates to output file
  dsOut.to_netcdf(dataOut, encoding={'lon':   {'dtype': 'float32'},
                                     'lat':   {'dtype': 'float32'},
                                     'time':  {'dtype': 'int32'}}
                 )

  # close xarray.Datasets
  dsIn.close()
  dsOut.close()

  return

def prep2d():

  # Prepare 2D maps of time-mean fields.

  print('')
  print('... processing 2D fields ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # load input as xarray.Dataset
  dsIn = xr.open_mfdataset(dataIn, mask_and_scale=False, chunks={'time': 12})

  # select time period 
  dsIn = dsIn.sel(time=slice(dateStart, dateEnd))

  # optionally select djf/jja months
  if do_djf:
    dsIn = dsIn.sel(time=is_djf(dsIn['time.month']))
  if do_jja:
    dsIn = dsIn.sel(time=is_jja(dsIn['time.month']))

  # chunking, so that we fit to memory 
  # and hopefully get some speedup...
  dsIn = dsIn.chunk(chunks={'time': 12})
 
  # calculate time-mean
  dsIn = dsIn.mean(dim='time')

  # interpolate to output resolution
  dsOut = dsIn.interp(
     lon=mylon, 
     lat=mylat, 
     method='cubic', 
     kwargs={"fill_value": "extrapolate"}
  )

  # shift longitude by -180 degrees
  dsOut.coords['lon'] = (dsOut.coords['lon'] + 180) % 360 - 180
  dsOut = dsOut.sortby(dsOut.lon)

  # write (append) output xarray.Dataset to output file
  dsOut.to_netcdf(dataOut, mode='a')

  # close xarray.Datasets
  dsIn.close()
  dsOut.close()

  return

def prepgmts():

  # Prepare global-mean time-series of 2D fields. This is 
  # always done for the full period included in the input 
  # files and pyicon will select internally the actual period 
  # needed for validation. Hence, the output form of this 
  # function will always be the same irrespectively of the 
  # resolution and the period you defined via res/dateStart/dateEnd.

  print('')
  print('... processing global-mean time-series of 2D fields ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # load input as xarray.Dataset
  dsIn = xr.open_mfdataset(dataIn, mask_and_scale=False, chunks={'time': 12})

  # rename variable to represent global mean ts
  varNameList=list(dsIn.data_vars)
  for varName in varNameList:
    dsIn = dsIn.rename_vars({varName: varName+'_gmts'})

  # chunking, so that we fit to memory 
  # and hopefully get some speedup...
  dsIn = dsIn.chunk(chunks={'time': 12})

  # calculate cell area
  area = calc_cell_area(dsIn.lon.values, dsIn.lat.values)

  # create xarray.Datarray
  area_xr = xr.DataArray(area, 
            coords={'lon': dsIn.lon.values,'lat': dsIn.lat.values}, 
            dims=["lon", "lat"])
 
  # calculate cell-area weighted global-mean
  dsWg = dsIn.weighted(area_xr)
  dsOut = dsWg.mean(dim=['lon', 'lat'])

  # write (append) output xarray.Dataset to output file
  dsOut.to_netcdf(dataOut, mode='a')

  # close xarray.Datasets
  dsIn.close()
  dsOut.close()

  return

def calc_cell_area (lon, lat):

  # Calculate the area of each grid cell. This is needed 
  # for calculating a cell area weighted global mean.

  # radius of the Earth (m)
  radius = 6371000.

  # difference in longitude between neighboring 
  # cell centres (constant)
  dlon = np.radians(lon[2] - lon[1])

  # initialize area
  area = np.zeros((lon.size,lat.size))

  # calculate area
  for j in np.arange(1,np.size(lat)-1):
    lat1 = (lat[j-1] +  lat[j]  )*0.5
    lat2 = (lat[j]   +  lat[j+1])*0.5
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    # A = R^2 |sin(lat1)-sin(lat2)| |lon1-lon2| where R is earth radius (6378 km)
    area[:,j] = np.square(radius)*(np.sin(lat2)-np.sin(lat1))*dlon

  # Earth's surface area
  area_e = 4. * np.pi * np.square(radius)

  # area fraction w.r.t. Earth's surface area
  area = area / area_e

  return area

def is_djf(month):
  return (month == 12) | (month == 1) | (month == 2)

def is_jja(month):
  return (month == 6) | (month == 7) | (month == 8)

#---------------------------- Start ----------------------------

if do_djf and do_jja:
  print('')
  print('do_djf and do_jja should not be True at the same time!')
  sys.exit()

if do_djf:
  print('')
  print('Preparing CERES data for pyicon for the period:', yearStart,'-', yearEnd, ' DJF')
elif do_jja:
  print('')
  print('Preparing CERES data for pyicon for the period:', yearStart,'-', yearEnd, ' JJA')
else:
  print('')
  print('Preparing CERES data for pyicon for the period:', yearStart,'-', yearEnd)

# dates corresponding to yearStart, yearEnd
dateStart = np.datetime64(str(yearStart)+'-01-01')
dateEnd   = np.datetime64(str(yearEnd)+'-12-01')

# lon, lat at output resolution
mylon = np.arange(0,360,res)
mylat = np.arange(-90,90+res,res)

# input file
dataIn = f'{pathIn}CERES_EBAF-TOA_Ed4.2_Subset_200003-202211.nc'

# define output file
if do_djf:
  dataOut = f'{pathOut}{prefix}_pyicon_{yearStart}-{yearEnd}_djf_{res}x{res}deg.nc'
elif do_jja:
  dataOut = f'{pathOut}{prefix}_pyicon_{yearStart}-{yearEnd}_jja_{res}x{res}deg.nc'
else:
  dataOut = f'{pathOut}{prefix}_pyicon_{yearStart}-{yearEnd}_{res}x{res}deg.nc'

# prepare dims & coords
prepcoords()

# prepare 2D fields
prep2d()

# prepare global-mean time-series of 2D fields
prepgmts()

print('')
print('All done!')
print('Output file: ', dataOut)

#---------------------------------------------------------------
