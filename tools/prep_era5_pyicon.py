#---------------------------------------------------------------
# Prepare ERA5 data as reference for pyicon
# G. Boeloeni, 30/11/2022
#
# To run this code:
#
# 1) set 'User settings' below
#
# 2) type:
# conda activate pyicon_py39
# python prep_era5_pyicon.py
#---------------------------------------------------------------

import sys, glob, os 
import numpy as np
from scipy import interpolate
import xarray as xr

#----------------------- User settings -------------------------

# input path (ERA5 data 1959-2021)
#pathIn = '/hpc/uwork/icon-sml/pyICON/ERA5/'
#pathIn = '/project/d121/ppothapa/data_for_pyicon/era5_observations/'
pathIn = '/scratch/snx3000/ppothapa/era5_observations/'


# output path
#pathOut = './'
#pathOut = '/project/d121/ppothapa/data_for_pyicon/Era5_python_output/'
pathOut = '/scratch/snx3000/ppothapa/Era5_python_output/'


# start and end year for the time mean
yearStart = 1979
yearEnd   = 1982

#dateStart = 2004-04-01
#dateEnd   = 2006-02-01

# prefix
prefix = 'era5'

# time average interval
avgInt = 'monthly'

# do we need means over djf / jja?
do_djf = False
do_jja = False

# output resolution (deg)
res = 1.0 # can be chosen arbitrarily but ERA5's resolution 
          # is 0.25x0.25 i.e. a higher resolution than that 
          # will add only noise

plev1 = [250, 300, 500, 700, 850, 1000]
          # pressure levels at which horizontal plots will 
          # be provided in pyicon
          # Note: pressure levels (plev2) for the lat-pres 
          # cross-section plots is read from the input

# define list of 2D variables
varFnameList2D = [
'surface_pressure',
'mean_sea_level_pressure',
'2m_temperature',
'2m_dewpoint_temperature',
'skin_temperature',
'sea_surface_temperature',
'10m_u_component_of_wind',
'10m_v_component_of_wind',
'10m_wind_speed',
'eastward_turbulent_surface_stress',
'northward_turbulent_surface_stress',
'eastward_gravity_wave_surface_stress',
'northward_gravity_wave_surface_stress',
'total_column_water_vapour',
'total_cloud_cover',
'total_precipitation',
'large_scale_precipitation',
'convective_precipitation',
'surface_runoff',
'evaporation',
'surface_sensible_heat_flux',
'surface_latent_heat_flux',
'toa_incident_solar_radiation',
'top_net_solar_radiation',
'top_net_thermal_radiation',
'surface_net_solar_radiation',
'surface_net_thermal_radiation',
'surface_solar_radiation_downwards',
'surface_thermal_radiation_downwards'
]

# define list of 3D variables
varFnameList3D = [
#'u_component_of_wind',
#'v_component_of_wind',
'temperature',
'geopotential',
'relative_humidity',
'specific_humidity',
'fraction_of_cloud_cover',
'specific_cloud_ice_water_content',
'specific_cloud_liquid_water_content',
#-------------------------------------
# additional variables available in the raw 
# data but not used so far in pyicon:
#-------------------------------------
#'vertical_velocity',
#'vorticity',
#'divergence',
#'specific_rain_water_content',
#'specific_snow_water_content',
#'potential_vorticity',
#'ozone_mass_mixing_ratio'
]

#------------------------- Functions ---------------------------

def prepcoords():

  # This function serves just aesthetics. Without calling it 
  # everything should still work but then variables and 
  # coordinates come in a mixed order in the output file.

  print('')
  print('... preparing dimensions & cooridnates ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # get varFname as 1st element of varFnameList3D
  varFname = varFnameList3D[0]

  # input file
  dataIn = f'{pathIn}{prefix}_{varFname}_*_{avgInt}.nc'

  # load input as xarray.Dataset
  dsIn = xr.open_mfdataset(dataIn)

  # rename dimensions & coordinates
  dsIn = dsIn.rename_dims({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})
  dsIn = dsIn.rename_vars({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})

  # shift lon by -180 degrees
  mylonshft = mylon - 180

  # create output xarray.Dataset with coordinates
  dsOut = xr.Dataset(
     coords=dict(plev1=(['plev1'], plev1), 
                 plev2=(['plev2'], dsIn.plev2.values), 
                 lon=(['lon'], mylonshft), 
                 lat=(['lat'], mylat),
                 time=(['time'], dsIn.time.values),),
     attrs=dict(description='ERA5 data for pyicon'),
  )

  # write xarray.Dataset with coordinates to output file
  dsOut.to_netcdf(dataOut, encoding={'plev1': {'dtype': 'int32'},
                                     'plev2': {'dtype': 'int32'},
                                     'lon':   {'dtype': 'float32'},
                                     'lat':   {'dtype': 'float32'},
                                     'time':  {'dtype': 'int64'}}
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

  # loop over 2D fields
  for varFname in varFnameList2D:

    print('... processing', varFname)

    # input file
    dataIn = f'{pathIn}{prefix}_{varFname}_*_{avgInt}.nc'

    # load input as xarray.Dataset
    dsIn = xr.open_mfdataset(dataIn, chunks={'time': 12})

    # rename dimensions & coordinates
    dsIn = dsIn.rename_dims({'longitude': 'lon', 'latitude': 'lat'})
    dsIn = dsIn.rename_vars({'longitude': 'lon', 'latitude': 'lat'})

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

    # read variable name
    varName=list(dsIn.data_vars)
    
    # interpolate to output resolution
    imeth = 'cubic'
    if varName == ['sst']:
      # missing values (land points) can't be handled by cubic
      imeth = 'linear'
    dsOut = dsIn.interp(
       lon=mylon, 
       lat=mylat, 
       method=imeth, 
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

def prep3dmaps():

  # Prepare 3D fields on a reduced set of pressure 
  # levels (plev1), i. e. 2D maps at plev1 levels.

  print('')
  print('... processing 3D fields on pressure levels plev1 ...')
  if yearEnd-yearStart > 15:
    print('... !!! Beware: this might take a while for long periods !!! ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # loop over 3D fields
  for varFname in varFnameList3D:

    print('... processing', varFname)

    # input file
    dataIn = f'{pathIn}{prefix}_{varFname}_*_{avgInt}.nc'

    # load input as xarray.Dataset
    dsIn = xr.open_mfdataset(dataIn, chunks={'time': 12, 'level':1})

    # rename dimensions & coordinates
    dsIn = dsIn.rename_dims({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})
    dsIn = dsIn.rename_vars({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})

    # select time period 
    dsIn = dsIn.sel(time=slice(dateStart, dateEnd))
    dsIn = dsIn.sel(plev2=plev1)

    # optionally select djf/jja months
    if do_djf:
      dsIn = dsIn.sel(time=is_djf(dsIn['time.month']))
    if do_jja:
      dsIn = dsIn.sel(time=is_jja(dsIn['time.month']))

    # rename dimensions & coordinates
    dsIn = dsIn.rename_dims({'plev2': 'plev1'})
    dsIn = dsIn.rename_vars({'plev2': 'plev1'})

    # chunking, so that we fit to memory 
    # and hopefully get some speedup...
    dsIn = dsIn.chunk(chunks={'time': 12, 'plev1': 1})
 
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

    # reverse pressure levels
    dsOut = dsOut.reindex(plev1=list(reversed(dsOut.plev1)))

    # write (append) output xarray.Dataset to output file
    dsOut.to_netcdf(dataOut, mode='a')

    # close xarray.Datasets
    dsIn.close()
    dsOut.close()

  return

def prep3dcrss():

  # Prepare lat-pressure cross-sections of the 
  # zonal-mean of 3D fields.

  print('')
  print('... processing 3D fields to get cross-sections of the zonal-mean ...')
  if yearEnd-yearStart > 15:
    print('... !!! Beware: this might take a while for long periods !!! ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # loop over 3D fields
  for varFname in varFnameList3D:

    print('... processing', varFname)

    # input file
    dataIn = f'{pathIn}{prefix}_{varFname}_*_{avgInt}.nc'

    # load input as xarray.Dataset
    dsIn = xr.open_mfdataset(dataIn, chunks={'time': 12, 'level':4})

    # rename dimensions & coordinates & var
    dsIn = dsIn.rename_dims({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})
    dsIn = dsIn.rename_vars({'level': 'plev2', 'longitude': 'lon', 'latitude': 'lat'})

    # rename variable to represent zonal-mean
    varName=list(dsIn.data_vars)[0]
    dsIn = dsIn.rename_vars({varName: varName+'_zm'})

    # select time period 
    dsIn = dsIn.sel(time=slice(dateStart, dateEnd))

    # optionally select djf/jja months
    if do_djf:
      dsIn = dsIn.sel(time=is_djf(dsIn['time.month']))
    if do_jja:
      dsIn = dsIn.sel(time=is_jja(dsIn['time.month']))

    # chunking, so that we fit to memory 
    # and hopefully get some speedup...
    dsIn = dsIn.chunk(chunks={'time': 12, 'plev2': 4})
 
    # calculate time-mean zonal-mean
    dsIn = dsIn.mean(dim=['time', 'lon'])

    # interpolate to output resolution
    dsOut = dsIn.interp(
       lat=mylat, 
       method='cubic', 
       kwargs={"fill_value": "extrapolate"}
    )

    # reverse pressure levels
    dsOut = dsOut.reindex(plev2=list(reversed(dsOut.plev2)))

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
  # needed for # validation. Hence, the output form of this 
  # function will always be the same irrespectively of the 
  # resolution and the period you defined via res/dateStart/dateEnd.

  print('')
  print('... processing global-mean time-series of 2D fields ...')

  # enable keeping attributes
  xr.set_options(keep_attrs=True)

  # loop over 2D fields
  for varFname in varFnameList2D:

    print('... processing', varFname)

    # input file
    dataIn = f'{pathIn}{prefix}_{varFname}_*_{avgInt}.nc'

    # load input as xarray.Dataset
    dsIn = xr.open_mfdataset(dataIn, chunks={'time': 12})

    # rename dimensions & coordinates
    dsIn = dsIn.rename_dims({'longitude': 'lon', 'latitude': 'lat'})
    dsIn = dsIn.rename_vars({'longitude': 'lon', 'latitude': 'lat'})

    # rename variable to represent global mean ts
    varName=list(dsIn.data_vars)[0]
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
  print('Preparing ERA5 data for pyicon for the period:', yearStart,'-', yearEnd, ' DJF')
elif do_jja:
  print('')
  print('Preparing ERA5 data for pyicon for the period:', yearStart,'-', yearEnd, ' JJA')
else:
  print('')
  print('Preparing ERA5 data for pyicon for the period:', yearStart,'-', yearEnd)


# dates corresponding to yearStart, yearEnd
dateStart = np.datetime64(str(yearStart)+'-01-01')
dateEnd   = np.datetime64(str(yearEnd)+'-12-01')

# lon, lat at output resolution
mylon = np.arange(0,360,res)
#mylat = np.arange(-90,90+res,res)
mylat = np.arange(-90,90,res)

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

# prepare 3D maps
prep3dmaps()

# prepare zonal-mean cross-sections
prep3dcrss()

# prepare global-mean time-series of 2D fields
prepgmts()

print('')
print('All done!')
print('Output file: ', dataOut)

#---------------------------------------------------------------
