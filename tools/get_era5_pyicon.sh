#!/bin/bash

#---------------------------------------------------------------
# Download script for ERA5 data to be used in pyicon
# G. Boeloeni, 24.11.2022
#---------------------------------------------------------------

#---------------------------------------------------------------
# This script is using era5cli:
# https://era5cli.readthedocs.io/en/stable/index.html
# To install it visit the above page. Alternatively 
# it can be installed via miniconda:
# conda install -c conda-forge era5cli
#---------------------------------------------------------------

era5cli = "/project/d121/ppothapa/miniconda3/bin/era5cli"

years_sta=1979
years_end=2021

vars2d='                                
surface_pressure                        
mean_sea_level_pressure                 
2m_temperature                          
2m_dewpoint_temperature                 
skin_temperature
sea_surface_temperature                 
10m_u_component_of_wind                 
10m_v_component_of_wind                 
10m_wind_speed                          
eastward_turbulent_surface_stress       
northward_turbulent_surface_stress      
eastward_gravity_wave_surface_stress    
northward_gravity_wave_surface_stress   
total_column_water_vapour               
total_cloud_cover
total_precipitation                     
large_scale_precipitation               
convective_precipitation                
surface_runoff                          
evaporation                             
surface_sensible_heat_flux              
surface_latent_heat_flux                
toa_incident_solar_radiation            
top_net_solar_radiation                 
top_net_thermal_radiation               
surface_net_solar_radiation             
surface_net_thermal_radiation           
surface_solar_radiation_downwards       
surface_thermal_radiation_downwards'

vars3d='
u_component_of_wind
v_component_of_wind
vertical_velocity
vorticity
divergence
temperature
geopotential
relative_humidity
specific_humidity
specific_rain_water_content
specific_snow_water_content
fraction_of_cloud_cover
specific_cloud_ice_water_content
specific_cloud_liquid_water_content
potential_vorticity
ozone_mass_mixing_ratio'

echo ''
echo 'Downloading ERA5 data for the period: '${years_sta}'-'${years_end}
echo ''
echo '... getting 2D variables ...'
for var in ${vars2d}
do
  echo '...' ${var}
  /project/d121/ppothapa/miniconda3/bin/era5cli monthly --startyear ${years_sta} --endyear ${years_end} --merge --variables ${var} 1>>log.2d 2>&1
done

echo ''
echo '... getting 3D variables ...'
for var in ${vars3d}
do
  echo '...' ${var}
  /project/d121/ppothapa/miniconda3/bin/era5cli monthly --startyear ${years_sta} --endyear ${years_end} --merge --variables ${var} 1>>log.3d 2>&1
done

exit
