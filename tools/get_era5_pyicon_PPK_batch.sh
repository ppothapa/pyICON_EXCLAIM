#! /bin/ksh
#=============================================================================

# daint gpu batch job parameters
# ------------------------------
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=d121
#SBATCH --job-name=post_era5.run
##SBATCH --output=LOG.post.era5.run.%j.o
##SBATCH --error=LOG.post.era5.run.%j.o
#SBATCH --nodes=1
#SBATCH --time=24:00:00
##SBATCH --ntasks-per-node=1
#=============================

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


vars3d="
divergence
geopotential
relative_humidity
specific_humidity
specific_rain_water_content
specific_snow_water_content
fraction_of_cloud_cover
specific_cloud_ice_water_content
specific_cloud_liquid_water_content
potential_vorticity
ozone_mass_mixing_ratio"

echo ""
echo "Downloading ERA5 data for the period: '${years_sta}'-'${years_end}"
echo ''
echo "... getting 3D variables ..."
for var in ${vars3d}
do
  echo "..." ${var}
  /project/d121/ppothapa/miniconda3/bin/era5cli monthly --startyear ${years_sta} --endyear ${years_end} --merge --variables ${var} 1>>log.3d 2>&1
done

#exit
