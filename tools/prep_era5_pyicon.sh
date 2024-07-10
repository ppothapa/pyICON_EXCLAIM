#! /bin/ksh
#=============================================================================

# daint gpu batch job parameters
# ------------------------------
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=d121
#SBATCH --job-name=prep_era5.run
#SBATCH --output=LOG.prep.era5.run.%j.o
#SBATCH --error=LOG.prep.era5.run.%j.o
#SBATCH --nodes=1
#SBATCH --time=23:00:00

echo python script starts 

python prep_era5_pyicon.py 


echo pythin scripts ends
