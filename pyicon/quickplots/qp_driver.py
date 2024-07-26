import sys, glob, os
import argparse
#from ipdb import set_trace as mybreak

#class DS_attributes(object):
#  def __init__(self, prefix, output_freq='yearly'):
#    self.prefix = prefix
#    self.output_freq = output_freq
#  return

# --- initialization / default values for config file
run = ''
runname = ''
path_data = ''

# --- path to quickplots
path_quickplots = '../../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = False

# --- decide which set of plots to do
do_ocean_plots = True
do_atmosphere_plots = False
do_conf_dwd = False
do_djf = False
do_jja = False
do_hamocc_plots = False

# --- grid information
gname = ''
lev   = ''
gname_atm = ''
lev_atm = ''

# --- path to interpolation files
path_grid        = ''
path_grid_atm    = ''
path_ckdtree     = ''
path_ckdtree_atm = 'auto'

# --- grid files and reference data
fpath_tgrid                 = 'auto'
fpath_tgrid_atm             = 'auto' 
fpath_ref_data_oce          = '' 
fpath_ref_data_atm          = '' # meant for ERA5 if defined
fpath_ref_data_atm_djf      = '' # meant for ERA5 DJF if defined
fpath_ref_data_atm_jja      = '' # meant for ERA5 JJA if defined
fpath_ref_data_atm_rad      = '' # meant for CERES if defined
fpath_ref_data_atm_rad_djf  = '' # meant for CERES DJF if defined
fpath_ref_data_atm_rad_jja  = '' # meant for CERES JJA if defined
fpath_ref_data_atm_prec     = '' # meant for GPM if defined
fpath_ref_data_atm_prec_djf = '' # meant for GPM DJF if defined
fpath_ref_data_atm_prec_jja = '' # meant for GPM JJA if defined
fpath_fx                    = '' 

# --- nc file prefixes
D_variable_container = dict(
  default  = '_P1M_to',
  to       = '_P1M_to',
  so       = '_P1M_so',
  u        = '_P1M_u',
  v        = '_P1M_v',
  massflux = '_P1M_mass_flux',
  moc      = '_P1M_moc',
  mon      = '_P1M_mon',
  ice      = '_P1M_2d',
  monthly  = '_P1M_2d',
  sqr      = '_P1M_sqr',
)

atm_2d      = '' 
atm_3d      = '' 
atm_mon     = '' 

time_mode_atm = 'num2date'  # 'num2date' is the new default previously, 'float2date' was regularly used

# --- time average information (can be overwritten by qp_driver call)
#tave_ints = [['2004-03-01', '2004-05-01']]
tave_ints = [['1979-01-01', '1982-01-01']]
# --- decide which data files to take for time series plots
tstep     = '????????????????'
tstep_ts  = '????????????????'
# --- set this to 12 for yearly averages in timeseries plots, set to 0 for no averaging
ave_freq = 365

time_at_end_of_interval = True

# --- decide if time-series (ts) plots are plotted for all the 
#     available data or only for the intervall defined by tave_int
use_tave_int_for_ts = False

# --- xarray usage
xr_chunks = None
load_xarray_dset = False
save_data = False
path_nc = './'

# --- information for re-gridding
sec_name_30w   = '30W_300pts'
rgrid_name     = 'global_0.3'
#rgrid_name_atm = 'global_1.5_era5'
rgrid_name_atm = 'global_1.0_era'

verbose = False
do_write_final_config = False

# --- list containing figures which should not be plotted
red_list = ['tas_gmean','rsdt_gmean','rsut_gmean','rlut_gmean','radtop_gmean','prec_gmean','evap_gmean','pme_gmean']
# --- list containing figures which should be plotted 
# (if empty all figures will be plotted)
green_list = []

help_text = """
Driver for pyicon quickplots.

Usage notes:
------------

In ipython / Jupyter:
%run qp_driver.py /path/to/config_file.py

From command line:
python qp_driver.py /path/to/config_file.py

In batch mode (slurm):
python qp_driver.py --batch /path/to/config_file.py

For debugging:
%run qp_driver.py /path/to/config_file.py --debug --tave_int=1610-01-01,1620-01-01

Just creating the web page without producing any new figures:
%run qp_driver.py /path/to/config_file.py --no_plots --tave_int=1610-01-01,1620-01-01

Argument list:
--------------
"""

# --- read input arguments
parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)

# --- necessary arguments
parser.add_argument('fpath_config', metavar='fpath_config', type=str,
                    help='path to quickplot configure file')
# --- optional arguments
parser.add_argument('--batch', type=bool, default=False, 
                    help='suppress displaying figures, required for batch mode (same as --slurm)')
parser.add_argument('--slurm', default=False, 
                    action='store_true', #const=False,
                    help='suppress displaying figures, required for batch mode (same as --batch=True)')
parser.add_argument('--no_plots', default=False, 
                    action='store_true', #const=False,
                    help='do not make any plots')
parser.add_argument('--debug', default=False, 
                    action='store_true', #const=False,
                    help='only limitted number of plots are made (specified in qp_driver debugging section)')
parser.add_argument('--path_quickplots', metavar='path_quickplots', type=str, default='none',
                    help='path where the quickplots website and figures are storred')
parser.add_argument('--tave_int', metavar='tave_int', type=str, default='none',
                    help='specify time averaging interval e.g. --tave_int=1600-01-01, 1700-12-31')
parser.add_argument('--run', metavar='run', type=str, default='none',
                    help='name of simulation; if specified, it is used to alter path_data by replacing "run" from config file')
parser.add_argument('--path_data', metavar='path_data', type=str, default='none',
                    help='path of simulation data; if specified, it is used to replace path_data')
iopts = parser.parse_args()

print('--------------------------------------------------------------------------------')
print('Input arguments:')
print('--------------------------------------------------------------------------------')
for var in iopts.__dict__.keys():
  print(var, ' = ', getattr(iopts, var))
print('--------------------------------------------------------------------------------')

fpath_config = iopts.fpath_config

if iopts.batch or iopts.slurm:
  print('using batch mode')

# --- continue loading modules
import matplotlib
if iopts.batch or iopts.slurm:
  matplotlib.use('Agg')
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import json
#sys.path.append('/home/mpim/m300602/pyicon')
import pyicon as pyic
import pyicon.quickplots as pyicqp
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from qp_cmaps import PyicCmaps

#cm_wbgyr = PyicCmaps().WhiteBlueGreenYellowRed

# ------ r2b4
#exec(open("../../config_qp/conf-icon_08-nib002.py").read())
#exec(open("../../config_qp/conf-mac-icon_08-nib0002.py").read())

# ------ r2b6
#exec(open("../../config_qp/conf-icon_08-nib003.py").read())
# does not work so far. diff. grid file
#exec(open("../../config_qp/conf-ocean_omip_long_r2b6_19224-YVF.py").read()) 

# ------ r2b8
#exec(open("../../config_qp/conf-ocean_era51h_r2b8_19074-AMK.py").read())
#exec(open("../../config_qp/conf-exp.ocean_ncep6h_r2b8_20096-FZA.py").read())

# ------ r2b6
#exec(open("../../config_qp/conf-icon_08-nib0004.py").read())
#exec(open("../../config_qp/conf-icon_08-nib0006.py").read())
#exec(open("../../config_qp/conf-jkr0042.py").read())
#exec(open("../../config_qp/conf-slo1284.py").read())

if not os.path.isfile(fpath_config):
  raise ValueError("::: Error: Config file %s does not exist! :::" % (fpath_config))
exec(open(fpath_config).read())

# --- overwrite variables if given by argument parsing
if iopts.path_data!='none':
  path_data = iopts.path_data
if not path_data.endswith('/'):
  path_data += '/'
if iopts.run!='none':
  #path_data = path_data.replace(run, iopts.run)
  run = iopts.run
if iopts.path_quickplots!='none':
  path_quickplots = iopts.path_quickplots
if not iopts.tave_int=='none':
  tave_int = iopts.tave_int.split(',')
  tave_ints = [tave_int]

path_quickplots = os.path.abspath(path_quickplots) + '/'

config_file = f"""
run = \'{run}\'
runname = \'{runname}\'
path_data = \'{path_data}\'

# --- path to quickplots
path_quickplots = \'{path_quickplots}\'

# --- set this to True if the simulation is still running
omit_last_file = {omit_last_file}

# --- decide which set of plots to do
do_ocean_plots      = {do_ocean_plots}
do_atmosphere_plots = {do_atmosphere_plots}
do_conf_dwd         = {do_conf_dwd}
do_djf              = {do_djf}
do_jja              = {do_jja}
do_hamocc_plots     = {do_hamocc_plots}

# --- grid information
gname     = \'{gname}\'
lev       = \'{lev}\'
gname_atm = \'{gname_atm}\'
lev_atm   = \'{lev_atm}\'

# --- path to interpolation files
path_grid        = \'{path_grid}\'
path_grid_atm    = \'{path_grid_atm}\'
path_ckdtree     = \'{path_ckdtree}\'
path_ckdtree_atm = \'{path_ckdtree_atm}\'

# --- grid files and reference data
fpath_tgrid                 = \'{fpath_tgrid}\'
fpath_tgrid_atm             = \'{fpath_tgrid_atm}\'
fpath_ref_data_oce          = \'{fpath_ref_data_oce}\'
fpath_ref_data_atm          = \'{fpath_ref_data_atm}\'
fpath_ref_data_atm_djf      = \'{fpath_ref_data_atm_djf}\'
fpath_ref_data_atm_jja      = \'{fpath_ref_data_atm_jja}\'
fpath_ref_data_atm_rad      = \'{fpath_ref_data_atm_rad}\'
fpath_ref_data_atm_rad_djf  = \'{fpath_ref_data_atm_rad_djf}\'
fpath_ref_data_atm_rad_jja  = \'{fpath_ref_data_atm_rad_jja}\'
fpath_ref_data_atm_prec     = \'{fpath_ref_data_atm_prec}\'
fpath_ref_data_atm_prec_djf = \'{fpath_ref_data_atm_prec_djf}\'
fpath_ref_data_atm_prec_jja = \'{fpath_ref_data_atm_prec_jja}\'
fpath_fx                    = \'{fpath_fx}\'

# --- nc file prefixes
D_variable_container = \'{D_variable_container}\'

atm_2d      = \'{atm_2d}\'
atm_3d      = \'{atm_3d}\'
atm_mon     = \'{atm_mon}\'

time_mode_atm = \'{time_mode_atm}\'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = {tave_ints}
# --- decide which data files to take for time series plots
tstep     = \'{tstep}\'
# --- set this to 12 for yearly averages in timeseries plots, set to 0 for no averaging
ave_freq = {ave_freq}

time_at_end_of_interval = {time_at_end_of_interval}

# --- decide if time-series (ts) plots are plotted for all the 
#     available data or only for the intervall defined by tave_int
use_tave_int_for_ts = {use_tave_int_for_ts}

# --- xarray usage
xr_chunks = {xr_chunks}
load_xarray_dset = {load_xarray_dset}
save_data = {save_data}
path_nc = \'{path_nc}\'

# --- information for re-gridding
sec_name_30w   = \'{sec_name_30w}\'
rgrid_name     = \'{rgrid_name}\'
rgrid_name_atm = \'{rgrid_name_atm}\'

verbose = {verbose}
do_write_final_config = {do_write_final_config}

# --- list containing figures which should not be plotted
red_list = {red_list}
# --- list containing figures which should be plotted 
# (if empty all figures will be plotted)
green_list = {green_list}
"""

fpath_config_full = os.path.abspath(fpath_config[:-3]+'_full.py')
print('--------------------------------------------------------------------------------')
print(f'Configurations for pyicon quickplots:')
if do_write_final_config: 
  print(f'Written to:\n{fpath_config_full}')
print('--------------------------------------------------------------------------------')
print(config_file)
print('--------------------------------------------------------------------------------')
if do_write_final_config: 
  f = open(fpath_config_full, 'w')
  f.write(config_file)
  f.close()

# -------------------------------------------------------------------------------- 
# Settings
# -------------------------------------------------------------------------------- 
projection = 'PlateCarree'

#projection = 'none'

# check if djf / jja plot is required and resolve confilicts
if do_djf and do_jja:
  print('')
  print('do_djf and do_jja can\'t be set to True at the same time.')
  print('Set at least one of them to False!')
  sys.exit()
if do_ocean_plots and (do_djf or do_jja):
  print('')
  print('do_djf & do_jja are not implemented for ocean plots!')
  print('Set either do_ocean_plots=False or do_djf=do_jja=False!')
  sys.exit()
if (do_djf or do_jja) and ave_freq != 12:
  print('')
  print('do_djf & do_jja are implemented only for ave_freq=12 (yearly averaging of monthly data)!')
  sys.exit()

# set reference data based on do_djf/do_jja & give warnings
if do_djf:
  if fpath_ref_data_atm_djf != '' and os.path.isfile(fpath_ref_data_atm_djf):
    fpath_ref_data_atm = fpath_ref_data_atm_djf
  else:
    print('')
    print('do_djf=True is set but fpath_ref_data_atm_djf is not defined or file does not exist!')
  if fpath_ref_data_atm_rad_djf != '' and os.path.isfile(fpath_ref_data_atm_rad_djf):
    fpath_ref_data_atm_rad = fpath_ref_data_atm_rad_djf
  else:
    print('')
    print('do_djf=True is set but fpath_ref_data_atm_rad_djf is not defined or file does not exist!')
  if fpath_ref_data_atm_prec_djf != '' and os.path.isfile(fpath_ref_data_atm_prec_djf):
    fpath_ref_data_atm_prec = fpath_ref_data_atm_prec_djf
  else:
    print('')
    print('do_djf=True is set but fpath_ref_data_atm_prec_djf is not defined or file does not exist!')
elif do_jja:
  if fpath_ref_data_atm_jja != '' and os.path.isfile(fpath_ref_data_atm_jja):
    fpath_ref_data_atm = fpath_ref_data_atm_jja
  else:
    print('')
    print('do_jja=True is set but fpath_ref_data_atm_jja is not defined or file does not exist!')
  if fpath_ref_data_atm_rad_jja != '' and os.path.isfile(fpath_ref_data_atm_rad_jja):
    fpath_ref_data_atm_rad = fpath_ref_data_atm_rad_jja
  else:
    print('')
    print('do_jja=True is set but fpath_ref_data_atm_rad_jja is not defined or file does not exist!')
  if fpath_ref_data_atm_prec_jja != '' and os.path.isfile(fpath_ref_data_atm_prec_jja):
    fpath_ref_data_atm_prec = fpath_ref_data_atm_prec_jja
  else:
    print('')
    print('do_jja=True is set but fpath_ref_data_atm_prec_jja is not defined or file does not exist!')

# is ERA5 defined?
exist_era5 = fpath_ref_data_atm != '' and os.path.isfile(fpath_ref_data_atm)
# is CERES defined?
exist_ceres = fpath_ref_data_atm_rad != '' and os.path.isfile(fpath_ref_data_atm_rad)
# is GPM defined?
exist_gpm = fpath_ref_data_atm_prec != '' and os.path.isfile(fpath_ref_data_atm_prec)
# is any reference defined?
exist_ref = exist_era5 or exist_ceres or exist_gpm

# check if CERES & GPM reference data exist and fall back to ERA5 if not
if not exist_ceres:
  if exist_era5:
    fpath_ref_data_atm_rad = fpath_ref_data_atm
    print('')
    print('CERES reference data not defined or file does not exist --> ERA5 is used instead!')
if not exist_gpm:
  if exist_era5:
    fpath_ref_data_atm_prec = fpath_ref_data_atm
    print('')
    print('GPM reference data not defined or file does not exist --> ERA5 is used instead!')

print('')
print('Reference data used:')
if exist_ref:
  print('in general:')
  print(fpath_ref_data_atm)
  print('radiation & total cloud cover:')
  print(fpath_ref_data_atm_rad)
  print('precipitation:')
  print(fpath_ref_data_atm_prec)
else:
  print('None.')
  print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
  print('No bias plots will be generated!')
  print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('')

# --- structure of web page
fig_names = []
fig_names += ['sec:Overview']
fig_names += ['tab_overview']
if do_ocean_plots:
  fig_names += ['sec:Upper ocean']
  fig_names += ['ssh', 'ssh_variance', 'sst', 'sss', 'mlotst_mar', 'mlotst_sep'] 
  fig_names += ['sec:Ice']
  fig_names += ['ice_concentration_nh', 'ice_thickness_nh', 'snow_thickness_nh',] 
  fig_names += ['ice_concentration_sh', 'ice_thickness_sh', 'snow_thickness_sh',]
  fig_names += ['sec:Sections']
  fig_names += ['temp30w', 'salt30w', 'dens30w']
  fig_names += ['sec:Zonal averages']
  fig_names += ['temp_gzave', 'temp_azave', 'temp_ipzave']
  fig_names += ['salt_gzave', 'salt_azave', 'salt_ipzave']
  fig_names += ['sec:Transports']
  fig_names += ['bstr', 'passage_transports', 'tab_passage_transports']
  fig_names += ['arctic_budgets']
  fig_names += ['amoc', 'pmoc', 'gmoc']
  fig_names += ['ke_100m', 'ke_2000m']
  fig_names += ['heat_flux', 'heat_hfbasin', 'freshwater_flux']
  fig_names += ['sec:Biases']
  fig_names += ['sst_bias', 'temp_bias_gzave', 'temp_bias_azave', 'temp_bias_ipzave']
  fig_names += ['sss_bias', 'salt_bias_gzave', 'salt_bias_azave', 'salt_bias_ipzave']
  fig_names += ['sec:Time series']
  fig_names += ['ts_amoc', 'ts_heat_content', 'ts_ssh', 'ts_sst', 'ts_sss', 'ts_hfl', 'ts_wfl', 'ts_ice_volume_nh', 'ts_ice_volume_sh', 'ts_ice_extent_nh', 'ts_ice_extent_sh',]
if do_atmosphere_plots:
  fig_names += ['sec:Surface fluxes']
  fig_names += ['atm_tauu', 'atm_tauu_bias', 'atm_tauv', 'atm_tauv_bias']
  fig_names += ['atm_curl_tau', 'atm_wek']
  fig_names += ['atm_surf_shfl', 'atm_surf_shfl_bias', 'atm_surf_lhfl', 'atm_surf_lhfl_bias']
  fig_names += ['sec:TOA fluxes']
  fig_names += ['atm_toa_netrad', 'atm_toa_netrad_bias', 'atm_toa_sob', 'atm_toa_sob_bias']
  #fig_names += ['atm_toa_sod', 'atm_toa_sod_bias']
  fig_names += ['atm_toa_sou', 'atm_toa_sou_bias']
  fig_names += ['atm_toa_thb', 'atm_toa_thb_bias']
  fig_names += ['sec:Atmosphere surface']
  fig_names += ['sea_ts', 'seaice_fraction'] # we have these to check what's in the forcing in uncoupled runs
  fig_names += ['atm_psl', 'atm_psl_bias', 'atm_w10m', 'atm_w10m_bias']
  fig_names += ['atm_ts', 'atm_ts_bias', 'atm_t2m', 'atm_t2m_bias']
  fig_names += ['atm_cwv', 'atm_cwv_bias', 'atm_tcc', 'atm_tcc_bias']
  fig_names += ['atm_tp', 'atm_tp_bias', 'atm_pme', 'atm_pme_bias']
  fig_names += ['sec:Atmosphere pressure levels']
  fig_names += ['atm_u_250', 'atm_u_250_bias', 'atm_v_250', 'atm_v_250_bias', 'atm_geop_500', 'atm_geop_500_bias']
  fig_names += ['atm_relhum_700', 'atm_relhum_700_bias', 'atm_temp_850', 'atm_temp_850_bias']
  fig_names += ['sec:Atmosphere zonal averages']
  fig_names += ['atm_temp_zave', 'atm_temp_zave_bias', 'atm_logv_temp_zave']
  fig_names += ['atm_u_zave', 'atm_u_zave_bias', 'atm_logv_u_zave']
  fig_names += ['atm_v_zave', 'atm_v_zave_bias', 'atm_logv_v_zave']
#  fig_names += ['atm_spechum_zave', 'atm_spechum_zave_bias']
  fig_names += ['atm_relhum_zave', 'atm_relhum_zave_bias']
  fig_names += ['atm_cc_zave', 'atm_cc_zave_bias']
  fig_names += ['atm_clw_zave', 'atm_clw_zave_bias']
  fig_names += ['atm_cli_zave', 'atm_cli_zave_bias']
  fig_names += ['atm_clwi_zave', 'atm_psi']
  fig_names += ['sec:Time series']
  fig_names += ['ts_t2m_gmean', 'ts_radtop_gmean']
  fig_names += ['ts_rsdt_gmean', 'ts_rsut_gmean', 'ts_rlut_gmean', 'ts_prec_gmean', 'ts_evap_gmean', 'ts_pme_gmean', 'ts_fwfoce_gmean']
  # --- variable names
  if not do_conf_dwd:
     vpfull   = 'pfull'
     vtas     = 'tas'
     vts      = 'ts'
     vprw     = 'prw'
     vpsl     = 'psl'
     vzg      = 'zg'
     vta      = 'ta'
     vtauu    = 'tauu'
     vtauv    = 'tauv'
     vcllvi   = 'cllvi'
     vclivi   = 'clivi'
     vpr      = 'pr'
     vclt     = 'clt'
     vevspsbl = 'evspsbl'
     vsfcwind = 'sfcwind'
     vua      = 'ua'
     vva      = 'va'
     vhus     = 'hus'
     vhur     = 'hur'
     vcl      = 'cl'
     vclw     = 'clw'
     vcli     = 'cli'
  else:
     vpfull   = 'pres'
     vtas     = 't_2m'
     vts      = 't_s'
     vsst     = 't_seasfc'
     vfrsi    = 'fr_seaice'
     vprw     = 'tqv_dia'
     vpsl     = 'pres_msl'
     vzg      = 'geopot'
     vta      = 'temp'
     vtauu    = 'umfl_s'
     vtauv    = 'vmfl_s'
     vcllvi   = 'tqc_dia'
     vclivi   = 'tqi_dia'
     vpr      = 'tot_prec_rate'
     vclt     = 'clct'
     vshfl_s  = 'shfl_s'
     vlhfl_s  = 'lhfl_s'
     vsob_t   = 'sob_t'
     vsou_t   = 'sou_t'
     vthb_t   = 'thb_t'
     vevspsbl = 'qhfl_s'
     vsfcwind = 'sp_10m'
     vua      = 'u'
     vva      = 'v'
     vhus     = 'qv'
     vhur     = 'rh'
     vcl      = 'clc'
     vclw     = 'tot_qc_dia'
     vcli     = 'tot_qi_dia'
if do_hamocc_plots:
  fig_names += ['sec:Hamocc time series']
  fig_names += ['ts_global_npp', 'ts_global_nppcya', 'ts_global_zoograzing', 'ts_global_netco2flux']
  fig_names += ['ts_global_surface_alk', 'ts_global_surface_dic', 'ts_global_surface_sil', 'ts_global_surface_phos']
  fig_names += ['ts_WC_denit', 'ts_sed_denit', 'ts_n2fix', 'ts_global_opal_prod', 'ts_global_caco3_prod']
  fig_names += ['ts_global_OMexp90', 'ts_global_calcexp90', 'ts_global_opalexp90']
  fig_names += ['sec:Hamocc surface maps']
  fig_names += ['srf_phyp', 'srf_zoop', 'srf_cya', 'srf_silicate', 'srf_nitrate', 'srf_phosphate']
  fig_names += ['srf_alk', 'srf_dic', 'srf_pH', 'srf_co2flux'] # srf_hion
  fig_names += ['sec:Hamocc sections']
  fig_names += ['dic_gzave', 'dic_azave', 'dic_ipzave', 'o2_gzave', 'o2_azave', 'o2_ipzave']

plist = fig_names
fig_names = []
for pitem in plist:
  if not pitem.startswith('sec:') and not pitem in red_list:
    fig_names += [pitem]

# --- use green list if it hast entries
if not len(green_list)==0:
  fig_names = green_list

# --- for debugging
if iopts.debug:
  print('XXXXXXXXXXXXXXXXX Debugging mode! XXXXXXXXXXXXXXX')
  fig_names = []
  fig_names += ['arctic_budgets']

fig_names = np.array(fig_names)

# -------------------------------------------------------------------------------- 
# Function to save figures
# -------------------------------------------------------------------------------- 
close_figs = True
def save_fig(title, path_pics, fig_name, FigInf=dict()):
  FigInf['name']  = fig_name+'.png'
  FigInf['fpath'] = path_pics+FigInf['name']
  FigInf['title'] = title
  print('Saving {:<40} {:<40}'.format(FigInf['fpath'], FigInf['title']))
  plt.savefig(FigInf['fpath'], dpi=300)
  with open(path_pics+fig_name+'.json', 'w') as fj:
    json.dump(FigInf, fj, sort_keys=True, indent=4)
  if close_figs:
    plt.close('all')
  return
plt.close('all')

def save_tab(text, title, path_pics, fig_name, FigInf=dict()):
  FigInf['name']  = fig_name+'.html'
  FigInf['fpath'] = path_pics+FigInf['name']
  FigInf['title'] = title
  print('Saving {:<40} {:<40}'.format(FigInf['fpath'], FigInf['title']))
  with open(FigInf['fpath'], 'w') as f:
    f.write(text)
  with open(path_pics+fig_name+'.json', 'w') as fj:
    json.dump(FigInf, fj, sort_keys=True, indent=4)
  return

def indfind(array, vals):
  vals = np.array(vals)
  inds = np.zeros(vals.size, dtype=int)
  for nn, val in enumerate(vals):
    inds[nn] = np.argmin(np.abs(array-val))
  return inds

def load_era3d_var(fpath, var, isort):
  f = Dataset(fpath, 'r')
  data = f.variables[var][:].mean(axis=0)
  f.close()
  data = data[:,:,isort]
  return data

if load_xarray_dset and not xr_chunks is None:
  print('Setting up LocalCluster:')
  from dask.distributed import Client, LocalCluster
  cluster = LocalCluster(interface='ib0')
  client = Client(cluster)
  print(client)

# -------------------------------------------------------------------------------- 
# Load all necessary data sets (can take a while)
# -------------------------------------------------------------------------------- 
if do_ocean_plots and not iopts.no_plots:

  IDsettings = dict(
    path_data    = path_data,
    path_grid    = path_grid,
    path_ckdtree = path_ckdtree,
    fpath_fx     = fpath_fx,
    fpath_tgrid  = fpath_tgrid,
    gname        = gname,
    lev          = lev,
    rgrid_name   = rgrid_name,
    do_triangulation       = False,
    omit_last_file         = omit_last_file,
    load_vertical_grid     = False,
    load_triangular_grid   = False, # needed for bstr
    load_rectangular_grid  = True,
    calc_coeff             = False,
    load_xarray_dset       = load_xarray_dset,
    xr_chunks              = xr_chunks,
    time_at_end_of_interval= time_at_end_of_interval,
    verbose                = verbose,
  )

  # --- start with IconData objects
  fname = '%s%s_%s.nc' % (run, D_variable_container['default'], tstep)
  print('Dataset %s' % (fname))
  IcD = pyic.IconData(
                 fname        = fname,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 path_ckdtree = path_ckdtree,
                 fpath_fx     = fpath_fx,
                 fpath_tgrid  = fpath_tgrid,
                 gname        = gname,
                 lev          = lev,
                 rgrid_name   = rgrid_name,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = True,
                 load_triangular_grid   = True, # needed for bstr
                 load_rectangular_grid  = True,
                 calc_coeff             = False,
                 load_xarray_dset       = load_xarray_dset,
                 xr_chunks              = xr_chunks,
                 time_at_end_of_interval= time_at_end_of_interval,
                 verbose                = verbose, 
                )
  fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
  [k100, k500, k800, k1000, k2000, k3000] = indfind(IcD.depthc, [100., 500., 800., 1000., 2000., 3000.])

  DIcD = dict()
  for var in ['to', 'so', 'u', 'v', 'massflux', 'ice', 'monthly', 'mon', 'sqr']:
    if var=='mon':
      tstep_tmp = tstep_ts
    else:
      tstep_tmp = tstep
    fname = '%s%s_%s.nc' % (run, D_variable_container[var], tstep_tmp)
    print(f'Data container for {var}: {fname}')
    DIcD[var] = pyic.IconData(
      fname=fname, **IDsettings)
  DIcD['massflux'].load_tgrid()
  DIcD['massflux'].dzw = IcD.dzw # needed for arctic budgets
  DIcD['massflux'].wet_c = IcD.wet_c # needed for arctic budgets
  DIcD['massflux'].wet_e = IcD.wet_e # needed for arctic budgets
  
  fname_moc = '%s%s_%s.nc' % (run, D_variable_container['moc'], tstep)
  print('Dataset %s' % (fname_moc))
  IcD_moc = pyic.IconData(
                 fname        = fname_moc,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 path_ckdtree = path_ckdtree,
                 fpath_tgrid  = fpath_tgrid,
                 gname        = gname,
                 lev          = lev,
                 rgrid_name   = rgrid_name,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = True,
                 calc_coeff             = False,
                 time_at_end_of_interval= time_at_end_of_interval,
                 verbose                = verbose,
                )
  IcD_moc.depthc = IcD.depthc
  IcD_moc.depthi = IcD.depthi

  #fname_monthly = '%s%s_%s.nc' % (run, D_variable_container['monthly'], tstep)
  #print('Dataset %s' % (fname_monthly))
  #IcD_monthly = pyic.IconData(
  #               fname        = fname_monthly,
  #               path_data    = path_data,
  #               path_grid    = path_grid,
  #               path_ckdtree = path_ckdtree,
  #               fpath_tgrid  = fpath_tgrid,
  #               gname        = gname,
  #               lev          = lev,
  #               rgrid_name   = rgrid_name,
  #               do_triangulation       = False,
  #               omit_last_file         = omit_last_file,
  #               load_vertical_grid     = False,
  #               load_triangular_grid   = False,
  #               load_rectangular_grid  = True,
  #               calc_coeff             = False,
  #               verbose                = verbose,
  #              )
  IcD_mon = DIcD['mon']
  IcD_monthly = DIcD['monthly']
  IcD_monthly.wet_c = IcD.wet_c
  #IcD_monthly.lon = IcD.lon
  #IcD_monthly.lat = IcD.lat

  #fname_ice = '%s%s_%s.nc' % (run, D_variable_container['ice'], tstep)
  #print('Dataset %s' % (fname_ice))
  #IcD_ice = pyic.IconData(
  #               fname        = fname_ice,
  #               path_data    = path_data,
  #               path_grid    = path_grid,
  #               path_ckdtree = path_ckdtree,
  #               fpath_tgrid  = fpath_tgrid,
  #               gname        = gname,
  #               lev          = lev,
  #               do_triangulation       = False,
  #               omit_last_file         = omit_last_file,
  #               load_vertical_grid     = False,
  #               load_triangular_grid   = False,
  #               load_rectangular_grid  = False,
  #               calc_coeff             = False,
  #               verbose                = verbose,
  #              )

  #fname_mon = '%s%s_%s.nc' % (run, D_variable_container['mon'], tstep)
  #print('Dataset %s' % (fname_mon))
  #IcD_mon = pyic.IconData(
  #               fname        = fname_mon,
  #               path_data    = path_data,
  #               path_grid    = path_grid,
  #               path_ckdtree = path_ckdtree,
  #               fpath_tgrid  = fpath_tgrid,
  #               gname        = gname,
  #               lev          = lev,
  #               do_triangulation       = False,
  #               omit_last_file         = omit_last_file,
  #               load_vertical_grid     = False,
  #               load_triangular_grid   = False,
  #               load_rectangular_grid  = False,
  #               calc_coeff             = False,
  #               verbose                = verbose,
  #              )

if do_atmosphere_plots and not iopts.no_plots:
  fname_atm_2d = '%s%s_%s.nc' % (run, atm_2d, tstep)
  print('Dataset %s' % (fname_atm_2d))
  IcD_atm2d = pyic.IconData(
                 fname        = fname_atm_2d,
                 path_data    = path_data,
                 path_grid    = path_grid_atm,
                 path_ckdtree = path_ckdtree_atm,
                 fpath_tgrid  = fpath_tgrid_atm,
                 gname        = gname_atm,
                 lev          = lev_atm,
                 rgrid_name   = rgrid_name_atm,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = True,
                 load_rectangular_grid  = True,
                 calc_coeff             = True,
                 verbose                = verbose,
                 time_mode    = time_mode_atm,
                 model_type   = 'atm',
                 do_conf_dwd   = do_conf_dwd,
                 time_at_end_of_interval= False,
                )

  fname_atm_3d = '%s%s_%s.nc' % (run, atm_3d, tstep)
  print('Dataset %s' % (fname_atm_3d))
  IcD_atm3d = pyic.IconData(
                 fname        = fname_atm_3d,
                 path_data    = path_data,
                 path_grid    = path_grid_atm,
                 path_ckdtree = path_ckdtree_atm,
                 fpath_tgrid  = fpath_tgrid_atm,
                 gname        = gname_atm,
                 lev          = lev_atm,
                 rgrid_name   = rgrid_name_atm,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = True,
                 calc_coeff             = False,
                 verbose                = verbose,
                 time_mode    = time_mode_atm,
                 model_type   = 'atm',
                 do_conf_dwd   = do_conf_dwd,
                 time_at_end_of_interval= False,
                )
  fpath_ckdtree_atm = IcD_atm3d.rgrid_fpath_dict[rgrid_name_atm]

  fname_atm_mon = '%s%s_%s.nc' % (run, atm_mon, tstep_ts)
  print('Dataset %s' % (fname_atm_mon))
  IcD_atm_mon = pyic.IconData(
                 fname        = fname_atm_mon,
                 path_data    = path_data,
                 path_grid    = path_grid_atm,
                 path_ckdtree = path_ckdtree_atm,
                 fpath_tgrid  = fpath_tgrid_atm,
                 gname        = gname_atm,
                 lev          = lev_atm,
                 rgrid_name   = rgrid_name_atm,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = False,
                 calc_coeff             = False,
                 verbose                = verbose,
                 time_mode    = time_mode_atm,
                 model_type   = 'atm',
                 do_conf_dwd   = do_conf_dwd,
                 time_at_end_of_interval= False,
                )
  
  if do_ocean_plots==False:
    IcD_monthly = IcD_atm_mon
    IcD = IcD_atm3d

if do_hamocc_plots and not iopts.no_plots:
  if not do_ocean_plots:
    fname = '%s%s_%s.nc' % (run, D_variable_container['default'], tstep)
    print('Dataset %s' % (fname))
    IcD = pyic.IconData(
                   fname        = fname,
                   path_data    = path_data,
                   path_grid    = path_grid,
                   path_ckdtree = path_ckdtree,
                   fpath_fx     = fpath_fx,
                   fpath_tgrid  = fpath_tgrid,
                   gname        = gname,
                   lev          = lev,
                   rgrid_name   = rgrid_name,
                   do_triangulation       = False,
                   omit_last_file         = omit_last_file,
                   load_vertical_grid     = True,
                   load_triangular_grid   = True, # needed for bstr
                   load_rectangular_grid  = True,
                   calc_coeff             = False,
                   time_at_end_of_interval= time_at_end_of_interval,
                   verbose                = verbose,
                  )
    fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
    [k100, k500, k800, k1000, k2000, k3000] = indfind(IcD.depthc, [100., 500., 800., 1000., 2000., 3000.])

    fname_monthly = '%s%s_%s.nc' % (run, D_variable_container['monthly'], tstep)
    print('Dataset %s' % (fname_monthly))
    IcD_monthly = pyic.IconData(
                   fname        = fname_monthly,
                   path_data    = path_data,
                   path_grid    = path_grid,
                   path_ckdtree = path_ckdtree,
                   fpath_tgrid  = fpath_tgrid,
                   gname        = gname,
                   lev          = lev,
                   rgrid_name   = rgrid_name,
                   do_triangulation       = False,
                   omit_last_file         = omit_last_file,
                   load_vertical_grid     = False,
                   load_triangular_grid   = False,
                   load_rectangular_grid  = True,
                   calc_coeff             = False,
                   time_at_end_of_interval= time_at_end_of_interval,
                   verbose                = verbose,
                  )
    IcD_monthly.wet_c = IcD.wet_c

  fname_ham_inv = '%s%s_%s.nc' % (run, ham_inv, tstep)
  print('Dataset %s' % (fname_ham_inv))
  IcD_ham_inv = pyic.IconData(
                 fname        = fname_ham_inv,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 path_ckdtree = path_ckdtree,
                 fpath_fx     = fpath_fx,
                 fpath_tgrid  = fpath_tgrid,
                 gname        = gname,
                 lev          = lev,
                 rgrid_name   = rgrid_name,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = True,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = True,
                 calc_coeff             = False,
                 time_at_end_of_interval= time_at_end_of_interval,
                 verbose                = verbose,
                )
  IcD_ham_inv.wet_c = IcD.wet_c

  fname_ham_2d = '%s%s_%s.nc' % (run, ham_2d, tstep)
  print('Dataset %s' % (fname_ham_2d))
  IcD_ham_2d = pyic.IconData(
                 fname        = fname_ham_2d,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 path_ckdtree = path_ckdtree,
                 fpath_tgrid  = fpath_tgrid,
                 gname        = gname,
                 lev          = lev,
                 rgrid_name   = rgrid_name,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = True,
                 calc_coeff             = False,
                 time_at_end_of_interval= time_at_end_of_interval,
                 verbose                = verbose,
                )
  IcD_ham_2d.wet_c = IcD.wet_c

  fname_ham_mon = '%s%s_%s.nc' % (run, ham_mon, tstep_ts)
  print('Dataset %s' % (fname_ham_mon))
  IcD_ham_mon = pyic.IconData(
                 fname        = fname_ham_mon,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 path_ckdtree = path_ckdtree,
                 fpath_tgrid  = fpath_tgrid,
                 gname        = gname,
                 lev          = lev,
                 do_triangulation       = False,
                 omit_last_file         = omit_last_file,
                 load_vertical_grid     = False,
                 load_triangular_grid   = False,
                 load_rectangular_grid  = False,
                 calc_coeff             = False,
                 time_at_end_of_interval= time_at_end_of_interval,
                 verbose                = verbose,
                )
print('Done reading datasets')


print(f'--------------------------------------------------------------------------------')
if do_ocean_plots and not iopts.no_plots:
  print(f'--- ocean:')
  print(f'fpath_fx          = {IcD.fpath_fx}')
  print(f'fpath_tgrid       = {IcD.fpath_tgrid}')
  print(f'path_data         = {path_data}')
  print(f'gname             = {gname}')
  print(f'lev               = {lev}')
  print(f'rgrid_name        = {rgrid_name}')
if do_atmosphere_plots and not iopts.no_plots:
  print(f'--- atmosphere:')
  #print(f'fpath_fx_atm      = {IcD_atm3d.fpath_fx}')
  print(f'fpath_tgrid_atm   = {IcD_atm3d.fpath_tgrid}')
  print(f'path_data         = {path_data}')
  print(f'gname_atm         = {gname_atm}')
  print(f'lev_atm           = {lev_atm}')
  print(f'rgrid_name_atm    = {rgrid_name_atm}')
if do_hamocc_plots and not iopts.no_plots:
  print(f'--- hamocc:')
  print(f'fpath_fx          = {IcD.fpath_fx}')
  print(f'fpath_tgrid       = {IcD.fpath_tgrid}')
  print(f'path_data         = {path_data}')
  print(f'gname             = {gname}')
  print(f'lev               = {lev}')
  print(f'rgrid_name        = {rgrid_name}')
print(f'--------------------------------------------------------------------------------')

# -------------------------------------------------------------------------------- 
# timing
# -------------------------------------------------------------------------------- 

for tave_int in tave_ints:
  t1 = tave_int[0].replace(' ', '')
  t2 = tave_int[1].replace(' ', '')

  if isinstance(t1,str) and t1=='auto':
    t1 = IcD.times[0]
  else:
    t1 = np.datetime64(t1)
  if isinstance(t2,str) and t2=='auto':
    t2 = IcD.times[-1]
  else:
    t2 = np.datetime64(t2)

  # -------------------------------------------------------------------------------- 
  # making new directories and copying files
  # -------------------------------------------------------------------------------- 
  # --- path to module
  path_qp_driver = os.path.dirname(pyicqp.__file__)+'/'

  # --- make directory for this current time average
  if runname=='':
    if do_djf:
      dirname = f'qp-{run}-djf'
    elif do_jja:
      dirname = f'qp-{run}-jja'
    else:
      dirname = f'qp-{run}'
  else:
    if do_djf:
      dirname = f'qp-{runname}-{run}-djf'
    elif do_jja:
      dirname = f'qp-{runname}-{run}-jja'
    else:
      dirname = f'qp-{runname}-{run}'
  path_qp_sim = f'{path_quickplots}/{dirname}/'
  path_qp_tav = f'{path_quickplots}/{dirname}/tave_{t1}-{t2}/'
  if not os.path.exists(path_qp_tav):
    os.makedirs(path_qp_tav)
  
  # --- make directory for pictures
  rpath_pics = 'pics/'
  path_pics = path_qp_tav+rpath_pics
  if not os.path.exists(path_pics):
    os.makedirs(path_pics)
  
  # --- copy css style file
  shutil.copyfile(path_qp_driver+'qp_css.css', path_qp_tav+'qp_css.css')
  
  # --- backup qp_driver (this script)
  fname_this_script = __file__.split('/')[-1]
  shutil.copyfile(path_qp_driver+fname_this_script, path_qp_tav+'bcp_'+fname_this_script)

  # --- backup config file
  fname_config = fpath_config.split('/')[-1]
  shutil.copyfile(fpath_config, path_qp_tav+'bcp_'+fname_config)
  
  # --- initialize web page
  if runname=='':
    if do_djf:
      title_str = run+'-djf'
    elif do_jja:
      title_str = run+'-jja'
    else:
      title_str = run
  else:
    if do_djf:
      title_str = '%s | %s' % (runname, run+'-djf')
    elif do_jja:
      title_str = '%s | %s' % (runname, run+'-jja')
    else:
      title_str = '%s | %s' % (runname, run)
  qp = pyicqp.QuickPlotWebsite(
    title=title_str, 
    author=os.environ.get('USER'),
    date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    path_data=path_data,
    info=f'time average from {t1} to {t2}',
    fpath_css='./qp_css.css',
    fpath_html=path_qp_tav+'qp_index.html'
    )
    
  if not iopts.no_plots:
    # -------------------------------------------------------------------------------- 
    # specify time averaging indices
    # -------------------------------------------------------------------------------- 
    # Ocean
    mask_int = (IcD_monthly.times>=t1) & (IcD_monthly.times<=t2)
    months = IcD_monthly.times.astype('datetime64[M]').astype(int) % 12 + 1
    it_ave_months = np.where( mask_int )[0]
    # Note: In ICON the date of an average is set to the end of the averaging interval
    #       Thus, the 4th month corresponds to March and the 10th to September
    if IcD_monthly.time_at_end_of_interval:
      it_ave_mar = np.where( mask_int & (months==4)  )[0]
      it_ave_sep = np.where( mask_int & (months==10) )[0]
    else:
      it_ave_mar = np.where( mask_int & (months==3) )[0]
      it_ave_sep = np.where( mask_int & (months==9) )[0]
    if it_ave_mar.size==0:
      it_ave_mar = it_ave_months
    if it_ave_sep.size==0:
      it_ave_sep = it_ave_months
    #it_ave_mar = it_ave[2::12] # this only workes if tave_int start with Feb. E.g.: 1610-02-01,1620-01-01
    #it_ave_sep = it_ave[8::12] # this only workes if tave_int start with Feb. E.g.: 1610-02-01,1620-01-01
    it_ave_years = (IcD.times>=t1) & (IcD.times<=t2)
    # Atmosphere
    mask_int_atm = (IcD_atm_mon.times>=t1) & (IcD_atm_mon.times<=t2)
    months = IcD_atm_mon.times.astype('datetime64[M]').astype(int) % 12 + 1
    it_ave_atm= np.where( mask_int_atm )[0]
    if IcD_atm_mon.time_at_end_of_interval:
      it_ave_djf = np.where( mask_int_atm & ((months==1) | (months==2) | (months==3))  )[0]
      it_ave_jja = np.where( mask_int_atm & ((months==7) | (months==8) | (months==9))  )[0]
    else:
      it_ave_djf = np.where( mask_int_atm & ((months==12) | (months==1) | (months==2))  )[0]
      it_ave_jja = np.where( mask_int_atm & ((months==6 ) | (months==7) | (months==8))  )[0]
    if do_djf:
      it_ave_atm = it_ave_djf
    elif do_jja:
      it_ave_atm = it_ave_jja
    print('tpoints for year average from yearly data:  ', IcD.times[it_ave_years])
    print('tpoints for year average from monthly data: ', IcD_monthly.times[it_ave_months])
    print('tpoints for Mar average:                    ', IcD_monthly.times[it_ave_mar])
    print('tpoints for Sep avarage:                    ', IcD_monthly.times[it_ave_sep])
    print('tpoints for DJF average:                    ', IcD_monthly.times[it_ave_djf])
    print('tpoints for JJA avarage:                    ', IcD_monthly.times[it_ave_jja])
    print('tpoints used for atmo plots:                ', IcD_monthly.times[it_ave_atm])

    if mask_int.sum()==0 or mask_int_atm.sum()==0:
      raise ValueError(f'::: Error: Cannot find any data in {path_data} for time period from {t1} unitl {t2}! :::')

    # ================================================================================ 
    # start with plotting
    # ================================================================================ 

    # -------------------------------------------------------------------------------- 
    # upper ocean
    # -------------------------------------------------------------------------------- 
    #fname = '%s%s_%s.nc' % (run, oce_def, tstep)
    fname = '%s%s_%s.nc' % (run, D_variable_container['monthly'], tstep)
    Ddict_global = dict(
      xlim=[-180.,180.], ylim=[-90.,90.],
      rgrid_name=rgrid_name,
      path_ckdtree=path_ckdtree,
      projection=projection,
                )
    
    # ---
    fig_name = 'mlotst_mar'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='mlotst', depth=0,
                               it_ave=it_ave_mar,
                               title='mixed layer depth March [m]',
                               #clim=[0,5000.], cincr=250., cmap=PyicCmaps().WhiteBlueGreenYellowRed,
                               clim=[0,5000.], cincr=250., cmap='RdYlBu_r',
                               do_mask=True,
                               IcD=IcD_monthly,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Mixed layer depth March', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'mlotst_sep'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='mlotst', depth=0,
                               it_ave=it_ave_sep,
                               title='mixed layer depth September [m]',
                               clim=[0,5000.], cincr=250., cmap='RdYlBu_r',
                               do_mask=True,
                               IcD=IcD_monthly,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Mixed layer depth September', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'sst'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='to', depth=0, it=0,
                               t1=t1, t2=t2,
                               clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
                               IcD=DIcD['to'],
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('SST', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'sss'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='so', depth=0, it=0,
                               t1=t1, t2=t2,
                               #clim=[32.,37], cincr=0.25, cmap='cmo.haline',
                               clim=[25.,40.], clevs=[25, 28, 30, 32, 32.5, 33, 33.5, 34, 34.5, 35, 35.5, 36, 37, 38, 40], cmap='cmo.haline',
                               IcD=DIcD['so'],
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('SSS', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'ssh'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='zos', depth=0, it=0,
                               t1=t1, t2=t2,
                               clim=2, cincr=0.2, cmap='RdBu_r',
                               IcD=DIcD['monthly'],
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('SSH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'ssh_variance'
    if fig_name in fig_names:
      #zos, it_ave          = pyic.time_average(IcD, 'zos',        t1=t1, t2=t2)
      #zos_square, it_ave   = pyic.time_average(IcD, 'zos_square', t1=t1, t2=t2)
      zos, it_ave          = pyic.time_average(DIcD['monthly'], 'zos',        t1=t1, t2=t2)
      zos_square, it_ave   = pyic.time_average(DIcD['monthly'], 'zos', t1=t1, t2=t2)
      zos_var = np.sqrt(zos_square-zos**2)
      IaV = pyic.IconVariable('zos_var', 'm', 'ssh variance')
      IaV.data = zos_var
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD, IaV, clim=[-2,0], cincr=0.2, cmap='RdYlBu_r',
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      logplot=True,
                      title='log$_{10}$(ssh variance) [m]', do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )
      save_fig('SSH variance', path_pics, fig_name)
    
    # ---
    fig_name = 'ice_concentration_nh'
    if fig_name in fig_names:

      #izice = 'all' 
      izice = 0
      conc_mar, it_ave = pyic.time_average(IcD_monthly, 'conc', it_ave=it_ave_mar, iz=izice)
      hi_mar, it_ave   = pyic.time_average(IcD_monthly, 'hi', it_ave=it_ave_mar, iz=izice)
      hs_mar, it_ave   = pyic.time_average(IcD_monthly, 'hs', it_ave=it_ave_mar, iz=izice)
      hiconc_mar = (hi_mar*conc_mar)
      hsconc_mar = (hs_mar*conc_mar)

      conc_sep, it_ave = pyic.time_average(IcD_monthly, 'conc', it_ave=it_ave_sep, iz=izice)
      hi_sep, it_ave   = pyic.time_average(IcD_monthly, 'hi', it_ave=it_ave_sep, iz=izice)
      hs_sep, it_ave   = pyic.time_average(IcD_monthly, 'hs', it_ave=it_ave_sep, iz=izice)
      hiconc_sep = (hi_sep*conc_sep)
      hsconc_sep = (hs_sep*conc_sep)

    # ---
    fig_name = 'ice_concentration_nh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.NorthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('conc_mar', '', 'sea ice concentration March')
      IaV.data = conc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, cmap='RdYlBu_r',
                      clim=[0,1], clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.,
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('conc_sep', 'm', 'sea ice concentration September')
      IaV.data = conc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, cmap='RdYlBu_r',
                      clim=[0,1], clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.,
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sea ice concentration NH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'ice_thickness_nh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.NorthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hiconc_mar', 'm', 'ice equiv. thickness March')
      IaV.data = hiconc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, 
                      clim=[0,6], clevs=[0, 0.01, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hiconc_sep', 'm', 'ice equiv. thickness September')
      IaV.data = hiconc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, 
                      clim=[0,6], clevs=[0, 0.01, 0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sea ice equiv. thickness NH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'snow_thickness_nh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.NorthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hsconc_mar', 'm', 'snow equiv. thickness March')
      IaV.data = hsconc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hsconc_sep', 'm', 'snow equiv. thickness September')
      IaV.data = hsconc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[60.,90.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Snow equiv. thickness NH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'ice_concentration_sh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.SouthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('conc_mar', '', 'sea ice concentration March')
      IaV.data = conc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, cmap='RdYlBu_r',
                      clim=[0,1], clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.,
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('conc_sep', 'm', 'sea ice concentration September')
      IaV.data = conc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, cmap='RdYlBu_r',
                      clim=[0,1], clevs=np.array([0,1.,5,10,15,20,30,40,50,60,70,80,85,90,95,99,100])/100.,
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sea ice concentration SH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'ice_thickness_sh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.SouthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hiconc_mar', 'm', 'ice equiv. thickness March')
      IaV.data = hiconc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], clevs=[0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hiconc_sep', 'm', 'ice equiv. thickness September')
      IaV.data = hiconc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,6], clevs=[0.1, 0.2, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6], cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sea ice equiv. thickness SH', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'snow_thickness_sh'
    if fig_name in fig_names:
      hca, hcb = pyic.arrange_axes(2,1, plot_cb=True, asp=1., fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="", ylabel="",
                   projection=ccrs.SouthPolarStereo(),
                                  )
      ii=-1

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hsconc_mar', 'm', 'snow equiv. thickness March')
      IaV.data = hsconc_mar
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      ii+=1; ax=hca[ii]; cax=hcb[ii]
      IaV = pyic.IconVariable('hsconc_sep', 'm', 'snow equiv. thickness September')
      IaV.data = hsconc_sep
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_monthly, IaV, clim=[0,1], cincr=0.05, cmap='RdYlBu_r',
                      projection='PlateCarree', xlim=[-180.,180.], ylim=[-90., -50.],
                      ax=ax, cax=cax,
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )

      for ax in hca:
        ax.set_extent([-180, 180, -90., -50.], ccrs.PlateCarree())
        ax.gridlines()
        ax.add_feature(cartopy.feature.LAND)
        ax.coastlines()

      FigInf = dict(long_name=IaV.long_name)
      save_fig('Snow equiv. thickness SH', path_pics, fig_name, FigInf)

    # --------------------------------------------------------------------------------
    # Load 3D ocean data
    # --------------------------------------------------------------------------------
    if do_ocean_plots:
      calc_bias = False
      for fig_name in fig_names:
        if '_bias' in fig_name: 
          calc_bias = True

      tmp_list = ['temp30w', 'temp_gzave', 'temp_azave', 'temp_ipzave', 
                  'salt30w', 'salt_gzave', 'salt_azave', 'salt_ipzave',
                  'arctic_budgets'
                 ]
      if np.any(np.in1d(fig_names, tmp_list)) or calc_bias:
        print('Load 3D temp and salt...')
        temp, it_ave = pyic.time_average(DIcD['to'], 'to', t1, t2, iz='all', use_xr=load_xarray_dset, load_xr_data=True)
        salt, it_ave = pyic.time_average(DIcD['so'], 'so', t1, t2, iz='all', use_xr=load_xarray_dset, load_xr_data=True)
        temp[temp==0.]=np.ma.masked
        salt[salt==0.]=np.ma.masked

      if calc_bias:
        print('Calculate bias')
        fpath_ckdtree = IcD.rgrid_fpath_dict[rgrid_name]
      
        f = Dataset(fpath_ref_data_oce, 'r')
        temp_ref = f.variables['T'][0,:,:]
        salt_ref = f.variables['S'][0,:,:]
        f.close()
        temp_ref[temp_ref==0.]=np.ma.masked
        salt_ref[salt_ref==0.]=np.ma.masked
      
        tbias = temp-temp_ref
        sbias = salt-salt_ref

    # --------------------------------------------------------------------------------
    # biases
    # --------------------------------------------------------------------------------
    # ---
    fig_name = 'sst_bias'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
      IaV.data = tbias[0,:]
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD, IaV, cmap='RdBu_r',
                      clim=10, clevs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10], do_write_data_range=True,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Tbias: surface', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'temp_bias_gzave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                      clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Tbias: global zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'temp_bias_azave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                      clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Tbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'temp_bias_ipzave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('temp_bias', 'deg C', 'temperature bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(tbias, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, cmap='RdBu_r',
                      clim=10, contfs=[-10,-7,-5,-3,-2,-1,-0.5,-0.1,0.1,0.5,1,2,3,5,7,10],
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Tbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'sss_bias'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
      IaV.data = sbias[0,:]
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD, IaV, clim=3., cmap='RdBu_r', cincr=0.1, do_write_data_range=True,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sbias: surface', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_bias_gzave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.1, contfs='auto',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sbias: global zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_bias_azave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.1, contfs='auto',
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sbias: Atlantic zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_bias_ipzave'
    if fig_name in fig_names:
      IaV = pyic.IconVariable('salt_bias', 'g/kg', 'salinity bias')
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(sbias, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, clim=1., cmap='RdBu_r', cincr=0.1, contfs='auto',
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig('Sbias: Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
    
    ## ---
    #fname = '%s_restart_oce_%s.nc' % (run, tstep)
    ## ---
    #
    ## ---
    #fig_name = 'vort'
    #if fig_name in fig_names:
    #  FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='vort', depth=0, it=0,
    #                      clim=1e-4, cincr=-1., cmap='RdBu_r',
    #                      **Ddict)
    #  FigInf['fpath'] = path_pics+fig_name+'.png'
    #  FigInf['title'] = 'Surface vorticity'
    #  plt.savefig(FigInf['fpath'])
    #  qp.add_subsection(FigInf['title'])
    #  qp.add_fig(FigInf['fpath'])
    
    # -------------------------------------------------------------------------------- 
    # Sections
    # -------------------------------------------------------------------------------- 
    #fname = '%s%s_%s.nc' % (run, oce_def, tstep)
    #Ddict = dict(
    #  #xlim=[-180.,180.], ylim=[-90.,90.],
    #  sec_name='30W_100pts',
    #  path_ckdtree=path_ckdtree,
    #            )
    
    # ---
    fig_name = 'temp30w'
    if fig_name in fig_names:
      IaV = DIcD['to'].vars['to']
      IaV.lon_sec, IaV.lat_sec, IaV.dist_sec, IaV.data = pyic.interp_to_section(temp, fpath_ckdtree=IcD.sec_fpath_dict[sec_name_30w])
      pyic.vplot_base(IcD, IaV, 
                      clim=[-2., 30.], cincr=2.0, cmap='cmo.thermal',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Temperature at 30W', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
      #                    IcD=IcD,
      #                    sec_name=sec_name_30w,
      #                    path_ckdtree=path_ckdtree)
      #save_fig('Temperature at 30W', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt30w'
    if fig_name in fig_names:
      IaV = DIcD['so'].vars['so']
      IaV.lon_sec, IaV.lat_sec, IaV.dist_sec, IaV.data = pyic.interp_to_section(salt, fpath_ckdtree=IcD.sec_fpath_dict[sec_name_30w])
      pyic.vplot_base(IcD, IaV, 
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Salinity at 30W', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
      #                    IcD=IcD,
      #                    sec_name=sec_name_30w,
      #                    path_ckdtree=path_ckdtree)
      #save_fig('Salinity at 30W', path_pics, fig_name, FigInf)
    
    # XX---
    fig_name = 'dens30w'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='rhopot', it=0,
                          t1=t1, t2=t2,
                          clim=[24., 29.], cincr=0.2, cmap='cmo.dense',
                          var_add=-1000.,
                          IcD=IcD,
                          sec_name=sec_name_30w,
                          path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('Density at 30W', path_pics, fig_name, FigInf)
    
    # -------------------------------------------------------------------------------- 
    # zonal averages
    # -------------------------------------------------------------------------------- 
    #Ddict = dict(
    #  path_ckdtree=path_ckdtree,
    #            )
    
    # ---
    fig_name = 'temp_gzave'
    if fig_name in fig_names:
      IaV = DIcD['to'].vars['to']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(temp, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[-2., 30.], cincr=2.0, cmap='cmo.thermal',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Temperature global zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
      #                    sec_name='zave:glob:%s'%rgrid_name,
      #                    IcD=IcD, path_ckdtree=path_ckdtree,)
      #save_fig('Temperature global zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'temp_azave'
    if fig_name in fig_names:
      IaV = DIcD['to'].vars['to']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(temp, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[-2., 30.], cincr=2.0, cmap='cmo.thermal',
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Temperature Atlantic zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
      #                    sec_name='zave:atl:%s'%rgrid_name,
      #                    IcD=IcD, xlim=[-30,90], path_ckdtree=path_ckdtree,)
      #save_fig('Temperature Atlantic zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'temp_ipzave'
    if fig_name in fig_names:
      IaV = DIcD['to'].vars['to']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(temp, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[-2., 30.], cincr=2.0, cmap='cmo.thermal',
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Temperature Indo-Pac. zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='to', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[-2.,30.], cincr=2.0, cmap='cmo.thermal',
      #                    sec_name='zave:indopac:%s'%rgrid_name,
      #                    IcD=IcD, xlim=[-30,65], path_ckdtree=path_ckdtree,)
      #save_fig('Temperature Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_gzave'
    if fig_name in fig_names:
      IaV = DIcD['so'].vars['so']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(salt, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Salinity global zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
      #                    sec_name='zave:glob:%s'%rgrid_name,
      #                    IcD=IcD, path_ckdtree=path_ckdtree,)
      #save_fig('Salinity global zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_azave'
    if fig_name in fig_names:
      IaV = DIcD['so'].vars['so']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(salt, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Salinity Atlantic zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
      #                    sec_name='zave:atl:%s'%rgrid_name,
      #                    IcD=IcD, xlim=[-30,90], path_ckdtree=path_ckdtree,)
      #save_fig('Salinity Atlantic zon. ave.', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'salt_ipzave'
    if fig_name in fig_names:
      IaV = DIcD['so'].vars['so']
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(salt, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD, IaV, 
                      clim=[32., 37.], cincr=0.25, cmap='cmo.haline',
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Salinity Indo-Pac. zon. ave.', path_pics, fig_name)
      #FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='so', it=0,
      #                    t1=t1, t2=t2,
      #                    clim=[32.,37.], cincr=0.25, cmap='cmo.haline',
      #                    sec_name='zave:indopac:%s'%rgrid_name,
      #                    IcD=IcD, xlim=[-30,65], path_ckdtree=path_ckdtree,)
      #save_fig('Salinity Indo-Pac. zon. ave.', path_pics, fig_name, FigInf)
    
    # -------------------------------------------------------------------------------- 
    # Transports
    # -------------------------------------------------------------------------------- 
    fig_name = 'ke_100m'
    if fig_name in fig_names:
      u, it_ave   = pyic.time_average(DIcD['u'], 'u', t1=t1, t2=t2, iz=k100)
      v, it_ave   = pyic.time_average(DIcD['v'], 'v', t1=t1, t2=t2, iz=k100)
      ke = 0.5*(u**2+v**2)
      IaV = pyic.IconVariable('ke', 'm^2/s^2', 'kinetic energy')
      IaV.data = ke
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD, IaV, clim=[-7,0], cincr=0.5, cmap='RdYlBu_r',
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      logplot=True, do_write_data_range=True,
                      title='log$_{10}$(kinetic energy) at %dm [m$^2$/s$^2$]'%(IcD.depthc[k100]),
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )
      save_fig('Kinetic energy 100m', path_pics, fig_name)
  
    fig_name = 'ke_2000m'
    if fig_name in fig_names:
      u, it_ave   = pyic.time_average(DIcD['u'], 'u', t1=t1, t2=t2, iz=k2000)
      v, it_ave   = pyic.time_average(DIcD['v'], 'v', t1=t1, t2=t2, iz=k2000)
      ke = 0.5*(u**2+v**2)
      IaV = pyic.IconVariable('ke', 'm^2/s^2', 'kinetic energy')
      IaV.data = ke
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD, IaV, clim=[-7,0], cincr=0.5, cmap='RdYlBu_r',
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      logplot=True, do_write_data_range=True,
                      title='log$_{10}$(kinetic energy) at %dm [m$^2$/s$^2$]'%(IcD.depthc[k2000]),
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )
      save_fig('Kinetic energy 2000m', path_pics, fig_name)
   
    if do_ocean_plots:
      tmp_plist = ['bstr', 'arctic_budgets', 'passage_transports']
      if np.any(np.in1d(fig_names, tmp_plist)):
        try:
          print('Load 3D mass_flux...')
          mass_flux, it_ave = pyic.time_average(DIcD['massflux'], 'mass_flux', t1, t2, iz='all')
          mass_flux_vint = mass_flux.sum(axis=0)
        except:
          try:
            mass_flux_vint, it_ave = pyic.time_average(DIcD['massflux'], 'verticallyTotal_mass_flux_e', t1, t2, iz='all')
          except:
            print("No mass flux found.")

    # XX ---
    fig_name = 'bstr'
    if fig_name in fig_names:
      print('Execute calc_bstr_vgrid...')
#      mass_flux_vint = mass_flux.sum(axis=0)
    
      # --- derive and interp bstr
      bstr = pyic.calc_bstr_vgrid(IcD, mass_flux_vint, lon_start=0., lat_start=90.)
      IaV = pyic.IconVariable('bstr', units='Sv', long_name='barotropic streamfunction',
                              coordinates='vlat vlon', is3d=False)
      IaV.data = bstr
      IaV.interp_to_rectgrid(fpath_ckdtree=fpath_ckdtree)
      # --- plot bstr
      pyic.hplot_base(IcD, IaV, cmap='RdBu_r',
                      clim=200, clevs=[-200,-160,-120,-80,-40,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,40,80,120,160,200], 
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Barotropic streamfunction', path_pics, fig_name)

    # ---
    fig_name = 'passage_transports'
    if fig_name in fig_names and os.path.exists(path_grid+'section_mask_'+gname+'.nc'):
      ax, cax, hm, Dstr = pyic.hplot_base(DIcD['massflux'], IaV, cmap='RdBu_r',
                      clim=200, clevs=[-200,-160,-120,-80,-40,-30,-25,-20,-15,-10,-5,5,10,15,20,25,30,40,80,120,160,200], 
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, asp=0.5, fig_size_fac=2,
                                   projection=ccrs.PlateCarree(),
                                  )
      ax = hca[0]
      cax = hcb[0]
      pyic.plot_settings(ax, template='global')
#      mass_flux_vint = mass_flux.sum(axis=0)

      f = Dataset(path_grid+'section_mask_'+gname+'.nc', 'r')
      snames = []
      for var in f.variables.keys():
          if var.startswith('mask'):
              snames += [var[5:]]
      Dmask = dict()
      Die = dict()
      Div = dict()
      Dtransp = dict()
      for var in snames:
          data = f.variables['mask_'+var][:]
          Dmask[var] = data
          exec('mask_%s = data' %var)
          data = f.variables['ie_'+var][:]
          data = data[data.mask==False]
          Die[var] = data
          exec('ie_%s = data'%var)
          data = f.variables['iv_'+var][:]
          data = data[data.mask==False]
          Div[var] = data
          exec('iv_%s = data'%var)
          Dtransp[var] = (mass_flux_vint*DIcD['massflux'].edge_length*Dmask[var]).sum()/1e6
      f.close()
      for var in snames:
          ax.plot(DIcD['massflux'].vlon[Div[var]], DIcD['massflux'].vlat[Div[var]], color='r')
          ax.text(DIcD['massflux'].vlon[Div[var]].mean()+3, DIcD['massflux'].vlat[Div[var]].mean(), 
                  '%.1f Sv'%np.abs(Dtransp[var]), 
                  color='r', fontsize=8,
                  bbox=dict(fc='w', ec='none', alpha=0.5))
      save_fig('Passage transports', path_pics, fig_name)

      # ---
      fig_name = 'tab_passage_transports'
      if fig_name in fig_names:
        data = np.zeros((len(snames),1))
        leftcol = []
        toprow  = ['transport [Sv]']
        for nn, var in enumerate(snames):
          data[nn,0] = np.abs(Dtransp[var]) 
          leftcol.append(var.replace('_',' ').title())
        text = pyicqp.write_table_html(data, leftcol=leftcol, toprow=toprow, prec='4.1f', width='40%') 
        save_tab(text, 'Tab: Passage transports', path_pics, fig_name)

    if do_ocean_plots:
      tmp_plist = ['arctic_budgets']
      if np.any(np.in1d(fig_names, tmp_plist)):
        print('Load 3D uo and vo...')
        uo, it_ave = pyic.time_average(DIcD['u'], 'u', t1, t2, iz='all', use_xr=load_xarray_dset, load_xr_data=True)
        vo, it_ave = pyic.time_average(DIcD['v'], 'v', t1, t2, iz='all', use_xr=load_xarray_dset, load_xr_data=True)
        uo[uo==0.]=np.ma.masked
        vo[vo==0.]=np.ma.masked

    # ---
    fig_name = 'arctic_budgets'
    if fig_name in fig_names:
      #try:
        from qp_arctic_budgets import arctic_budgets
        arctic_budgets(DIcD['massflux'], DIcD['ice'], DIcD['monthly'], t1, t2, temp, salt, mass_flux, uo, vo)
        save_fig('Arctic heat/water budgets', path_pics, fig_name)
      #except:
      #  print(f'::: Warning: Could not make plot {fig_name}. :::')
    
    # --- 
    #Ddict = dict(
    #  #xlim=[-180.,180.], ylim=[-90.,90.],
    #  sec_name='moc',
    #  path_ckdtree=path_ckdtree,
    #            )
    
    # ---
    fig_name = 'amoc'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='atlantic_moc', it=0,
                          t1=t1, t2=t2,
                          var_fac=1e-9,
                          clim=24, cincr=2., cmap='RdBu_r',
                          IcD=IcD_moc, xlim=[-30,90], sec_name='moc', path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('Atlantic MOC', path_pics, fig_name, FigInf)
    # ---
    fig_name = 'pmoc'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='pacific_moc', it=0,
                          t1=t1, t2=t2,
                          var_fac=1e-9,
                          clim=24, cincr=2., cmap='RdBu_r',
                          IcD=IcD_moc, xlim=[-30,65], sec_name='moc', path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('Pacific MOC', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'gmoc'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname_moc, var='global_moc', it=0,
                          t1=t1, t2=t2,
                          var_fac=1e-9,
                          clim=24, cincr=2., cmap='RdBu_r',
                          IcD=IcD_moc, sec_name='moc', path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('Global MOC', path_pics, fig_name, FigInf)
    
    # -------------------------------------------------------------------------------- 
    # implied heat flux
    # -------------------------------------------------------------------------------- 
    fig_name = 'heat_flux'
    if fig_name in fig_names:
      global_hfl,   it_ave = pyic.time_average(IcD_moc, 'global_hfl',   t1, t2, iz='all')
      atlantic_hfl, it_ave = pyic.time_average(IcD_moc, 'atlantic_hfl', t1, t2, iz='all')
      pacific_hfl,  it_ave = pyic.time_average(IcD_moc, 'pacific_hfl',  t1, t2, iz='all')
    
      f = Dataset(IcD_moc.flist_ts[0], 'r')
      lat_hlf = f.variables['lat'][:]
      f.close()
    
      hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="latitude", ylabel="",)
      ii=-1
      
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      ax.plot(lat_hlf, global_hfl/1e15, label='global_hfl')
      ax.plot(lat_hlf, atlantic_hfl/1e15, label='atlantic_hfl')
      ax.plot(lat_hlf, pacific_hfl/1e15, label='pacific_hfl')
      ax.grid(True)
      ax.legend()
      ax.set_title(f'implied heat transport [PW]')
    
      save_fig('Implied Heat transport', path_pics, fig_name)
    
      # -------------------------------------------------------------------------------- 

    # -------------------------------------------------------------------------------- 
    # heat flux
    # -------------------------------------------------------------------------------- 
    fig_name = 'heat_hfbasin'
    if fig_name in fig_names:
      global_hfbasin,   it_ave = pyic.time_average(IcD_moc, 'global_hfbasin',   t1, t2, iz='all')
      atlantic_hfbasin, it_ave = pyic.time_average(IcD_moc, 'atlantic_hfbasin', t1, t2, iz='all')
      pacific_hfbasin,  it_ave = pyic.time_average(IcD_moc, 'pacific_hfbasin',  t1, t2, iz='all')
    
      f = Dataset(IcD_moc.flist_ts[0], 'r')
      lat_hlf = f.variables['lat'][:]
      f.close()
    
      hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="latitude", ylabel="",)
      ii=-1
      
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      ax.plot(lat_hlf, global_hfbasin/1e15, label='global_hfbasin')
      ax.plot(lat_hlf, atlantic_hfbasin/1e15, label='atlantic_hfbasin')
      ax.plot(lat_hlf, pacific_hfbasin/1e15, label='pacific_hfbasin')
      ax.grid(True)
      ax.legend()
      ax.set_title(f'heat transport [PW]')
    
      save_fig('Heat transport', path_pics, fig_name)
    
    # -------------------------------------------------------------------------------- 
    # freshwater flux
    # -------------------------------------------------------------------------------- 
    fig_name = 'freshwater_flux'
    if fig_name in fig_names:
      global_wfl,   it_ave = pyic.time_average(IcD_moc, 'global_wfl',   t1, t2, iz='all')
      atlantic_wfl, it_ave = pyic.time_average(IcD_moc, 'atlantic_wfl', t1, t2, iz='all')
      pacific_wfl,  it_ave = pyic.time_average(IcD_moc, 'pacific_wfl',  t1, t2, iz='all')
    
      f = Dataset(IcD_moc.flist_ts[0], 'r')
      lat_hlf = f.variables['lat'][:]
      #ncv = f.variables['global_wfl']
      #long_name = ncv.long_name 
      #units = ncv.units
      f.close()
    
      hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
                   sharex=True, sharey=True, xlabel="latitude", ylabel="",)
      ii=-1
      
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      ax.plot(lat_hlf, global_wfl/1e6, label='global')
      ax.plot(lat_hlf, atlantic_wfl/1e6, label='Atlantic')
      ax.plot(lat_hlf, pacific_wfl/1e6, label='Pacific')
      ax.grid(True)
      ax.legend()
      ax.set_title(f'freshwater transport [10^6 m^3 s^-1]')
    
      FigInf = dict()
      #FigInf['fpath'] = fpath
      #FigInf['long_name'] = long_name
    
      save_fig('Freshwater transport', path_pics, fig_name, FigInf)
    
    # -------------------------------------------------------------------------------- 
    # time series
    # -------------------------------------------------------------------------------- 
    # --- ocean monitoring
    fname_mon = '%s%s_%s.nc' % (run, D_variable_container['mon'], tstep)

    fig_name = 'ts_amoc'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['amoc26n'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_heat_content'
    if fig_name in fig_names:
      try:
        FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['global_heat_content'], 
          t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
        )
        save_fig(fig_name, path_pics, fig_name)
      except:
        print(f'::: Warning: Could not make plot {fig_name}. :::')
    fig_name = 'ts_ssh'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['ssh_global'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_sst'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['sst_global'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_sss'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['sss_global'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_hfl'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon, ['HeatFlux_Total_global'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_wfl'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon,
        ['FrshFlux_Precipitation_global', 'FrshFlux_SnowFall_global', 'FrshFlux_Evaporation_global', 'FrshFlux_Runoff_global', 'FrshFlux_VolumeIce_global', 'FrshFlux_TotalOcean_global', 'FrshFlux_TotalIce_global', 'FrshFlux_VolumeTotal_global'], 
        title='Fresh water flux [m/s]',
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_ice_volume_nh'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon,
        ['ice_volume_nh', 'ice_volume_nh'], 
        title='sea ice volume Northern hemisphere [km^3]',
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, mode_ave=['max', 'min'], labels=['max', 'min'], use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_ice_volume_sh'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon,
        ['ice_volume_sh', 'ice_volume_sh'], 
        title='sea ice volume Southern hemisphere [km^3]',
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, mode_ave=['max', 'min'], labels=['max', 'min'], use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_ice_extent_nh'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon,
        ['ice_extent_nh', 'ice_extent_nh'], 
        title='sea ice extent Northern hemisphere [km^2]',
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, mode_ave=['max', 'min'], labels=['max', 'min'], use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_ice_extent_sh'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_mon, fname_mon,
        ['ice_extent_sh', 'ice_extent_sh'], 
        title='sea ice extent Southern hemisphere [km^2]',
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, mode_ave=['max', 'min'], labels=['max', 'min'], use_tave_int_for_ts=use_tave_int_for_ts,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
  
    # --- atmosphere monitoring
    fig_name = 'ts_t2m_gmean'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['tas_gmean'],
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file,
        var_add=-273.15, units='$^o$C', use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm, do_djf=do_djf, do_jja=do_jja,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_radtop_gmean'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['radtop_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm_rad, do_djf=do_djf, do_jja=do_jja,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_rsdt_gmean'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['rsdt_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm_rad, do_djf=do_djf, do_jja=do_jja,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_rsut_gmean'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['rsut_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm_rad, do_djf=do_djf, do_jja=do_jja,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_rlut_gmean'
    if fig_name in fig_names:
      if not do_conf_dwd:
         vfc = 1.
      else:
         vfc = -1.
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['rlut_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, var_fac=vfc, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm_rad, do_djf=do_djf, do_jja=do_jja,
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_prec_gmean'
    if fig_name in fig_names:
      vfc = 86400. #  convert mm (kg m-2) --> mm/day
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['prec_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, var_fac=vfc, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm_prec, do_djf=do_djf, do_jja=do_jja, units='mm/day',
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_evap_gmean'
    if fig_name in fig_names:
      vfc = 86400. #  convert mm (kg m-2) --> mm/day
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['evap_gmean'], 
        t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, var_fac=vfc, use_tave_int_for_ts=use_tave_int_for_ts,
        fpath_ref_data_atm=fpath_ref_data_atm, do_djf=do_djf, do_jja=do_jja, units='mm/day',
        save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
      )
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_pme_gmean'
    if fig_name in fig_names:
      if do_conf_dwd:
         vfc = 86400. #  convert mm (kg m-2) --> mm/day
         FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['pme_gmean'], 
           t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, var_fac=vfc, use_tave_int_for_ts=use_tave_int_for_ts,
           fpath_ref_data_atm=fpath_ref_data_atm, do_djf=do_djf, do_jja=do_jja, units='mm/day',
           save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
         )
         save_fig(fig_name, path_pics, fig_name)
      else:
         print('P-E only for ICON-NWP implemented... skipping...')
    fig_name = 'ts_fwfoce_gmean'
    if fig_name in fig_names:
      if not do_conf_dwd:
         FigInf, Dhandles = pyicqp.qp_timeseries(IcD_atm_mon, fname_atm_mon, ['fwfoce_gmean'], 
           t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts,
           fpath_ref_data_atm=fpath_ref_data_atm, do_djf=do_djf, do_jja=do_jja,
           save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
         )
         save_fig(fig_name, path_pics, fig_name)
      else:
         print('Fresh water flux not yet implemented for ICON-NWP... skipping...')

    # -------------------------------------------------------------------------------- 
    # Overview table 
    # -------------------------------------------------------------------------------- 
    fig_name = 'tab_overview'
    if fig_name in fig_names:
      data = []
      toprow = []

#var = 'amoc26n'
#Dd['tab_name'] = 'AMOC 26N [kg/s]'
#Dd['fac']      = 1e9
#Dd['prec']     = '.1f'

      if do_ocean_plots:
        varlist = ['amoc26n', 'ice_volume_nh', 'ice_volume_sh']
        Dd = pyicqp.time_averages_monitoring(IcD_mon, t1, t2, varlist)
        for var in varlist:
          val = Dd[var]['ave']*Dd[var]['fac']
          data.append( f"{val:{Dd[var]['prec']}}" )
          if Dd[var]['tab_name']=='':
            tab_name = f"{Dd[var]['long_name']} [{Dd[var]['units']}]"
          toprow.append( tab_name )

      if do_atmosphere_plots:
        if do_conf_dwd:
          varlist = ['tas_gmean', 'radtop_gmean', 'rsdt_gmean', 'rsut_gmean', 'rlut_gmean', 'prec_gmean', 'evap_gmean', 'pme_gmean'] # 'fwfoce_gmean']
          var_add_list = [-273.15, 0, 0, 0, 0, 0, 0, 0]
          var_units_list = ['$^o$C', '', '', '', '', '', '', '']
        else:
          varlist = ['tas_gmean', 'radtop_gmean', 'rsdt_gmean', 'rsut_gmean', 'rlut_gmean', 'prec_gmean', 'evap_gmean', 'fwfoce_gmean']
          var_add_list = [-273.15, 0, 0, 0, 0, 0, 0, 0]
          var_units_list = ['$^o$C', '', '', '', '', '', '', '']
        var_fac_list = [1]*len(varlist)
        Dd = pyicqp.time_averages_monitoring(IcD_atm_mon, t1, t2, varlist, var_add_list=var_add_list, var_fac_list=var_fac_list, var_units_list=var_units_list)
        for var in varlist:
          val = Dd[var]['ave']*Dd[var]['fac']
          data.append( f"{val:{Dd[var]['prec']}}" )
          if Dd[var]['tab_name']=='':
            tab_name = f"{Dd[var]['long_name']} [{Dd[var]['units']}]"
          toprow.append( tab_name )

      data = np.array(data)[np.newaxis,:]
      leftcol = [run]
      text = pyicqp.write_table_html(data, leftcol=leftcol, toprow=toprow, prec='.5g', width='100%') 
      save_tab(text, 'Tab: Overview', path_pics, fig_name)

    # -------------------------------------------------------------------------------- 
    # Additional plots
    # -------------------------------------------------------------------------------- 
    fname = '%s_idemix_%s.nc' % (run, tstep)
    #Ddict = dict(
    #  #xlim=[-180.,180.], ylim=[-90.,90.],
    #  sec_name='30W_100pts',
    #  path_ckdtree=path_ckdtree,
    #            )
    
    # ---
    fig_name = 'tke30w'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='tke', it=0,
                          t1=t1, t2=t2,
                          clim=[-8,0], cincr=-1., cmap='plasma',
                          logplot=True,
                          sec_name=sec_name_30w, path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('TKE at 30W', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'iwe30w'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='iwe', it=0,
                          t1=t1, t2=t2,
                          clim=[-8,0], cincr=-1., cmap='plasma',
                          logplot=True,
                          sec_name=sec_name_30w, path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('IWE at 30W', path_pics, fig_name, FigInf)
    
    # ---
    fig_name = 'kv30w'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_vplot(fpath=path_data+fname, var='K_tracer_h_to', it=0,
                          t1=t1, t2=t2,
                          clim=[-8,0], cincr=-1., cmap='plasma',
                          logplot=True,
                          sec_name=sec_name_30w, path_ckdtree=path_ckdtree,
                          save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                         )
      save_fig('k_v at 30W', path_pics, fig_name, FigInf)
  
    # -------------------------------------------------------------------------------- 
    # Surface fluxes
    # -------------------------------------------------------------------------------- 

    # ---
    fig_name = 'atm_tauu'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vtauu, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1e3,
                               units='mN/m$^2$',
                               clim=[-200.,200.], clevs=[-200,-100,-50,-20,0,20,50,100,200], cmap='RdYlBu_r',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('zonal wind stress', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_tauu_bias'
    if fig_name in fig_names and exist_ref:
      var = vtauu
      var_ref = 'ewss' # eastward turbulent stress
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:] / 86400 # N/m^2*s --> N/m^2 (with daily accumulation)
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'mN/m$^2$', 'zonal wind stress bias')
      IaV.data = data_bias*1e3
      pyic.hplot_base(IcD_atm2d, IaV, clim=100, clevs=[-100,-80,-60,-40,-20,0,20,40,60,80,100], cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],  do_write_data_range=True,
                      #land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_tauv'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vtauv, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1e3,
                               units='mN/m$^2$',
                               clim=[-200.,200.], clevs=[-200,-100,-50,-20,0,20,50,100,200], cmap='RdYlBu_r',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('meridional wind stress', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_tauv_bias'
    if fig_name in fig_names and exist_ref:
      var = vtauv
      var_ref = 'nsss' # northward turbulent stress
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:] / 86400 # N/m^2*s --> N/m^2 (with daily accumulation)
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'mN/m$^2$', 'meridional wind stress bias')
      IaV.data = data_bias*1e3
      pyic.hplot_base(IcD_atm2d, IaV, clim=100, clevs=[-100,-80,-60,-40,-20,0,20,40,60,80,100], cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],  do_write_data_range=True,
                      #land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_curl_tau'
    if fig_name in fig_names:
      IcD_atm2d.edge2cell_coeff_cc_t = pyic.calc_edge2cell_coeff_cc_t(IcD_atm2d) # to calculate wind stress curl
      tauu, it_ave   = pyic.time_average(IcD_atm2d, vtauu, t1=t1, t2=t2, iz=0, it_ave=it_ave_atm)
      tauv, it_ave   = pyic.time_average(IcD_atm2d, vtauv, t1=t1, t2=t2, iz=0, it_ave=it_ave_atm)
      if not do_conf_dwd:
         tauu[IcD_atm2d.cell_sea_land_mask>0] = np.ma.masked
         tauv[IcD_atm2d.cell_sea_land_mask>0] = np.ma.masked
      else:
         # ICON-NWP gridfiles do not contain land-sea mask. i
         # We take it therefore from output files (real variable 
         # 'fr_land \in [0., 1.]').
         fdt=tave_int[0]
         fdt=fdt[0:4]+fdt[5:7]+fdt[8:10]
         # fr_land does not have a time dimension, it is so far 
         # constant so time averaging does not work for it. We
         # simply take it from the first file of the period.
         file_frl = '%s_atm_2d_ml_' % (run)
         file_frl = file_frl+fdt+'T000000Z.nc'
         fpath_frl=path_data+file_frl
         if os.path.exists(fpath_frl):
            f = Dataset(fpath_frl, 'r')
            frl = f.variables['fr_land'][:]
            f.close()
            tauu = np.ma.masked_where(frl>0., tauu, copy=False)
            tauv = np.ma.masked_where(frl>0., tauv, copy=False)
         else:
            print ('File not found for reading land-sea mask... continue without.')
      p_tau = pyic.calc_3d_from_2dlocal(IcD_atm2d, tauu[np.newaxis,:], tauv[np.newaxis,:])
      ptp_tau = pyic.cell2edges(IcD_atm2d, p_tau)
      curl_tau = pyic.calc_curl(IcD_atm2d, ptp_tau)
      IaV = pyic.IconVariable('curl_tau', '1e-7 N m-3', 'wind stress curl', coordinates='vlat vlon')
      IaV.data = curl_tau[0,:]/1e-7
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm2d, IaV, clim=4., contfs=[-4,-3,-2,-1.5,-1.,-0.75,-0.5,-0.25,0.25,0.5,0.75,1.,1.5,2,3,4], cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='0.7', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('wind stress curl', path_pics, fig_name)
  
    # ---
    fig_name = 'atm_wek'
    if fig_name in fig_names:
      w_ek = pyic.calc_curl(IcD_atm2d, ptp_tau/IcD_atm2d.fe/IcD_atm2d.rho0)
      w_ek = w_ek[0,:]
      w_ek[np.abs(IcD_atm2d.vlat)<5.] = np.ma.masked
      IaV = pyic.IconVariable('w_ek', 'm / year', 'Ekman pumping', coordinates='vlat vlon')
      IaV.data = w_ek*86400*365
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm2d, IaV, clim=60., cincr=10., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='0.7', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('Ekman pumping', path_pics, fig_name)

    # ---
    fig_name = 'atm_surf_shfl'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vshfl_s, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1.,
                               units='$W m^{-2}$',
                               clim=[-150.,150.], cincr=25.0, cmap='RdBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('surface sensible heat flux', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_surf_shfl_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vshfl_s
      var_ref = 'sshf'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:] / 86400 # J/m^2 --> W/m^2 (with daily accumulation)
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'surface sensible heat flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=80., cincr=10., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_surf_lhfl'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vlhfl_s, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1.,
                               units='$W m^{-2}$',
                               clim=[-250.,0.], cincr=25.0, cmap='RdYlBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('surface latent heat flux', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_surf_lhfl_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vlhfl_s
      var_ref = 'slhf'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:] / 86400 # J/m^2 --> W/m^2 (with daily accumulation)
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'surface latent heat flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=140., cincr=20., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # -------------------------------------------------------------------------------- 
    # TOA fluxes
    # -------------------------------------------------------------------------------- 

    # ---
    fig_name = 'atm_toa_netrad'
    if fig_name in fig_names and do_conf_dwd:
      # --- interpolate data
      var1 = vsob_t
      var2 = vthb_t
      data2d1, it_ave = pyic.time_average(IcD_atm2d, var1, t1, t2, iz=0, it_ave=it_ave_atm)
      data2d2, it_ave = pyic.time_average(IcD_atm2d, var2, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di1 = pyic.interp_to_rectgrid(data2d1, fpath_ckdtree_atm, coordinates='clat clon')
      lon, lat, data2di2 = pyic.interp_to_rectgrid(data2d2, fpath_ckdtree_atm, coordinates='clat clon')
      # --- calculate incoming short wave
      datai = data2di2 + data2di1
      IaV = pyic.IconVariable('data', '$W m^{-2}$', 'TOA net radiation flux')
      IaV.data = datai
      pyic.hplot_base(IcD_atm2d, IaV, clim=120., cincr=20., cmap='RdBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_netrad_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var_ref1 = 'tsr' # ERA5 name sw net 
      var_ref2 = 'ttr' # ERA5 name lw net
      fpath_ref = fpath_ref_data_atm
      vfc = 1./86400. # J/m^2 --> W/m^2 (with daily accumulation)
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref1 = 'solar_mon'      # CERES name sw in
        var_ref2 = 'toa_sw_all_mon' # CERES name sw out
        var_ref3 = 'toa_lw_all_mon' # CERES name lw out
        vfc = 1. # already in W/m^2
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref1 = f.variables[var_ref1][:,:] * vfc
      data_ref2 = f.variables[var_ref2][:,:] * vfc
      if not exist_ceres:
        data_ref = data_ref1+data_ref2
      else:
        data_ref3 = f.variables[var_ref3][:,:] * vfc
        data_ref = data_ref1-data_ref2-data_ref3
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'TOA net radiation flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=50., cincr=5., cmap='RdBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sob'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vsob_t, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1.,
                               units='$W m^{-2}$',
                               title='TOA short wave net flux',
                               clim=[0.,350.], cincr=25.0, cmap='RdYlBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('TOA short wave net flux', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sob_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vsob_t
      var_ref = 'tsr' # ERA5 name
      fpath_ref = fpath_ref_data_atm
      vfc = 1./86400. # J/m^2 --> W/m^2 (with daily accumulation)
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref = 'solar_mon'       # CERES name sw in
        var_ref2 = 'toa_sw_all_mon' # CERES name sw out
        vfc = 1. # already in W/m^2
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      if exist_ceres:
        data_ref2 = f.variables[var_ref2][:,:] * vfc
        data_ref = data_ref-data_ref2
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'TOA short wave net flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=80., cincr=10., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sod'
    if fig_name in fig_names and do_conf_dwd:
      # --- interpolate data
      var1 = vsou_t
      var2 = vsob_t
      data2d1, it_ave = pyic.time_average(IcD_atm2d, var1, t1, t2, iz=0, it_ave=it_ave_atm)
      data2d2, it_ave = pyic.time_average(IcD_atm2d, var2, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di1 = pyic.interp_to_rectgrid(data2d1, fpath_ckdtree_atm, coordinates='clat clon')
      lon, lat, data2di2 = pyic.interp_to_rectgrid(data2d2, fpath_ckdtree_atm, coordinates='clat clon')
      # --- calculate incoming short wave
      datai = data2di2 + data2di1
      IaV = pyic.IconVariable('data', '$W m^{-2}$', 'TOA short wave incoming flux')
      IaV.data = datai
      pyic.hplot_base(IcD_atm2d, IaV, clim=[0.,400.], cincr=25., cmap='RdYlBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sod_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var_ref = 'tisr' # ERA5 name sw in
      fpath_ref = fpath_ref_data_atm
      vfc = 1./86400. # J/m^2 --> W/m^2 (with daily accumulation)
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref = 'solar_mon' # CERES name sw out
        vfc = 1. # already in W/m^2
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'TOA short wave incoming flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=10., cincr=2.5, cmap='RdBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sou'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vsou_t, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1.,
                               title='TOA short wave outgoing flux',
                               units='$W m^{-2}$',
                               clim=[0.,200.], cincr=10., cmap='RdYlBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('TOA short wave outgoing flux', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_sou_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vsou_t
      var_ref = 'tisr' # ERA5 name sw in
      var_ref2 = 'tsr' # ERA5 name sw net
      fpath_ref = fpath_ref_data_atm
      vfc = 1./86400. # J/m^2 --> W/m^2 (with daily accumulation)
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref = 'toa_sw_all_mon' # CERES name sw out
        vfc = 1. # already in W/m^2
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      if not exist_ceres:
        data_ref2 = f.variables[var_ref2][:,:] * vfc
        data_ref = data_ref-data_ref2
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'TOA short wave outgoing flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=80., cincr=10., cmap='RdBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_thb'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vthb_t, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=1.,
                               units='$W m^{-2}$',
                               title='TOA long wave net flux',
                               clim=[-300.,0.], cincr=25.0, cmap='RdYlBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('TOA long wave net flux', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_toa_thb_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vthb_t
      var_ref = 'ttr' # ERA5 name
      fpath_ref = fpath_ref_data_atm
      vfc = 1./86400. # J/m^2 --> W/m^2 (with daily accumulation)
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref = 'toa_lw_all_mon' # CERES name
        vfc = -1. # already in W/m^2 (but with negative sign)
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$W m^{-2}$', 'TOA long wave net flux bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=50., cincr=5., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # -------------------------------------------------------------------------------- 
    # Atmosphere 2D
    # -------------------------------------------------------------------------------- 

    # ----------  screen level parameters & vertically integrated quantities
    # ---
    fig_name = 'sea_ts' # we have it to check what's in the forcing of uncoupled runs
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vsst, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_add=-273.15,
                               units = '$^o$C',
                               clim=[0.,24.], cincr=2.0, cmap='cmo.thermal',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('sea surface temperature', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'seaice_fraction' # we have it to check what's in the forcing of uncoupled runs
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vfrsi, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_add=0.,
                               units = '',
                               clim=[0.,1.], cincr=0.1, cmap='RdBu',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('sea ice fraction', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_psl'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vpsl, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_add=-1000.,
                               var_fac=0.01,
                               units='hPa-1000',
                               clim=[-25.,25.], cincr=5.0, cmap='RdBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('sea level pressure', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_psl_bias'
    if fig_name in fig_names and exist_ref:
      var = vpsl
      var_ref = 'msl'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:]
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'hPa', 'sea level pressure bias')
      IaV.data = data_bias/100.
      pyic.hplot_base(IcD_atm2d, IaV, clim=10., cincr=1., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_w10m'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vsfcwind, it=0, iz=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               units = '$m s^{-1}$',
                               clim=[0., 10.], cincr=1., cmap='RdYlBu_r',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('10m wind speed', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_w10m_bias'
    if fig_name in fig_names and exist_ref:
      var = vsfcwind
      var_ref = 'si10'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:]
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$m s^{-1}$', '10m wind speed bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=6., cincr=0.5, cmap='RdBu_r',
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_ts'
    if fig_name in fig_names and do_conf_dwd:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vts, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_add=-273.15,
                               units = '$^o$C',
                               clim=[-30.,30.], cincr=5.0, cmap='cmo.thermal',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('surface temperature', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_ts_bias'
    if fig_name in fig_names and do_conf_dwd and exist_ref:
      var = vts
      var_ref = 'skt'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:]
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$^o$C', 'surface temperature bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=24., cincr=2., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_t2m'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vtas, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_add=-273.15,
                               units = '$^o$C',
                               clim=[-30.,30.], cincr=5.0, cmap='cmo.thermal',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('2m temperature', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_t2m_bias'
    if fig_name in fig_names and exist_ref:
      var = vtas
      var_ref = 't2m'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:]
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$^o$C', '2m temperature bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=12., cincr=1., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_cwv'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vprw, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               units = '$kg m^{-2}$',
                               clim=[0.,50.], cincr=5., cmap='BrBG',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('column water vapour', path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_cwv_bias'
    if fig_name in fig_names and exist_ref:
      var = vprw
      var_ref = 'tcwv'
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][:,:]
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$kg m^{-2}$', 'column water vapour bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=20., cincr=2., cmap='BrBG', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_tcc'
    if fig_name in fig_names:
      if not do_conf_dwd:
         vfc = 100.
      else:
         vfc = 1.
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vclt, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=vfc,
                               units='%',
                               clim=[0.,100.], cincr=10.0, cmap='BrBG',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('total cloud cover', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_tcc_bias'
    if fig_name in fig_names and exist_ref:
      var = vclt
      var_ref = 'tcc' # ERA5 name
      fpath_ref = fpath_ref_data_atm
      vfc = 100.
      if exist_ceres:
        fpath_ref = fpath_ref_data_atm_rad
        var_ref = 'cldarea_total_daynight_mon' # CERES name
        vfc = 1. # already in %
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      if not do_conf_dwd:
        datai = data2di * 100.
      else:
        datai = data2di
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '%', 'total cloud cover bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=60., cincr=10., cmap='BrBG', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_cllvi_clivi'
    if fig_name in fig_names:
      cllvi, it_ave = pyic.time_average(IcD_atm2d, vcllvi, t1, t2, iz=0, it_ave=it_ave_atm)
      clivi, it_ave = pyic.time_average(IcD_atm2d, vclivi, t1, t2, iz=0, it_ave=it_ave_atm)
      IaV = pyic.IconVariable('cclvi_clivi', 'g/m$^2', 'Liquid water + ice content')
      IaV.data = (cllvi+clivi)*1e3
      pyic.hplot_base(IcD_atm2d, IaV, clim=[10,300], clevs=[10,50,100,200,300], cmap='RdYlBu_r', 
                      use_tgrid=False,
                      projection=projection, do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_tp'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var=vpr, it=0,
                               it_ave=it_ave_atm,
                               t1=t1, t2=t2,
                               var_fac=86400.,
                               units='mm/day',
                               title='total precipitation',
                               clim=[0.,16.], cincr=1., cmap='RdYlBu',
                               land_facecolor='none',
                               IcD=IcD_atm2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('total precipitation', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_tp_bias'
    if fig_name in fig_names and exist_ref:
      var = vpr
      var_ref = 'tp' # ERA5 name
      fpath_ref = fpath_ref_data_atm
      vfc = 1000 # m/day --> mm/day
      if exist_gpm:
        fpath_ref = fpath_ref_data_atm_prec
        var_ref = 'precipitation' # GPM name sw in
        vfc = 24 # mm/h --> mm/day
      # --- interpolate data
      data2d, it_ave = pyic.time_average(IcD_atm2d, var, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di * 86400
      # --- reference
      f = Dataset(fpath_ref, 'r')
      data_ref = f.variables[var_ref][:,:] * vfc
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'mm/day', 'total precipitation bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=10., cincr=1., cmap='RdBu', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

  
    # ---
    fig_name = 'atm_pme'
    if fig_name in fig_names:
      pr, it_ave = pyic.time_average(IcD_atm2d, vpr, t1, t2, iz=0, it_ave=it_ave_atm)
      evspsbl, it_ave = pyic.time_average(IcD_atm2d, vevspsbl, t1, t2, iz=0, it_ave=it_ave_atm)
      IaV = pyic.IconVariable('pme', 'mm/day', 'P-E')
      IaV.data = (pr+evspsbl)*86400
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm2d, IaV, clim=15., cincr=1.5, cmap='RdBu', 
                      use_tgrid=False,
                      projection=projection, do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_pme_bias'
    if fig_name in fig_names and exist_ref:
      var1 = vpr
      var2 = vevspsbl
      var_ref1 = 'tp'
      var_ref2 = 'e'
      # --- interpolate data
      pr, it_ave = pyic.time_average(IcD_atm2d, var1, t1, t2, iz=0, it_ave=it_ave_atm)
      evspsbl, it_ave = pyic.time_average(IcD_atm2d, var2, t1, t2, iz=0, it_ave=it_ave_atm)
      lon, lat, pri = pyic.interp_to_rectgrid(pr, fpath_ckdtree_atm, coordinates='clat clon')
      lon, lat, evspsbli = pyic.interp_to_rectgrid(evspsbl, fpath_ckdtree_atm, coordinates='clat clon')
      datai = (pri+evspsbli)*86400
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref1 = f.variables[var_ref1][:,:] * 1000 # m/day --> mm/day
      data_ref2 = f.variables[var_ref2][:,:] * 1000 # m/day --> mm/day
      data_ref = data_ref1 + data_ref2
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'mm/day', 'P-E bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm2d, IaV, clim=10., cincr=1., cmap='RdBu', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    if do_atmosphere_plots:
      print('Calculate atmosphere pressure interpolation weights...')
      # --- load 3D pressure
      pfull, it_ave = pyic.time_average(IcD_atm3d, vpfull, t1, t2, iz='all', it_ave=it_ave_atm)
      # linear vert ax
      IcD_atm3d.plevc = np.array([100000, 97500, 95000, 92500, 90000, 87500, 85000, 82500, 80000, 77500, 75000, 70000, 65000, 60000, 55000, 50000, 45000, 40000, 35000, 30000, 25000, 22500, 20000, 17500, 15000, 12500, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300, 200, 100])
      icall, ind_lev, fac = pyic.calc_vertical_interp_weights(pfull, IcD_atm3d.plevc)
      # log10 vert ax
      IcD_atm3d.plev_log = np.array([100900,99500,97100,93900,90200,86100,81700,77200,72500,67900,63300,58800,54300,49900,45700,41600,37700,33900,30402,27015,23833,20867,18116,15578,13239,11066,9102,7406,5964,4752,3743,2914,2235,1685,1245,901,637,440,296,193,122,74,43,23,11,4,1])
      icall_log, ind_lev_log, fac_log = pyic.calc_vertical_interp_weights(pfull, IcD_atm3d.plev_log)
      ip250 = np.argmin((IcD_atm3d.plevc-250e2)**2)
      ip500 = np.argmin((IcD_atm3d.plevc-500e2)**2)
      ip700 = np.argmin((IcD_atm3d.plevc-700e2)**2)
      ip850 = np.argmin((IcD_atm3d.plevc-850e2)**2)

    # ----------  selected variables at selected pressure levels
    # ---
    fig_name = 'atm_u_250'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vua, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      IaV = pyic.IconVariable('datavi', '$m s^{-1}$', 'u-comp of wind @ 250 hPa')
      IaV.data = datavi[ip250,:]
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm3d, IaV, clim=[-50.,50.], cincr=10., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name)

    # ---
    fig_name = 'atm_u_250_bias'
    if fig_name in fig_names and exist_ref:
      var = vua
      var_ref = 'u'
      # --- interpolate data
      data, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      data2d = datavi[ip250,:]
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][5,:,:] # dim=5 --> 250 hPa
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$m s^{-1}$', 'u-comp of wind @ 250 hPa bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm3d, IaV, clim=[-10.,10.], cincr=2., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_v_250'
    if fig_name in fig_names:
      ta, it_ave = pyic.time_average(IcD_atm3d, vva, t1, t2, iz='all', it_ave=it_ave_atm)
      tavi = ta[ind_lev,icall]*fac+ta[ind_lev-1,icall]*(1.-fac)
      IaV = pyic.IconVariable('tavi', '$m s^{-1}$', 'v-comp of wind @ 250 hPa')
      IaV.data = tavi[ip250,:]
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm3d, IaV, clim=[-10.,10.], cincr=1., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name)

    # ---
    fig_name = 'atm_v_250_bias'
    if fig_name in fig_names and exist_ref:
      var = vva
      var_ref = 'v'
      # --- interpolate data
      data, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      data2d = datavi[ip250,:]
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][5,:,:] # dim=5 --> 250 hPa
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$m s^{-1}$', 'v-comp of wind @ 250 hPa bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm3d, IaV, clim=4., cincr=0.5, cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_geop_500'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vzg, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev+1,icall]*(1.-fac)
      IaV = pyic.IconVariable('datavi', 'm', 'geopotential height @ 500 hPa')
      IaV.data = datavi[ip500,:]
      if do_conf_dwd:
        IaV.data *= 1/9.81
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm3d, IaV, clim=[5000,6000], cincr=100., cmap='RdYlBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name)

    # ---
    fig_name = 'atm_geop_500_bias'
    if fig_name in fig_names and exist_ref:
      var = vzg
      var_ref = 'z'
      # --- interpolate data
      data, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev+1,icall]*(1.-fac)
      data2d = datavi[ip500,:]
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][3,:,:] / 9.81 # dim=3 --> 500 hPa
      f.close()
      # --- calculate bias
      if do_conf_dwd:
        datai *= 1/9.81
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', 'm', 'geopotential @ 500 hPa bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm3d, IaV, clim=[-150.,150], cincr=25., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_relhum_700'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vhur, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      IaV = pyic.IconVariable('datavi', '%', 'relative humidity @ 700 hPa')
      IaV.data = datavi[ip700,:]
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm3d, IaV, clim=[0.,100.], cincr=10., cmap='BrBG', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name)

    # ---
    fig_name = 'atm_relhum_700_bias'
    if fig_name in fig_names and exist_ref:
      var = vhur
      var_ref = 'r'
      # --- interpolate data
      data, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      data2d = datavi[ip700,:]
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][2,:,:] # dim=2 --> 700 hPa
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '%', 'relative humidity @ 700 hPa bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm3d, IaV, clim=40., cincr=5., cmap='BrBG', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_temp_850'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vta, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      datavi += -273.15
      IaV = pyic.IconVariable('datavi', '$^o$C', 'temperature @ 850 hPa')
      IaV.data = datavi[ip850,:]
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      pyic.hplot_base(IcD_atm3d, IaV, clim=[-35,35], cincr=5., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      do_write_data_range=True,
                      land_facecolor='none',
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name)

    # ---
    fig_name = 'atm_temp_850_bias'
    if fig_name in fig_names and exist_ref:
      var = vta
      var_ref = 't'
      # --- interpolate data
      data, it_ave = pyic.time_average(IcD_atm3d, var, t1, t2, iz='all', it_ave=it_ave_atm)
      datavi = data[ind_lev,icall]*fac+data[ind_lev-1,icall]*(1.-fac)
      data2d = datavi[ip850,:]
      lon, lat, data2di = pyic.interp_to_rectgrid(data2d, fpath_ckdtree_atm, coordinates='clat clon')
      datai = data2di
      # --- reference
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref = f.variables[var_ref][1,:,:] # dim=1 --> 850 hPa
      f.close()
      # --- calculate bias
      data_bias = datai-data_ref
      IaV = pyic.IconVariable('data_bias', '$^o$C', 'temperature @ 850 hPa bias')
      IaV.data = data_bias
      pyic.hplot_base(IcD_atm3d, IaV, clim=10., cincr=1., cmap='RdBu_r', 
                      use_tgrid=False,
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.], 
                      land_facecolor='none', do_write_data_range=True,
                      asp=0.5,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      FigInf = dict(long_name=IaV.long_name)
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ----------  zonal averages
    # --- temperature
    fig_name = 'atm_temp_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vta, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)

      IaV = pyic.IconVariable('temp', '$^o$C', 'temperature')
      IaV.data = 1.*data_zave
      IaV.data += -273.15
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[-80., 30.], cincr=10., contfs='auto', cmap='RdYlBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_temp_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['t_zm'][:,:]
      f.close()

      IaV = pyic.IconVariable('temp bias', '$^o$C', 'temperature bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=10, contfs=[-10,-5,-2,-1,-0.5,0.5,1,2,5,10], cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_logv_temp_zave'
    if fig_name in fig_names:
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev_log, fac_log, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('temp', '$^o$C', 'log temperature')
      IaV.data = 1.*data_zave
      IaV.data += -273.15
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[-80., 30.], cincr=5, contfs='auto',
                      cmap='RdYlBu_r',
                      vertaxtype='log10', daxl=2.1,
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # --- zon. vel.
    fig_name = 'atm_u_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vua, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('u', 'm/s', 'u-comp of wind')
      IaV.data = 1.*data_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=30., cincr=5., contfs='auto',
                      cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_u_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['u_zm'][:,:]
      f.close()
      IaV = pyic.IconVariable('u', 'm/s', 'u-comp of wind bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=20, contfs=[-20,-10,-5,-2,-1,1,2,5,10,20], cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_logv_u_zave'
    if fig_name in fig_names:
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev_log, fac_log, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('u', 'm/s', 'log u-comp of wind')
      IaV.data = 1.*data_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=30., cincr=5, contfs='auto',
                      cmap='RdBu_r',
                      vertaxtype='log10', daxl=2.1,
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # --- mer. vel.
    fig_name = 'atm_v_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vva, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('v', 'm/s', 'v-comp of wind')
      IaV.data = 1.*data_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=3, cincr=0.5, contfs='auto',
                      cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_v_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['v_zm'][:,:]
      f.close()
      IaV = pyic.IconVariable('v', 'm/s', 'v-comp of wind bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=1, contfs=[-1.0,-0.5,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.5,1.0], cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_logv_v_zave'
    if fig_name in fig_names:
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev_log, fac_log, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('v', 'm/s', 'log v-comp of wind')
      IaV.data = 1.*data_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=3., cincr=0.5, contfs='auto',
                      cmap='RdBu_r',
                      vertaxtype='log10', daxl=2.1,
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_spechum_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vhus, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('hus', 'g/kg', 'specific humidity')
      IaV.data = data_zave * 1000.
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=0.5, contfs=[0.005,0.01,0.03,0.05,0.1,0.3,0.5,1.,2.,5.,8.], cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_spechum_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['q_zm'][:,:]
      f.close()
      IaV = pyic.IconVariable('hus', 'g/kg', 'specific humidity bias')
      IaV.data = data_zave - data_ref_zave * 1000
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=1, contfs=[-1.0,-0.5,-0.2,-0.1,-0.05,0.05,0.1,0.2,0.5,1.0], cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_relhum_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vhur, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('hur', '%', 'relative humidity')
      if do_conf_dwd:
         IaV.data = data_zave
      else:
         IaV.data = data_zave * 100.
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[0,100.], cincr=10., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_relhum_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['r_zm'][:,:]
      f.close()
      IaV = pyic.IconVariable('hur', '%', 'relative humidity bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=50., cincr=10., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_cc_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vcl, t1, t2, iz='all', it_ave=it_ave_atm)
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      IaV = pyic.IconVariable('cc', '%', 'cloud cover')
      if do_conf_dwd:
         IaV.data = data_zave
      else:
         IaV.data = data_zave * 100.
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2.5, contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_cc_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['cc_zm'][:,:] * 100.
      f.close()
      IaV = pyic.IconVariable('cc', '%', 'cloud cover bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=25., cincr=5., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_clw_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vclw, t1, t2, iz='all', it_ave=it_ave_atm)
      IaV = pyic.IconVariable('clw', 'mg/kg', 'cloud water')
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      data_zave *= 1e6
      IaV.data = data_zave
      clw = IaV.data # save it for the plot clw+cli
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[0,40.], cincr=2., contfs='auto',  cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_clw_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['clwc_zm'][:,:] * 1e6
      f.close()
      IaV = pyic.IconVariable('clw', 'mg/kg', 'cloud water bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=20., cincr=2., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_cli_zave'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vcli, t1, t2, iz='all', it_ave=it_ave_atm)
      IaV = pyic.IconVariable('cli', 'mg/kg', 'cloud ice')
      lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      data_zave *= 1e6
      IaV.data = data_zave
      cli = IaV.data # save it for the plot clw+cli
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, clim=[0,15.], cincr=1., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)

    # ---
    fig_name = 'atm_cli_zave_bias'
    if fig_name in fig_names and exist_ref:
      f = Dataset(fpath_ref_data_atm, 'r')
      data_ref_zave = f.variables['ciwc_zm'][:,:] * 1e6
      f.close()
      IaV = pyic.IconVariable('cli', 'mg/kg', 'cloud ice bias')
      IaV.data = data_zave - data_ref_zave
      IaV.lat_sec = lat_sec
      pyic.vplot_base(IcD_atm3d, IaV, 
                      clim=10., cincr=2., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_clwi_zave'
    if fig_name in fig_names:
      lat_sec = IaV.lat_sec
      IaV = pyic.IconVariable('clwi', 'mg/kg', 'cloud water+ice')
      IaV.data = clw+cli
      IaV.lat_sec = lat_sec 
      pyic.vplot_base(IcD_atm3d, IaV, clim=[0,25.], cincr=2., contfs='auto', cmap='BrBG',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # ---
    fig_name = 'atm_psi'
    if fig_name in fig_names:
      data, it_ave = pyic.time_average(IcD_atm3d, vva, t1, t2, iz='all', it_ave=it_ave_atm)
      IaV = pyic.IconVariable('psi', '10$^9$kg/s', 'meridional stream function')
      IaV.lat_sec, data_zave = pyic.zonal_average_atmosphere(data, ind_lev, fac, fpath_ckdtree_atm)
      plevi = np.concatenate(([107500],0.5*(IcD_atm3d.plevc[1:]+IcD_atm3d.plevc[:-1]),[0.]))
      dp = np.diff(plevi)
      IaV.data = (-2.*np.pi*6.371e6/9.81)*np.cos(IaV.lat_sec*np.pi/180.)* ( (data_zave[::-1,:]*dp[::-1,np.newaxis]).cumsum(axis=0) )[::-1,:]*1e-9
      pyic.vplot_base(IcD_atm3d, IaV, clim=[-80,80], cincr=10., contfs='auto', cmap='RdBu_r',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig(IaV.long_name, path_pics, fig_name, FigInf)
  
    # --- North Pole
  
    # ---
    fig_name = 'np_zonal_wind_stress'
    if fig_name in fig_names:
      tauu, it_ave   = pyic.time_average(IcD_atm2d, vtauu, iz='all', it_ave=it_ave_atm)
      IaV = pyic.IconVariable('tauu', 'mN/m$^2$', 'zonal wind stress')
      IaV.data = tauu*1e3
      IaV.interp_to_rectgrid(fpath_ckdtree_atm)
      ax, cax, hm, Dstr = pyic.hplot_base(IcD_atm2d, IaV, clim=200, cincr=25, cmap='RdYlBu_r',
                      projection='NorthPolarStereo', xlim=[-180.,180.], ylim=[60.,90.],
                      crs_features=False, do_plot_settings=False, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )
      pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[60,90], do_xyticks=False, do_xyminorticks=False, do_gridlines=True, land_facecolor='none')
      #FigInf = pyicqp.qp_hplot(fpath=path_data+fname, var='tauu', it=0,
      #                         t1=t1, t2=t2,
      #                         var_fac=1e3,
      #                         units='mN/m$^2$',
      #                         clim=[-200.,200.], cincr=25.0, cmap='RdYlBu_r',
      #                         IcD=IcD_atm2d, **Ddict_global)
      save_fig('zonal wind stress', path_pics, fig_name, FigInf)

    # ------------------------------------------------------------------------------
    # HAMOCC timeseries
    # ------------------------------------------------------------------------------

    # --- 
    fig_name = 'ts_global_npp'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_primary_production'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_nppcya'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_npp_cya'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_zoograzing'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_zooplankton_grazing'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_netco2flux'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_net_co2_flux'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_n2fix'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['N2_fixation'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_WC_denit'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['WC_denit'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_sed_denit'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['SED_denit'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_surface_alk'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_surface_alk'], t1=t1, t2=t2, ave_freq=ave_freq, var_fac=1e6, units='mmol m$^{-3}$', omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_surface_dic'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_surface_dic'], t1=t1, t2=t2, ave_freq=ave_freq, var_fac=1e6, units='mmol C m$^{-3}$', omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_surface_phos'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_surface_phosphate'], t1=t1, t2=t2, ave_freq=ave_freq, var_fac=1e6, units='mmol P m$^{-3}$', omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_surface_sil'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_surface_silicate'], t1=t1, t2=t2, ave_freq=ave_freq, var_fac=1e6, units='mmol Si m$^{-3}$', omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_opal_prod'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_opal_production'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_caco3_prod'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_caco3_production'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_OMexp90'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_OM_export_at_90m'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_calcexp90'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_calc_export_at_90m'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)
    fig_name = 'ts_global_opalexp90'
    if fig_name in fig_names:
      FigInf, Dhandles = pyicqp.qp_timeseries(IcD_ham_mon, fname_ham_mon, ['global_opal_export_at_90m'], t1=t1, t2=t2, ave_freq=ave_freq, omit_last_file=omit_last_file, use_tave_int_for_ts=use_tave_int_for_ts, save_data=save_data, fpath_nc=path_nc+fig_name+'.nc')
      save_fig(fig_name, path_pics, fig_name)

    # ------------------------------------------------------------------------------
    # HAMOCC surface maps
    # ------------------------------------------------------------------------------
    if do_hamocc_plots:
      # go through tracers seperately to avoid unncessary loading of all tracers
      tmp_list = ['dic_gzave', 'dic_azave', 'dic_ipzave']
      if np.any(np.in1d(fig_names, tmp_list)):
        dissic, it_ave = pyic.time_average(IcD_ham_inv, 'dissic', t1, t2, iz='all')
        dissic[dissic==0.]=np.ma.masked

      tmp_list = ['o2_gzave', 'o2_azave', 'o2_ipzave']
      if np.any(np.in1d(fig_names, tmp_list)):
        o2, it_ave = pyic.time_average(IcD_ham_inv, 'o2', t1, t2, iz='all')
        o2[o2==0.]=np.ma.masked
    

    # ---
    fig_name = 'srf_phyp'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='phyp', depth=0, it=0,
                               t1=t1, t2=t2,
                               clim=[0,0.8], cincr=0.05, cmap='cmo.algae',
                               var_fac=1e6, units='mmol P m$^{-3}$',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface phytoplankton', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_zoop'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='zoop', depth=0, it=0,
                               t1=t1, t2=t2,
                               clim=[0,0.08], cincr=0.005, cmap='cmo.dense',
                               var_fac=1e6, units='mmol P m$^{-3}$',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface zooplankton', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_cya'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='phydiaz', depth=0, it=0,
                               t1=t1, t2=t2,
                               clim=[0,0.5], cincr=0.025, cmap='cmo.amp',
                               var_fac=1e6, units='mmol P m$^{-3}$',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface cyanobacteria', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_alk'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='talk', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol m$^{-3}$',
                               clim=[2000,2600], cincr=40., cmap='cmo.haline',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface alkalinity', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_dic'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='dissic', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol m$^{-3}$',
                               clim=[1700,2300], cincr=40., cmap='cmo.haline',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface DIC', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_hion'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='hi', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol m$^{-3}$',
                               clim=[0.0,0.015], cincr=0.001, cmap='cmo.thermal',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface h+ ion conc.', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_pH'
    if fig_name in fig_names:
      hion, it_ave   = pyic.time_average(IcD_ham_inv, 'hi', t1=t1, t2=t2, iz=0)
      ph = - np.log10(hion)
      IaV = pyic.IconVariable('pH', '', 'pH')
      IaV.data = ph
      IaV.interp_to_rectgrid(fpath_ckdtree)
      pyic.hplot_base(IcD_ham_inv, IaV, clim=[7.8,8.4], cincr=0.025, cmap='plasma_r',
                      projection=projection, xlim=[-180.,180.], ylim=[-90.,90.],
                      do_write_data_range=True,
                      title='surface pH',
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                      )
      save_fig('Surface pH', path_pics, fig_name)

    # ---
    fig_name = 'srf_nitrate'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='no3', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol N m$^{-3}$',
                               clim=[0,40], cincr=2.5, cmap='cmo.matter',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface nitrate', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_phosphate'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='po4', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol P m$^{-3}$',
                               clim=[0,4.0], cincr=0.25, cmap='cmo.matter',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface phosphate', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_silicate'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_inv, var='si', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e6, units='mmol Si m$^{-3}$',
                               clim=[0,90], cincr=6, cmap='cmo.matter',
                               IcD=IcD_ham_inv,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('Surface silicate', path_pics, fig_name, FigInf)

    # ---
    fig_name = 'srf_co2flux'
    if fig_name in fig_names:
      FigInf = pyicqp.qp_hplot(fpath=path_data+fname_ham_2d, var='co2flux', depth=0, it=0,
                               t1=t1, t2=t2,
                               var_fac=1e9, units='$\mu$mol C m$^{-1}$ s$^{-1}$',
                               clim=[-0.3,0.3], cincr=0.03, cmap='cmo.balance',
                               IcD=IcD_ham_2d,
                               save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                               **Ddict_global)
      save_fig('CO2 flux', path_pics, fig_name, FigInf)

    # ------------------------------------------------------------------------------
    # HAMOCC sections
    # ------------------------------------------------------------------------------
    # ---
    fig_name = 'dic_gzave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['dissic']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(dissic*1e6, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[2000,2450], cincr=25.0, cmap='cmo.haline',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('DIC global zon. ave.', path_pics, fig_name)

    # ---
    fig_name = 'dic_azave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['dissic']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(dissic*1e6, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[2000,2450], cincr=25.0, cmap='cmo.haline',
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('DIC Atlantic zon. ave.', path_pics, fig_name)

    # ---
    fig_name = 'dic_ipzave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['dissic']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(dissic*1e6, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[2000,2450], cincr=25.0, cmap='cmo.haline',
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('DIC Indo-Pac. zon. ave.', path_pics, fig_name)

    # ---
    fig_name = 'o2_gzave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['o2']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(o2*1e6, basin='global', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[0,450], cincr=15, cmap='RdYlBu',
                      asp=0.5, do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('O2 global zon. ave.', path_pics, fig_name)

    # ---
    fig_name = 'o2_azave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['o2']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(o2*1e6, basin='atl', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[0,450], cincr=15, cmap='RdYlBu',
                      asp=0.5, xlim=[-30,90], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('O2 Atlantic zon. ave.', path_pics, fig_name)  

    # ---
    fig_name = 'o2_ipzave'
    if fig_name in fig_names:
      IaV = IcD_ham_inv.vars['o2']
      IaV.units='mmol m$^{-3}$'
      IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(o2*1e6, basin='indopac', 
                                 fpath_fx=IcD.fpath_fx, fpath_ckdtree=fpath_ckdtree)
      pyic.vplot_base(IcD_ham_inv, IaV, 
                      clim=[0,450], cincr=15, cmap='RdYlBu',
                      asp=0.5, xlim=[-30,65], do_write_data_range=True,
                      save_data=save_data, fpath_nc=path_nc+fig_name+'.nc',
                     )
      save_fig('O2 Indo-Pac. zon. ave.', path_pics, fig_name)

  # --------------------------------------------------------------------------------
  # Website
  # --------------------------------------------------------------------------------
  flist_all = glob.glob(path_pics+'*.json')
  
  print('Make QP website for the following figures:')
  for plot in plist:
    fpath = path_pics+plot+'.json'
    if plot.startswith('sec'):
      qp.add_section(plot.split(':')[1])
    elif plot.startswith('tab'):
      if os.path.exists(fpath):
        print(fpath)
        flist_all.remove(fpath)
        with open(fpath) as file_json:
          FigInf = json.load(file_json)
        qp.add_subsection(FigInf['title'])
        #rfpath_pics = rpath_pics+FigInf['name']
        qp.add_html(FigInf['fpath'])
      else:
        print('::: Warning: file does not exist: %s!:::' %fpath)
        #raise ValueError('::: Error: file does not exist: %s!:::' %fpath)
    else:
      if os.path.exists(fpath):
        print(fpath)
        flist_all.remove(fpath)
        with open(fpath) as file_json:
          FigInf = json.load(file_json)
        qp.add_subsection(FigInf['title'])
        rfpath_pics = rpath_pics+FigInf['name']
        qp.add_fig(rfpath_pics)
      else:
        print('::: Warning: file does not exist: %s!:::' %fpath)
        #raise ValueError('::: Error: file does not exist: %s!:::' %fpath)
  
  # --- execute qp_link_all to add all residual figs that can be found
  qp.add_section('More plots')
  for fpath in flist_all:
    print(fpath)
    with open(fpath) as file_json:
      FigInf = json.load(file_json)
    qp.add_subsection(FigInf['title'])
    rfpath_pics = rpath_pics+FigInf['name']
    qp.add_fig(rfpath_pics)
  
  qp.write_to_file()

  # --- execute qp_link_all to add link of this time average
  print("Executing qp_link_all.py")
  #os.system(f"python {path_qp_driver}qp_link_all.py {path_qp_sim}")
  pyicqp.link_all(path_quickplots=path_quickplots, path_search=path_qp_sim, do_conf_dwd=do_conf_dwd)


# --- add link for this simulation
print("Executing qp_link_all.py")
#os.system(f"python {path_qp_driver}qp_link_all.py")
pyicqp.link_all(path_quickplots=path_quickplots, do_conf_dwd=do_conf_dwd)

# --- add page for additional information
pyicqp.add_info(run=run, path_data=path_data, path_qp_sim=path_qp_sim)

### --------------------------------------------------------------------------------
### show figures
### --------------------------------------------------------------------------------
##if fig_names.size<3:
##  plt.show()
print('All done!')
