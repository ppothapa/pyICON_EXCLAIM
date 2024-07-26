# --- path to quickplots
path_quickplots = '../all_qps/'

# --- set this to True if the simulation is still running
omit_last_file = True

# --- do ocean and/or atmosphere plots
do_atmosphere_plots = True
do_conf_dwd         = True
do_djf              = False
do_jja              = False
do_ocean_plots      = False

# --- grid information
gname     = 'R2B4-R2B4'
lev       = 'L40'
gname_atm = 'r2b4_atm_r0013'
lev_atm   = 'L90'

# --- path to interpolation files
#  Here is the DWD Path
#path_grid        = '/hpc/uwork/icon-sml/pyICON/grids/'+gname+'/'
#path_grid_atm    = '/hpc/uwork/icon-sml/pyICON/grids/'+gname_atm+'/'
#path_ckdtree     = path_grid+'/ckdtree/'
#path_ckdtree_atm = path_grid_atm+'/ckdtree/'
#End of DWD Path

# PPK Comment, as far as I can see, these are all Ocean files- and is difficult to understand them.


path_grid        = '/project/d121/ppothapa/data_for_pyicon/r2b6_oce_r0004/'
path_grid_atm    = '/project/d121/ppothapa/data_for_pyicon/r2b4_atm_r0013/'
path_ckdtree     = '/project/d121/ppothapa/data_for_pyicon/r2b6_oce_r0004/ckdtree/'
path_ckdtree_atm = '/project/d121/ppothapa/data_for_pyicon/r2b4_atm_r0013/ckdtree/'



# --- grid files and reference data
#path_pool_oce       = '/hpc/uwork/icon-sml/pyICON/grids/'
path_pool_oce       = '/pool/data/ICON/grids/'
gnameu = gname.split('_')[0].upper()
fpath_tgrid                 = '/scratch/snx3000/ppothapa/for_praveen/0013/icon_grid_0013_R02B04_G.nc'
fpath_tgrid_atm             = '/scratch/snx3000/ppothapa/for_praveen/0013/icon_grid_0013_R02B04_G.nc'
fpath_ref_data_oce          = path_grid + 'ts_phc3.0_annual_icon_grid_0043_R02B04_G_L40.nc'
#fpath_ref_data_atm          = path_grid_atm + 'era5_pyicon_2001-2010_1.5x1.5deg.nc'
#fpath_ref_data_atm          = '/project/d121/ppothapa/data_for_pyicon/era/pyicon_prepare_era.nc'

fpath_ref_data_atm          = '/scratch/snx3000/ppothapa/Era5_python_output/old/era5_pyicon_2004-2006_1.0x1.0deg_slice.nc'

fpath_ref_data_atm_djf      = path_grid_atm + 'era5_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_jja      = path_grid_atm + 'era5_pyicon_2001-2010_jja_1.5x1.5deg.nc'
#fpath_ref_data_atm_rad      = path_grid_atm + 'ceres_pyicon_2001-2010_1.5x1.5deg.nc'
fpath_ref_data_atm_rad_djf  = path_grid_atm + 'ceres_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_rad_jja  = path_grid_atm + 'ceres_pyicon_2001-2010_jja_1.5x1.5deg.nc'
#fpath_ref_data_atm_prec     = path_grid_atm + 'gpm_pyicon_2001-2010_1.5x1.5deg.nc'
fpath_ref_data_atm_prec_djf = path_grid_atm + 'gpm_pyicon_2001-2010_djf_1.5x1.5deg.nc'
fpath_ref_data_atm_prec_jja = path_grid_atm + 'gpm_pyicon_2001-2010_jja_1.5x1.5deg.nc'
fpath_fx                    = path_grid + 'oce_fx.19600102T000000Z.nc'

# --- nc file prefixes ocean
oce_def     = '_oce_def'
oce_moc     = '_oce_moc'
oce_mon     = '_oce_mon'
oce_ice     = '_oce_ice'
oce_monthly = '_oce_dbg'

# --- nc file prefixes atmosphere
atm_2d      = '_atm_2d_ml'
atm_3d      = '_atm_3d_ml'
atm_mon     = '_atm_mon'

# --- nc output
save_data = False
path_nc = '/scratch/m/m300602/tmp/test_pyicon_output/'

# --- time average information (can be overwritten by qp_driver call)
tave_ints = [
#['1630-02-01', '1640-01-01'],
['2004-04-15', '2006-02-15'],
]

ave_freq = 12

#output_freq = 'daily'


# --- decide if time-series (ts) plots are plotted for all the 
#     available data or only for the intervall defined by tave_int

use_tave_int_for_ts = False

# --- what to plot and what not?
# --- not to plot:
#red_list = ['tas_gmean','rsdt_gmean','rsut_gmean','rlut_gmean','radtop_gmean','prec_gmean','evap_gmean','pme_gmean']
red_list = ['atm_u_250_bias','atm_v_250_bias']
# --- to plot:
#green_list = ['atm_t2m','atm_t2m_bias','atm_tauu', 'atm_tauu_bias', 'atm_tauv','atm_tauv_bias', 'atm_curl_tau', 'atm_surf_shfl', 'atm_toa_netrad', 'atm_surf_lhfl', 'atm_toa_sob', 'atm_toa_sou', 'atm_toa_thb','sea_ts', 'seaice_fraction','atm_psl','atm_w10m','atm_ts','atm_sfc_sob', 'atm_sfc_thb','atm_cwv','atm_tcc', 'atm_tp','atm_pme','atm_u_250','atm_v_250', 'atm_geop_500','atm_relhum_700','atm_temp_850','atm_temp_zave','atm_logv_temp_zave','atm_u_zave','atm_logv_u_zave','atm_v_zave','atm_logv_v_zave','atm_relhum_zave','atm_cc_zave','atm_clw_zave','atm_cli_zave', 'atm_clwi_zave', 'tas_gmean','rsdt_gmean','rsut_gmean','rlut_gmean','radtop_gmean','prec_gmean','evap_gmean','pme_gmean']
