import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import sys
import pyicon as pyic
import cartopy
import cartopy.crs as ccrs 

#class MyObject(object):
#  def __init__(self):
#    return

def arctic_budgets(IcD, IcD_ice, IcD_dbg, t1, t2, to, so, mass_flux, uo, vo):
#if True:
  # --- specify domain
  ireg = ( 
        ((IcD.clat>65.0) & (IcD.clon>=-180.) &(IcD.clon<-50.)) # west
      | ((IcD.clat>78.0) & (IcD.clon>=-60.) &(IcD.clon<15))    # Greenland - Spitzbergen
  
      | ((IcD.clat>66.) & (IcD.clon>=15) & (IcD.clon<68))      # Spitzbergen - Norway
      | ((IcD.clat>65.0) & (IcD.clon>=68) &(IcD.clon<180.))      # north east
         )
  ic_ireg = np.where(ireg)[0]
  
  # --- derive outer section of domain
  iedge = IcD.edge_of_cell[ireg,:]
  iedge = iedge.reshape(iedge.size)
  oedge = IcD.orientation_of_normal[ireg,:]
  oedge = oedge.reshape(iedge.size)
  iedge_out, ind, cnts = np.unique(iedge, return_index=True, return_counts=True)
  iedge_out = iedge_out[(cnts==1)]
  oedge_out = oedge[ind]
  oedge_out = oedge_out[(cnts==1)]
  
  # --- split outer section in single sections
  Diedge_out = dict()
  Doedge_out = dict()
  Dcol = dict()
  Dvec = dict()
  
  name = 'Fram_Strait'
  ind = (IcD.elat[iedge_out]<78.5) & (IcD.elon[iedge_out]>-30) & (IcD.elon[iedge_out]<=14.)
  Diedge_out[name] = iedge_out[ind]
  Doedge_out[name] = oedge_out[ind]
  Dcol[name] = 'g'
  Dvec[name] = [[-15,70.5],[0,83.5]]
  name = 'Barents_Sea'
  # ind = (IcD.elat[iedge_out]<77.5) & (IcD.elon[iedge_out]>15) & (IcD.elon[iedge_out]<=90.)
  ind = (IcD.elat[iedge_out]>=67) & (IcD.elat[iedge_out]<78) & (IcD.elon[iedge_out]>=14.) & (IcD.elon[iedge_out]<16)
  Diedge_out[name] = iedge_out[ind]
  Doedge_out[name] = oedge_out[ind]
  Dcol[name] = 'b'
  # Dvec[name] = [[52.5,69.5],[52.5,83.5]]
  Dvec[name] = [[4,65.],[31,75.]]
  name = 'Davis_Strait'
  ind = (IcD.elat[iedge_out]<65.5) & (IcD.elon[iedge_out]>-70) & (IcD.elon[iedge_out]<=-44.)
  Diedge_out[name] = iedge_out[ind]
  Doedge_out[name] = oedge_out[ind]
  Dcol[name] = 'y'
  Dvec[name] = [[-57.,57.5],[-57.,71.5]]
  name = 'Hudson_Bay'
  ind = (IcD.elat[iedge_out]<67.0) & (IcD.elon[iedge_out]>-90) & (IcD.elon[iedge_out]<=-70.)
  Diedge_out[name] = iedge_out[ind]
  Doedge_out[name] = oedge_out[ind]
  Dcol[name] = 'm'
  Dvec[name] = [[-80.,59],[-80.,73]]
  name = 'Bering_Strait'
  ind = (IcD.elat[iedge_out]<65.5) & (IcD.elon[iedge_out]<170) & (IcD.elon[iedge_out]<=-160.)
  Diedge_out[name] = iedge_out[ind]
  Doedge_out[name] = oedge_out[ind]
  Dcol[name] = 'orange'
  Dvec[name] = [[-175.,57.5],[-175.,71.5]]
  #return iedge_out, oedge_out, Diedge_out, Doedge_out, Dcol, Dvec

  # --- load data
  # --- 3d fields
#  mass_flux, it_ave = pyic.time_average(IcD, 'mass_flux', t1=t1, t2=t2, iz='all')
#  uo, it_ave        = pyic.time_average(IcD, 'u', t1=t1, t2=t2, iz='all')
#  vo, it_ave        = pyic.time_average(IcD, 'v', t1=t1, t2=t2, iz='all')
#  to, it_ave        = pyic.time_average(IcD, 'to', t1=t1, t2=t2, iz='all')
#  so, it_ave        = pyic.time_average(IcD, 'so', t1=t1, t2=t2, iz='all')
  # ------ 2d fields
  #zo, it_ave        = pyic.time_average(IcD, 'zos', t1=t1, t2=t2, iz='all')
  hi, it_ave        = pyic.time_average(IcD_ice, 'hi', t1=t1, t2=t2, iz='all')
  hs, it_ave        = pyic.time_average(IcD_ice, 'hs', t1=t1, t2=t2, iz='all')
  conc, it_ave      = pyic.time_average(IcD_ice, 'conc', t1=t1, t2=t2, iz='all')
  ice_u, it_ave     = pyic.time_average(IcD_ice, 'ice_u', t1=t1, t2=t2, iz='all')
  ice_v, it_ave     = pyic.time_average(IcD_ice, 'ice_v', t1=t1, t2=t2, iz='all')
  #zUnderIce, it_ave = pyic.time_average(IcD_dbg, 'zUnderIce', t1=t1, t2=t2, iz='all')
  
  #mass_flux = mass_flux.astype(np.float64)
  #uo = uo.astype(np.float64)
  #vo = vo.astype(np.float64)
  #wo = wo.astype(np.float64)
  #to = to.astype(np.float64)
  #so = so.astype(np.float64)
  #zo = zo.astype(np.float64)
  
  HeatFlux_Total, it_ave         = pyic.time_average(IcD_dbg, 'HeatFlux_Total', t1=t1, t2=t2, iz='all')
  #FrshFlux_TotalOcean, it_ave    = pyic.time_average(IcD_dbg, 'FrshFlux_TotalOcean', t1=t1, t2=t2, iz='all')
  FrshFlux_Runoff, it_ave = pyic.time_average(IcD_dbg, 'FrshFlux_Runoff', t1=t1, t2=t2, iz='all')
  FrshFlux_Evaporation, it_ave   = pyic.time_average(IcD_dbg, 'FrshFlux_Evaporation', t1=t1, t2=t2, iz='all')
  FrshFlux_Precipitation, it_ave = pyic.time_average(IcD_dbg, 'FrshFlux_Precipitation', t1=t1, t2=t2, iz='all')
  #FrshFlux_SnowFall, it_ave      = pyic.time_average(IcD_dbg, 'FrshFlux_SnowFall', t1=t1, t2=t2, iz='all')
  FrshFlux_SnowFall, it_ave      = pyic.time_average(IcD_dbg, 'totalsnowfall', t1=t1, t2=t2, iz='all')
  FrshFlux_TotalOcean            = FrshFlux_Runoff + FrshFlux_Precipitation + FrshFlux_Evaporation #+ FrshFlux_SnowFall
  
  HeatFlux_Total *= -1
  FrshFlux_Runoff *= -1
  FrshFlux_Evaporation *= -1
  FrshFlux_Precipitation *= -1
  FrshFlux_SnowFall *= -1
  FrshFlux_TotalOcean *= -1
  
  # --- vertical integrl of mass flux
  mass_flux_vint = mass_flux.sum(axis=0)

  draftave = ( (IcD.rhoi/IcD.rho0*hi + IcD.rhos/IcD.rho0*hs) * conc ).sum(axis=0)
  dz = 1.*IcD.dzw
  #dz[0,:] += zo
  dz[0,:] += -draftave
  
  # --- derive fluxes on cell center
  sref = 34.8
  
  ut = IcD.cp*IcD.rho0*uo*(to)
  vt = IcD.cp*IcD.rho0*vo*(to)
  us = uo*(sref-so)/sref
  vs = vo*(sref-so)/sref
  
  uice_dz = (hi*IcD.rhoi/IcD.rho0*conc*(sref-5.)/sref*ice_u[np.newaxis,:]).sum(axis=0)
  vice_dz = (hi*IcD.rhoi/IcD.rho0*conc*(sref-5.)/sref*ice_v[np.newaxis,:]).sum(axis=0)
  usno_dz = (hs*IcD.rhos/IcD.rho0*conc*(sref-0.)/sref*ice_u[np.newaxis,:]).sum(axis=0)
  vsno_dz = (hs*IcD.rhos/IcD.rho0*conc*(sref-0.)/sref*ice_v[np.newaxis,:]).sum(axis=0)
  
  uo_dz = (uo*dz)
  vo_dz = (vo*dz)
  ut_dz = (ut*dz)
  vt_dz = (vt*dz)
  us_dz = (us*dz)
  vs_dz = (vs*dz)
  
  us_dz[0,:] += usno_dz + uice_dz
  vs_dz[0,:] += vsno_dz + vice_dz
  
  usol_dz = draftave*ice_u
  vsol_dz = draftave*ice_v
  
  uo_dz[0,:] += usol_dz
  vo_dz[0,:] += vsol_dz
  
  # --- map fluxes to cell edges
  print('map fluxes')
  IcD.edge2cell_coeff_cc_t = pyic.calc_edge2cell_coeff_cc_t(IcD)
  
  # calculate 3d p-array
  p_vo_3d = pyic.calc_3d_from_2dlocal(IcD, uo_dz, vo_dz)
  p_vt_3d = pyic.calc_3d_from_2dlocal(IcD, ut_dz, vt_dz)
  p_vs_3d = pyic.calc_3d_from_2dlocal(IcD, us_dz, vs_dz)
  p_vsol_3d = pyic.calc_3d_from_2dlocal(IcD, usol_dz[np.newaxis,:], vsol_dz[np.newaxis,:])
  # calculate edge array
  ptp_vo_e = pyic.cell2edges(IcD, p_vo_3d*IcD.wet_c[:,:,np.newaxis])*IcD.wet_e
  ptp_vt_e = pyic.cell2edges(IcD, p_vt_3d*IcD.wet_c[:,:,np.newaxis])*IcD.wet_e
  ptp_vs_e = pyic.cell2edges(IcD, p_vs_3d*IcD.wet_c[:,:,np.newaxis])*IcD.wet_e
  ptp_vsol_e = pyic.cell2edges(IcD, p_vsol_3d*IcD.wet_c[0:1,:,np.newaxis])*IcD.wet_e[0:1,:]
  
  print('derive section/area fluxes')

  # --- derive lateral fluxes
  DTmass = dict()
  DTmsol = dict()
  DTheat = dict()
  DTfrwa = dict()
  for name in Diedge_out.keys():
      iedge_out = Diedge_out[name]
      oedge_out = Doedge_out[name]
      DTmass[name] = (   mass_flux_vint[iedge_out]
                       * IcD.edge_length[iedge_out]
                       * oedge_out ).sum()/1e6
      DTmsol[name] = (   ptp_vsol_e[:,iedge_out]
                       * IcD.edge_length[np.newaxis,iedge_out]
                       * oedge_out[np.newaxis,:] ).sum()/1e6
      DTheat[name] = (   ptp_vt_e[:,iedge_out]
                       * IcD.edge_length[np.newaxis,iedge_out]
                       * oedge_out[np.newaxis,:] ).sum()/1e12
      DTfrwa[name] = (   ptp_vs_e[:,iedge_out]
                       * IcD.edge_length[np.newaxis,iedge_out]
                       * oedge_out[np.newaxis,:] ).sum()/1e6
  
  # --- derive domain integrated surface fluxes
  T_heat_surf = (HeatFlux_Total[ireg]     *IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e12
  T_frwa_surf = (FrshFlux_TotalOcean[ireg]*IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e6
  #T_frwa_ruof = (FrshFlux_Runoff[ireg]*IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e6
  #T_frwa_evap = (FrshFlux_Evaporation[ireg]*IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e6
  #T_frwa_prec = (FrshFlux_Precipitation[ireg]*IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e6
  #T_frwa_snow = (FrshFlux_SnowFall[ireg]*IcD.cell_area[ireg]*IcD.wet_c[0,ireg]).sum()/1e6
  DTmass['surface'] = T_frwa_surf
  DTmsol['surface'] = 0.0
  DTfrwa['surface'] = T_frwa_surf
  DTheat['surface'] = T_heat_surf
  
  # --- derive residual fluxes as sum of all other fluxes
  tmp1, tmp2, tmp3, tmp4 = 0., 0., 0., 0.
  for name in DTmass.keys():
      tmp1 += DTmass[name]
      tmp2 += DTmsol[name]
      tmp3 += DTfrwa[name]
      tmp4 += DTheat[name]
  DTmass['residuum'] = tmp1
  DTmsol['residuum'] = tmp2
  DTfrwa['residuum'] = tmp3
  DTheat['residuum'] = tmp4

  print('start plotting')
  unit = 'mSv'
  fac = 1e3
  # unit = 'km^3/year'
  # fac = -86400*365/1000**3 * 1e6
  # unit = 'kg/s'
  # fac = -1/1000 * 1e6
  prec = '.1f'
  
  # --- NorthPolarStereo projection
  ccrs_proj = ccrs.NorthPolarStereo()
  hca, hcb = pyic.arrange_axes(2,2, asp=1.0, projection=ccrs_proj, fig_size_fac=2,
                               sharex=True, sharey=True, plot_cb=False)
  ii=-1

  fpath_ckdtree = IcD.rgrid_fpath
  lon, lat, iregi = pyic.interp_to_rectgrid(ireg, fpath_ckdtree=fpath_ckdtree)
  
  for kk in range(len(hca)):
      ii+=1; ax=hca[ii]; cax=hcb[ii]
      pyic.shade(lon, lat, iregi, ax=ax, cax=cax, clim=[0,2], projection=ccrs.PlateCarree())
      pyic.plot_settings(ax=ax, xlim=[-180,180], ylim=[54,90], 
                         do_xyticks=False, do_xyminorticks=False, do_gridlines=True, 
                         land_facecolor='0.7',
                        )
      ax.scatter(IcD.elon[iedge_out], IcD.elat[iedge_out], s=2, c='r', transform=ccrs.PlateCarree())
  
      for nn, name in enumerate(DTmass.keys()):
          if kk==0:
              val = DTmass[name]*fac
              ax.set_title(f'mass transport [{unit}]')
          elif kk==1:
              val = DTmsol[name]*fac
              ax.set_title(f'solid mass transport [{unit}]')
          elif kk==2:
              val = DTfrwa[name]*fac
              ax.set_title(f'fresh water transport [{unit}]')
          elif kk==3:
              val = DTheat[name]
              ax.set_title(f'heat transport [TW]')

          if (name!='surface') & (name!='residuum'):
              if val>0:
                  text = f'{val:{prec}}'
                  arrowstyle = "<-"
              else:
                  text = f'{-val:{prec}}'
                  arrowstyle = "->"
              ax.scatter(IcD.elon[Diedge_out[name]], IcD.elat[Diedge_out[name]], s=2, c=Dcol[name], transform=ccrs.PlateCarree())
              ax.annotate(text, 
                          xy=(Dvec[name][1]), xytext=(Dvec[name][0]),
                          ha='center', va='center',
                          xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                          arrowprops=dict(arrowstyle=arrowstyle, fc='r', ec='r', linewidth=2), 
                          bbox=dict(fc='w', ec='none'),
                          zorder=10)
          elif (name=='surface'):
              if val>0:
                  text = f'surface:\n$\\bigodot$ {val:{prec}}'
              else:
                  text = f'surface:\n$\\bigotimes$ {-val:{prec}}'
              ax.text(0, 90, text,
                      bbox=dict(fc='w', ec='none'),
                      transform=ccrs.PlateCarree(),
                      ha='center', va='center',
                     )
          else:
              if val>0:
                  text = f'$\\bigodot$ {val:{prec}}'
              else:
                  text = f'$\\bigotimes$ {-val:{prec}}'
              ax.text(140, 65, 'residuum:\n'+text,
                      bbox=dict(fc='w', ec='none'),
                      transform=ccrs.PlateCarree(),
                      ha='left', va='top',
                     )
  return

#if False:
if __name__ == "__main__":
  #import numpy as np
  #import matplotlib.pyplot as plt
  #from netCDF4 import Dataset
  #import sys
  #import pyicon as pyic
  #import cartopy
  #import cartopy.crs as ccrs 

  runname   = ''
  run       = 'hel20134-STR'
  gname     = 'r2b6'
  lev       = 'L64'
  
  # t1 = '1880-02-01'
  # t2 = '1890-01-01'
  t1 = '1501-01-01'
  t2 = '1502-01-01'
  
  path_data     = '/work/mh0033/m211054/projects/icon/ruby/icon-oes_fluxadjust/experiments/'+run+'/'
  path_grid     = f'/mnt/lustre01/work/mh0033/m300602/icon/grids/{gname}/'
  path_ckdtree  = f'{path_grid}ckdtree/'
  fpath_ckdtree = f'{path_grid}ckdtree/rectgrids/{gname}_res0.30_180W-180E_90S-90N.npz'
  # fpath_fx      = '/mnt/lustre01/work/mh0033/m211054/projects/icon/ruby/icon-oes_fluxadjust/experiments/exp.ocean_omip_long_tke_r2b6_20133-CIQ/exp.ocean_omip_long_tke_r2b6_20133-CIQ_fx.nc'
  # fpath_tgrid   = '/work/mh0033/m211054/projects/icon/topo/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_srtm30_1min/OceanOnly_Global_IcosSymmetric_0039km_rotatedZ37d_BlackSea_Greenland_modified_sills_srtm30_1min.nc'
  fpath_tgrid   = 'auto'
  fpath_fx      = 'auto'
  
  fname_def = run+'_oce_def_????????????????.nc'
  fname_moc = run+'_oce_moc_????????????????.nc'
  fname_dbg = run+'_oce_dbg_????????????????.nc'
  fname_ice = run+'_oce_ice_????????????????.nc' 
  
  IcD = pyic.IconData(
                 fname        = fname_def,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 fpath_fx     = fpath_fx,
                 fpath_tgrid  = fpath_tgrid,
                 do_triangulation = True,
                 omit_last_file   = False,
                 verbose = True,
                )
  
  
  print('dbg')
  IcD_dbg = pyic.IconData(
                 fname        = fname_dbg,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 fpath_fx     = fpath_fx,
                 fpath_tgrid  = fpath_tgrid,
                 do_triangulation   = False,
                 omit_last_file     = False,
                 load_vertical_grid = False,
                 load_triangular_grid = False,
                 calc_coeff         = False,
                 verbose = True,
                )
  
  print('ice')
  IcD_ice = pyic.IconData(
                 fname        = fname_ice,
                 path_data    = path_data,
                 path_grid    = path_grid,
                 gname        = gname,
                 lev          = lev,
                 fpath_fx     = fpath_fx,
                 fpath_tgrid  = fpath_tgrid,
                 do_triangulation   = False,
                 omit_last_file     = False,
                 load_vertical_grid = False,
                 load_triangular_grid = False,
                 calc_coeff         = False,
                 verbose = True,
                )


  plt.close('all')
  arctic_budgets(IcD, IcD_ice, IcD_dbg)
  plt.show()


