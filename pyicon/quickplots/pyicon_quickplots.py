import sys, glob, os
import json
# --- calculations
import numpy as np
# --- reading data 
from netCDF4 import Dataset, num2date
import datetime
# --- plotting
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
#import my_toolbox as my
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmocean
# --- debugging
from ipdb import set_trace as mybreak  
import pyicon as pyic

# ================================================================================ 
# Quick Plots
# ================================================================================ 

# --------------------------------------------------------------------------------
# Horizontal plots
# --------------------------------------------------------------------------------
def qp_hplot(fpath, var, IcD='none', depth=-1e33, iz=0, it=0,
              t1='needs_to_be_specified', t2='none',
              it_ave=[],
              rgrid_name="orig",
              path_ckdtree="",
              var_fac = 1.,
              var_add = 0.,
              clim='auto', cincr=-1., cmap='auto',
              clevs=None,
              contfs=None,
              conts=None,
              xlim=[-180,180], ylim=[-90,90], projection='none',
              use_tgrid=False,
              crs_features=True,
              adjust_axlims=False,
              asp=0.5,
              title='auto', units='',
              xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              do_plot_settings=True,
              land_facecolor='0.7',
              do_mask=False,
              do_write_data_range=True,
              ):

  #for fp in [fpath]:
  #  if not os.path.exists(fp):
  #    raise ValueError('::: Error: Cannot find file %s! :::' % (fp))


  # --- set-up grid and region if not given to function
  if isinstance(IcD,str) and IcD=='none':
    # get fname and path_data from fpath
    fname = fpath.split('/')[-1]
    path_data = ''
    for el in fpath.split('/')[1:-1]:
      path_data += '/'
      path_data += el
    path_data += '/'

    IcD = IconData(
                   fname   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   rgrid_name   = rgrid_name,
                   omit_last_file = False,
                  )
  #else:
  #  print('Using given IcD!')

  if depth!=-1e33:
    iz = np.argmin((IcD.depthc-depth)**2)
  IaV = IcD.vars[var]
  step_snap = it

  # --- seems to be necessary for RUBY
  if IaV.coordinates=='':
    IaV.coordinates = 'clat clon'

  # --- load data 
  #IaV.load_hsnap(fpath=IcD.flist_ts[step_snap], 
  #                    it=IcD.its[step_snap], 
  #                    iz=iz,
  #                    step_snap = step_snap
  #                   ) 
  IaV.time_average(IcD, t1, t2, it_ave, iz=iz)

  # --- mask data
  if do_mask:
    IaV.data = np.ma.array(IaV.data)
    if IaV.coordinates=='clat clon':
      IaV.data[IcD.wet_c[iz,:]==0] = np.ma.masked
    elif IaV.coordinates=='elat elon':
      IaV.data[IcD.wet_e[iz,:]==0] = np.ma.masked
    else:
      raise ValueError("::: Error: Unknownc coordinates for mask!:::")

  # --- interpolate data 
  if not use_tgrid:
    IaV.interp_to_rectgrid(fpath_ckdtree=IcD.rgrid_fpath)
  # --- crop data

  IaV.data *= var_fac
  IaV.data += var_add

  if units!='':
    IaV.units = units

  # --- cartopy projection
  if projection=='none':
    ccrs_proj = None
  else:
    ccrs_proj = getattr(ccrs, projection)()

  # --- do plotting
  (ax, cax, 
   mappable,
   Dstr
  ) = pyic.hplot_base(
              IcD, IaV, 
              ax=ax, cax=cax,
              clim=clim, cmap=cmap, cincr=cincr,
              clevs=clevs,
              contfs=contfs,
              conts=conts,
              xlim=xlim, ylim=ylim,
              adjust_axlims=adjust_axlims,
              title=title, 
              projection=projection,
              crs_features=crs_features,
              use_tgrid=use_tgrid,
              logplot=logplot,
              asp=asp,
              do_plot_settings=do_plot_settings,
              land_facecolor=land_facecolor,
              do_write_data_range=do_write_data_range,
             )

  # --- contour labels
  if contfs=='auto':
    Cl = ax.clabel(mappable, colors='k', fontsize=6, fmt='%.1f', inline=False)
    for txt in Cl:
      txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))


  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

def qp_vplot(fpath, var, IcD='none', it=0,
              t1='needs_to_be_specified', t2='none',
              it_ave=[],
              sec_name="specify_sec_name",
              path_ckdtree="",
              var_fac=1.,
              var_add=0.,
              clim='auto', cincr=-1., cmap='auto',
              clevs=None,
              contfs='auto',
              conts='auto',
              xlim=[-90,90], ylim=[6000,0], projection='none',
              asp=0.5,
              title='auto', xlabel='', ylabel='',
              verbose=1,
              ax='auto', cax='auto',
              logplot=False,
              log2vax=False,
              mode_load='normal',
              do_plot_settings=True,
              do_write_data_range=True,
              ):

  #for fp in [fpath]:
  #  if not os.path.exists(fp):
  #    raise ValueError('::: Error: Cannot find file %s! :::' % (fp))

  # --- load data set
  if isinstance(IcD,str) and IcD=='none':
    # get fname and path_data from fpath
    fname = fpath.split('/')[-1]
    path_data = ''
    for el in fpath.split('/')[1:-1]:
      path_data += '/'
      path_data += el
    path_data += '/'

    IcD = IconData(
                   fname   = fname,
                   path_data    = path_data,
                   path_ckdtree = path_ckdtree,
                   #rgrid_name   = rgrid_name
                   omit_last_file = False,
                  )
  #else:
  #  print('Using given IcD!')

  IaV = IcD.vars[var]
  step_snap = it

  # --- seems to be necessary for RUBY
  if IaV.coordinates=='':
    IaV.coordinates = 'clat clon'

  # --- load data
  # FIXME: MOC and ZAVE cases could go into load_vsnap
  if sec_name.endswith('moc'):
    #IaV.load_moc(
    #               fpath=IcD.flist_ts[step_snap], 
    #               it=IcD.its[step_snap], 
    #               step_snap = step_snap
    #              ) 
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    IaV.data = IaV.data[:,:,0]/1e9 # MOC in nc-file as dim (nt,nz,ny,ndummy=1)
    f = Dataset(IcD.flist_ts[0], 'r')
    IaV.lat_sec = f.variables['lat'][:]
    IaV.depth = f.variables['depth'][:]
    f.close()
    IaV.mask = IaV.data==0.
    IaV.data[IaV.mask] = np.ma.masked
  elif sec_name.startswith('zave'):
    basin      = sec_name.split(':')[1]
    rgrid_name = sec_name.split(':')[2]
    #IaV.lat_sec, IaV.data = pyic.zonal_average(
    #                               fpath_data=IcD.flist_ts[step_snap], 
    #                               var=var, basin=basin, it=it,
    #                               fpath_fx=IcD.fpath_fx, 
    #                               fpath_ckdtree=IcD.rgrid_fpaths[
    #                                 np.where(IcD.rgrid_names==rgrid_name)[0][0]]
    #                                     )
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    IaV.lat_sec, IaV.data = pyic.zonal_average_3d_data(
                                   IaV.data, 
                                   basin=basin, coordinates=IaV.coordinates,
                                   fpath_fx=IcD.fpath_fx,
                                   fpath_ckdtree=IcD.rgrid_fpaths[
                                     np.where(IcD.rgrid_names==rgrid_name)[0][0]],
                                                      )
  else:
    sec_fpath = IcD.sec_fpaths[np.where(IcD.sec_names==sec_name)[0][0] ]
    #IaV.load_vsnap(
    #               fpath=IcD.flist_ts[step_snap], 
    #               fpath_ckdtree=sec_fpath,
    #               it=IcD.its[step_snap], 
    #               step_snap = step_snap
    #              ) 
    IaV.time_average(IcD, t1, t2, it_ave, iz='all')
    # --- interpolate data 
    if not IcD.use_tgrid:
      IaV.interp_to_section(fpath_ckdtree=sec_fpath)

    IaV.data *= var_fac
    IaV.data += var_add

  # --- do plotting
  (ax, cax, 
   mappable,
   Dstr
  ) = pyic.vplot_base(
                 IcD, IaV, 
                 ax=ax, cax=cax,
                 clim=clim, cmap=cmap, cincr=cincr,
                 clevs=clevs,
                 contfs=contfs,
                 conts=conts,
                 title=title, 
                 log2vax=log2vax,
                 logplot=logplot,
                 do_plot_settings=do_plot_settings,
                 do_write_data_range=do_write_data_range,
                )

  # --- contour labels
  if contfs=='auto':
    Cl = ax.clabel(mappable, colors='k', fontsize=6, fmt='%.1f', inline=False)
    for txt in Cl:
      txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0))

  # ---
  ax.set_xlim(xlim)
  ax.set_ylim(ylim)

  # --- output
  FigInf = dict()
  FigInf['fpath'] = fpath
  FigInf['long_name'] = IaV.long_name
  #FigInf['IcD'] = IcD
  return FigInf

def qp_timeseries(IcD, fpath, vars_plot, 
                  var_fac=1., var_add=0.,
                  title='', units='',
                  t1='none', t2='none',
                  ave_freq=0,
                  omit_last_file=True,
                  mode_ave=['mean'],
                  labels=None,
                  do_write_data_range=True,
                 ): 
  if len(mode_ave)==1:
    mode_ave = [mode_ave[0]]*len(vars_plot)
    dfigb = 0.0
  else:
    do_write_data_range = False
    dfigb = 0.7
  flist = glob.glob(IcD.path_data+fpath)
  flist.sort()
  if omit_last_file:
    flist = flist[:-1]
  times, flist_ts, its = pyic.get_timesteps(flist)
  if ave_freq>0:
    nskip = times.size%ave_freq
    if nskip>0:
      times = times[:-nskip]
    nresh = int(times.size/ave_freq)
    times = np.reshape(times, (nresh, ave_freq)).transpose()
    #times_ave = times.mean(axis=0)
    times = times[int(ave_freq/2),:] # get middle of ave_freq

  hca, hcb = pyic.arrange_axes(1,1, plot_cb=False, asp=0.5, fig_size_fac=2.,
               sharex=True, sharey=True, xlabel="time [years]", ylabel="", 
               dfigb=dfigb,
               )
  ii=-1
  ii+=1; ax=hca[ii]; cax=hcb[ii]

  for mm, var in enumerate(vars_plot):
    data = np.array([])
    for nn, fpath in enumerate(flist):
      f = Dataset(fpath, 'r')
      data_file = f.variables[var][:,0,0]
      data = np.concatenate((data, data_file))
      f.close()
    #print(f'{var}: {data.size}')
    # --- apply time averaging
    if ave_freq>0:
      if nskip>0:
        data = data[:-nskip]
      #if times.size%ave_freq != 0:
      #  raise ValueError(f'::: Time series has wrong size: {times.size} for ave_req={ave_freq}! :::')
      print(f'{var}: {data.size} {times.size}')
      data = np.reshape(data, (nresh, ave_freq)).transpose()
      if mode_ave[mm]=='mean':
        data = data.mean(axis=0)
      elif mode_ave[mm]=='min':
        data = data.min(axis=0)
      elif mode_ave[mm]=='max':
        data = data.max(axis=0)

    if labels is None:
      label = var
    else:
      label = labels[mm]
    data *= var_fac
    data += var_add
    ax.plot(times, data, label=label)
  ax.grid(True)
  if len(vars_plot)==1:
    f = Dataset(fpath, 'r')
    if units=='':
      units = f.variables[var].units
      units = f' [{units}]'
    if title=='':
      long_name = f.variables[var].long_name
      title = long_name+units
    f.close()
  ax.set_title(title)
  if len(vars_plot)>1:
    ax.legend()

  if not (isinstance(t1,str) and t1=='none'):
    ax.axvline(t1, color='k')
  if not (isinstance(t2,str) and t2=='none'):
    ax.axvline(t2, color='k')

  if do_write_data_range:
    info_str = 'min: %.2g;        mean: %.2g;        std: %.2g;        max: %.2g' % (data.min(), data.mean(), data.std(), data.max())
    ax.text(0.5, -0.18, info_str, ha='center', va='top', transform=ax.transAxes)

  FigInf = dict()
  return FigInf

##def qp_hor_plot(fpath, var, IC='none', iz=0, it=0,
##              grid='orig', 
##              path_rgrid="",
##              clim='auto', cincr='auto', cmap='auto',
##              xlim=[-180,180], ylim=[-90,90], projection='none',
##              title='auto', xlabel='', ylabel='',
##              verbose=1,
##              ax='auto', cax=1,
##              ):
##
##
##  # --- load data
##  fi = Dataset(fpath, 'r')
##  data = fi.variables[var][it,iz,:]
##  if verbose>0:
##    print('Plotting variable: %s: %s' % (var, IC.long_name)) 
##
##  # --- set-up grid and region if not given to function
##  if isinstance(IC,str) and clim=='none':
##    pass
##  else:
##    IC = IconDataFile(fpath, path_grid='/pool/data/ICON/oes/input/r0003/')
##    IC.identify_grid()
##    IC.load_tripolar_grid()
##    IC.data = data
##    if grid=='orig':
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      IC.Tri = matplotlib.tri.Triangulation(IC.vlon, IC.vlat, 
##                                            triangles=IC.vertex_of_cell)
##      IC.mask_big_triangles()
##      use_tgrid = True
##    else: 
##      # --- rectangular grid
##      if not os.path.exists(path_rgrid+grid):
##        raise ValueError('::: Error: Cannot find grid file %s! :::' % 
##          (path_rgrid+grid))
##      ddnpz = np.load(path_rgrid+grid)
##      IC.lon, IC.lat = ddnpz['lon'], ddnpz['lat']
##      IC.Lon, IC.Lat = np.meshgrid(IC.lon, IC.lat)
##      IC.data = icon_to_regular_grid(IC.data, IC.Lon.shape, 
##                          distances=ddnpz['dckdtree'], inds=ddnpz['ickdtree'])
##      IC.data[IC.data==0] = np.ma.masked
##      IC.crop_grid(lon_reg=xlim, lat_reg=ylim, grid=grid)
##      use_tgrid = False
##  IC.data = IC.data[IC.ind_reg]
##
##  IC.long_name = fi.variables[var].long_name
##  IC.units = fi.variables[var].units
##  IC.name = var
##
##  ax, cax, mappable = hplot_base(IC, var, clim=clim, title=title, 
##    projection=projection, use_tgrid=use_tgrid)
##
##  fi.close()
##
##  # --- output
##  FigInf = dict()
##  FigInf['fpath'] = fpath
##  FigInf['long_name'] = long_name
##  FigInf['IC'] = IC
##  #ipdb.set_trace()
##  return FigInf

# ================================================================================ 
# ================================================================================ 
class QuickPlotWebsite(object):
  """ Creates a website where the quick plots can be shown.

Minimal example:

# --- setup
qp = QuickPlotWebsite(
  title='pyicon Quick Plot', 
  author='Nils Brueggemann', 
  date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  fpath_css='./mycss2.css'
  )

# --- content
for i in range(1,11):
  qp.add_section('Section %d' % i)
  for j in range(1,4):
    qp.add_subsection('Subsection %d.%d' % (i,j))
    qp.add_paragraph(('').join(['asdf %d'%(i)]*10))
    qp.add_fig('./pics/','fig_01.png')
qp.write_to_file()
  """

  def __init__(self, title='Quick Plot', author='', date='', 
               info='', path_data='',
               fpath_css='', fpath_html='./qp_index.html'):
    self.author = author 
    self.title = title
    self.date = date
    self.info = info
    self.path_data = path_data
    self.fpath_css = fpath_css
    self.fpath_html = fpath_html

    self.first_add_section_call = True

    self.main = ""
    self.toc = ""
    if 'timeaverages' in title:
      links = """
&emsp; <a href="../index.html">list simulations</a>
"""
    elif 'simulations' in title:
      links = ""
    else:
      links = """
&emsp; <a href="../qp_index.html">list time averages</a>
&emsp; <a href="../../index.html">list simulations</a>
"""

    self.header = """
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">

<head>
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta http-equiv="Content-Style-Type" content="text/css" />
  <meta name="generator" content="pyicon" />
  <meta name="author" content="{author}" />
  <title>{title}</title>
  <style type="text/css">code{{white-space: pre;}}</style>
  <link rel="stylesheet" href="{fpath_css}" type="text/css" />
</head>

<body>

<div id="header">
<h1 class="title">{title}</h1>
<p> {author} | {date} | {path_data} </p>
<p> {info} 
{links}
</p>
</div>

""".format(author=self.author, title=self.title, 
           date=self.date, 
           path_data=self.path_data,
           info=self.info,
           fpath_css=self.fpath_css, 
           links=links,
          )

#<div id="header">
#<h1 class="title">{title}</h1>
#<h2 class="author">{author}</h2>
#<h3 class="date">{date}</h3>
#</div>

    self.footer = """
</body>
</html>
"""
  
  def add_section(self, title='Section'):
    # --- add to main
    href = title.replace(' ', '-')
    self.main += '\n'
    #self.main += f"<h1 id=\"{href}\">{title}</h1>\n"
    self.main += f"  <div id=\"ctn\">"
    self.main += f"    <a name=\"{href}\">&nbsp;</a>"
    self.main += f"    <h1 class=\"target-label\">{title}</h2>"
    self.main += f"  </div>"
    # --- add to toc
    if self.first_add_section_call:
      self.first_add_section_call = False
      self.toc += """
<div id="TOC">
<ul>
"""
    else:
      self.toc += '</ul></li> \n'
    self.toc += f'<li><a href="#{href}">{title}</a><ul>\n'
    return

  def add_subsection(self, title='Subsection'):
    # --- add to main
    href = title.replace(' ', '-')
    #self.main += f"  <h2 id=\"{href}\">{title}</h2>\n"
    self.main += f"  <div id=\"ctn\">"
    self.main += f"    <a name=\"{href}\">&nbsp;</a>"
    self.main += f"    <h2 class=\"target-label\">{title}</h2>"
    self.main += f"  </div>"
    # --- add to toc
    self.toc += f'<li><a href="#{href}">{title}</a></li>\n'
    return

  def add_paragraph(self, text=''):
    self.main += '    <p>'
    self.main += text
    self.main += '    </p>'
    self.main += '\n'
    return
   
  def add_fig(self, fpath, width="1000"):
    self.main += f'    <div class="figure"> <img src="{fpath}" width="{width}" /> </div>'
    self.main += '\n'
    return
  
  def close_toc(self):
    # --- close toc
    self.toc += """</ul></li>
</ul>
</div>

"""
    return

  def write_to_file(self):
    # --- close toc
    self.close_toc()

    # --- write to output file
    f = open(self.fpath_html, 'w')
    f.write(self.header)
    f.write(self.toc)
    f.write(self.main)
    f.write(self.footer)
    f.close()
    return

# ================================================================================ 
# ================================================================================ 
