import sys, glob, os
import matplotlib.pyplot as plt
#import my_toolbox as my
import numpy as np
import cartopy.crs as ccrs
import cmocean
from netCDF4 import Dataset
import datetime
import time

import ipywidgets as widgets
from ipywidgets import interact, interactive, HBox, VBox
from IPython.display import display

import pyicon as pyic
reload(pyic)

#def my_slide(update_func, Dparas):
def my_slide(name='slider:', bnds=[0,10]):
  w1 = widgets.IntSlider(
    value=0,
    min=bnds[0],
    max=bnds[1],
    step=1,
    description=name,
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
  )
  
  b_inc = widgets.Button(
    description='+1',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='inc',
    icon=''
  )
  
  b_dec = widgets.Button(
    description='-1',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='dec',
    icon=''
  )
  
  def b_inc_update(b):
    val = w1.value
    val += 1
    if val > w1.max:
          val = w1.min   
    w1.value = val
      
  def b_dec_update(b):
    val = w1.value
    val -= 1
    if val < w1.min:
        val = w1.max
    w1.value = val
  
  b_inc.on_click(b_inc_update)
  b_dec.on_click(b_dec_update)
  #Box = HBox([b_dec, w1, b_inc])
  #display(Box)
  return b_dec, w1, b_inc

def updater(update_func, **Dparas):
  a = interact(update_func, **Dparas)
  return a
 
def hplot(IcD, var='', clim=[-1,1], edgecolor='none'):

  ##def stop_animation(w):
  ##  IP.continue_anim = False
  ##  return

  ##def play_animation(w):
  ##  w.iit += 1
  ##  print w.iit, bstop.value
  ##  time.sleep(1)
  ##  if w.iit < 10:
  ##    play_animation(w)
  ##  print 'final = ', bstop.value

  ##  #IcD.step_snap += 1
  ##  #IcD.load_hsnap([IP.var], step_snap=IcD.step_snap, it=IcD.it, iz=IcD.iz)
  ##  #IP.update(getattr(IcD, IP.var), IcD)
  ##  #IP.fig.canvas.draw()
  ##  #if IP.continue_anim==True:
  ##  #  time.sleep(0.5)
  ##  #  play_animation(w) 
  ##  #else:
  ##  #  print('stopping animation')
  ##  return

  def update_fig(var='', iz=0, step_snap=0):
    #print var, iz, step_snap
    IcD.load_hsnap([var], step_snap=step_snap, it=0, iz=iz)
    IP.update(getattr(IcD, var), IcD, 
              title=IcD.long_name[var]+' ['+IcD.units[var]+']')
    IP.var = var
    display(ax.figure)
    return

  def update_clim(w):
    climstr = w.owner.value
    #print climstr
    if climstr!='auto':
      clim = np.array(climstr.split(',')).astype(float)  
      if clim.size==1:
        clim = np.array([-clim[0], clim[0]])
    try:
      #print clim
      IP.hpc[0].set_clim(clim)
    except:
      print('Could not convert %s into clim.' % (climstr))
    return 

  def auto_clim(b1):
    #print IP.var
    min_val = getattr(IcD, IP.var).min()  
    max_val = getattr(IcD, IP.var).max()  
    climstr = '%.2g, %.2g' % (min_val, max_val)
    t1.value = climstr
    return

  def update_cmap(w):
    cmap = w.owner.value
    if cmap.startswith('cmo'):
      cmap = cmap.split('.')[-1]
      cmap = getattr(cmocean.cm, cmap)
      #exec('cmap = %s' % (cmap))
    IP.hpc[0].set_cmap(cmap) 
    return

  def update_reg(w):
    print w.owner.value
    print("Switching regions interactively is not yet supported!")
    #if reg=='global':
    #  IcD.fname_rgrid = 'r2b9ocean_r0.3_180w_180e_90s_90n.npz'
    #elif reg=='hurricane':
    #  IcD.fname_rgrid = 'r2b9ocean_r0.05_100w_30w_2n_40n.npz'
    #else:
    #  print('::: Region %s is not supported! Taking \'global\'. :::' %(reg)) 
    #  IcD.fname_rgrid = 'r2b9ocean_r0.3_180w_180e_90s_90n.npz'

    #IcD.load_grid()
    #plt.close('all')
    #IP, IcD = initialize_plot(IcD, var='to', clim=1.5) 
    #return IP, IcD
    return

  def initialize_plot(IcD, var='', clim=1.5):
    # --- load data
    IcD.load_hsnap([var], step_snap=0, it=0, iz=0)

    # --- make initial plot
    hca, hcb = pyic.arrange_axes(1,1, plot_cb=True, sasp=0.543, fig_size_fac=2.5,
                               sharex=False, sharey=False, xlabel="", ylabel="",
                               projection=ccrs.PlateCarree(),
                              )
    ii=-1

    ii+=1; ax=hca[ii]; cax=hcb[ii]
    IP = pyic.IP_hor_sec_rect(
      IcD, ax=ax, cax=cax,
      var=var, clim=clim, nc='auto', cmap='viridis',
      transform=ccrs.PlateCarree(),
      edgecolor=edgecolor,
      )
    IP.fpath_save = './test.pdf'
    IP.continue_anim = True
    IP.fig = plt.gcf()
    return IP, IcD

  def update_fpath_save(w):
    IP.fpath_save = w.owner.value
    return

  def save_fig(w):
    plt.savefig(IP.fpath_save)
    print('Saving figure %s' % (IP.fpath_save))
    return

  # --- initialize plot
  IP, IcD = initialize_plot(IcD, var=IcD.varnames[0], clim=clim)

  # --- make depth slider
  b_dec, w1, b_inc = my_slide(name='depth:', bnds=[0,IcD.depth.size-1])
  Box = HBox([b_dec, w1, b_inc])
  display(Box)

  # --- make time slider
  b_dec, w2, b_inc = my_slide(name='time:', bnds=[0,IcD.times.size-1])
  Box = HBox([b_dec, w2, b_inc])
  display(Box)

  ## --- play button
  #bplay = widgets.Button(
  #  description='play',
  #  disabled=False,
  #  button_style='', # 'success', 'info', 'warning', 'danger' or ''
  #  tooltip='dec',
  #  icon=''
  #)
  #bplay.iit = 0
  #bplay.on_click(play_animation)

  ## --- stop button
  ##bstop = widgets.Button(
  ##  description='stop',
  ##  disabled=False,
  ##  button_style='', # 'success', 'info', 'warning', 'danger' or ''
  ##  tooltip='dec',
  ##  iyicon=''
  ##)
  ##bstop.on_click(stop_animation)

  #bstop = widgets.Checkbox(
  #    value=False,
  #    description='Stop',
  #    disabled=False
  #)

  #Box = HBox([b_dec, w2, b_inc, bplay, bstop])
  #display(Box)
  
  # --- make clim widget
  t1 = widgets.Text(
      value='-100, 100',
      placeholder='-100, 100',
      description='clim =',
      disabled=False
  )
  t1.continuous_update=False
  t1.observe(update_clim, names='value')

  # --- clim auto button
  b1 = widgets.Button(
    description='auto',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='dec',
    icon=''
  )
  #b1.var = IP.var
  b1.on_click(auto_clim)

  # --- make cmap widget
  d1 = widgets.Dropdown(
    options=['viridis', 'plasma', 'RdBu_r', 'RdYlBu_r', 'cmo.thermal', 'cmo.haline', 'cmo.ice', 'cmo.dense', 'cmo.curl', 'cmo.delta'],
    value='viridis',
    description='cmap:',
    disabled=False,
              )
  d1.observe(update_cmap, names='value')

  # --- make varname widget
  d2 = widgets.Dropdown(
    options=IcD.varnames,
    value=IcD.varnames[0],
    description='var:',
    disabled=False,
              )

  # --- make region widget
  d3 = widgets.Dropdown(
    options=['global', 'hurricane'],
    value='global',
    description='region:',
    disabled=False,
              )
  d3.observe(update_reg, names='value')

  Box = HBox([d2, t1, b1, d1, d3])
  display(Box)

  # --- make save textbox
  ts = widgets.Text(
      value='./test.pdf',
      placeholder='./test.pdf',
      description='Name:',
      disabled=False
  )
  ts.continuous_update=False
  ts.observe(update_fpath_save, names='value')

  # --- save button
  bs = widgets.Button(
    description='save',
    disabled=False,
    button_style='', # 'success', 'info', 'warning', 'danger' or ''
    tooltip='dec',
    icon=''
  )
  #b1.var = IP.var
  bs.on_click(save_fig)

  Box = HBox([ts, bs])
  display(Box)

  #print(type(w1))
  #a = updater(update_fig, iz=w1, step_snap=w2)
  #a = interact(update_fig, IcD=IcD, IP=IP, iz=w1, step_snap=w2)
  a = interactive(update_fig, var=d2, iz=w1, step_snap=w2)
  #b = interact(update_clim, climstr=t1)
  #c = interact(update_cmap, cmap=d1)
  #d = interact(update_reg, reg=d3)
  return IcD, IP
