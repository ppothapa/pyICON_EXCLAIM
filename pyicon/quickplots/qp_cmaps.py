from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

def plot_examples(cmap):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), constrained_layout=True)
    if True:
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

class PyicCmaps(object):
  def __init__(self):
    #self.add_cmap(name='WhiteBlueGreenYellowRed')
    self.path_cmaps = './cmaps/'
    flist = glob.glob(self.path_cmaps+'*.rgb')
    for fpath in flist:
      name = fpath.split('/')[-1].split('.')[0]
      print(name)
      self.add_cmap(name=name)  
    return

  def add_cmap(self, name='WhiteBlueGreenYellowRed'):
    f = open(self.path_cmaps+name+'.rgb', 'r')
    txt = f.read()
    f.close()
    
    txt = txt.split('\n')
    txt = txt[2:-1]
    clist = np.zeros((3,len(txt)))
    for nn, line in enumerate(txt):
      col = line[2:15].split('  ')
      clist[0, nn] = col[0]
      clist[1, nn] = col[1]
      clist[2, nn] = col[2]
    clist = clist.transpose()/255.
    #print(clist[1:-1,:])
    newcmp = ListedColormap(clist[1:-1,:], name=name)
    newcmp.set_under(clist[0,:])
    newcmp.set_over(clist[-1,:])
    setattr(self, name, newcmp)
    return

  def show_example(self, name):
    newcmp = getattr(self, name)

    plt.close('all')
    plot_examples(newcmp)
    return

