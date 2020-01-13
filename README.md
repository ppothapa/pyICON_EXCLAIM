# User guide for pyicon

Pyicon is a python post-processing and visualization toolbox for ICON with a focus on ocean data. The three main features of pyicon are:

* a number of functions to facilitate the every-day script-based plotting of ICON data
* an interactive (ncview-like) plotting GUI for Jupyter notebook
* a monitoring suite ICON ocean simulations which combines all plots of an ICON simulation in a website

Pyicon is developed within the DFG-project TRR181 - Energy Transfers in Atmosphere and Ocean.

## Install pyicon

Pyicon has depends on frequently used python libraries in the geoscience business. They are easiest installed using conda

```
conda env create -f pyicon_env.yml
```

with the following yml-file (assumed to be named pyicon_env.yml):

~~~
name: pyicon_env
channels:
  - conda-forge
  - defaults
dependencies:
#  - python=2.7 # use this line to get python2.7 env
  - numpy
  - matplotlib
  - scipy
  - ipython
  - jupyter
  - jupyterlab  # optional
  - netcdf4
  - mpi4py      # optional
  - cartopy
  - cmocean
  - seawater
  - xarray
  - dask
~~~

Get pyicon by copying all files on mistral from:

```bash
git clone git@gitlab.dkrz.de:m300602/pyicon.git
```

Add pyicon to your PYTHONPATH either by setting:

```
export PYTHONPATH="/path/to/pyicon:${PYTHONPATH}"
```

or by adding it within your python script:

```
import sys
sys.path.insert(0,'/path/to/pyicon')
```

## Known bugs:

* there are always problems with vertical grid variables loaded from the ICON fx-files.

## Use pyicon with Jupyter

To use Jupyter on the DKRZ cluster you find valuable information here:

```
https://www.dkrz.de/up/systems/mistral/programming/jupyter-notebook
```

Easiest way is to download this script:

```
https://gitlab.dkrz.de/k202009/ssh_scripts/raw/master/start-jupyter?inline=false
```

and execute it (!!! not working!!!):

```
./start-jupyter -u username -i incfile.txt
```

with incfile.txt containing:

```
export PYTHONPATH="\${HOME}/python/pytbx/mypy"
export PATH="/home/mpim/m300602/miniconda2/bin:\$PATH"
source activate myenv_py3
```

!!! Needs to be updated!!!

Open jupyter notebook with ssh tunnel:

```bash
ssh -L 8000:localhost:15768 m300602@mistralpp.dkrz.de 'source /sw/rhel6-x64/etc/profile.mistral \
     && /home/mpim/m300602/miniconda2/envs/myenv/bin/jupyter notebook --no-browser --port 15768'
```

Try out example notebook
```
pyic_test_r2b9.ipynb
```

## Use pyicon with normal python scripts

!!! Needs to be updated!!!

Example script:

```python
import pyicon_jupyter as jup
import pyicon as pyic  
import numpy as np
import matplotlib.pyplot as plt

# create date set
path_data = '/mnt/lustre01/work/mh0033/m211054/projects/icon/solver/icon-oes-solver-HB/experiments/ocean_benchmark/outdata/'
IcD = pyic.IconData(
    fpath_grid_triangular  = path_data+'ocean_benchmark_20161011T000000Z.nc',
    #region                 = 'hurricane',
    fname_rgrid            = 'r2b9ocean_r0.05_100w_30w_2n_40n.npz',
    #fname_rgrid            = 'r2b9ocean_r0.3_180w_180e_90s_90n.npz',
    path_data              = path_data,
    search_str             = 'ocean_benchmark_????????????????.nc',
       )

# load data
IcD.load_hsnap(['temp', 'salt'], step_snap=step_snap, it=0, iz=iz)

# make axes
fig = plt.figure()
ax = fig.add_subplot(111)
cax = 0

# make plot
IP = pyic.IP_hor_sec_rect(                                                       
  IcD, ax=ax, cax=cax,
  var='temp', clim=[0, 10], nc='auto', cmap='viridis',                                 
  transform=None,                                                  
  edgecolor=None,                                                           
  )
```
