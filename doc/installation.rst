Installation of pyicon
----------------------

Pyicon depends on a number of python libraries typically used for geoscience data evaluation and plotting. 
The easiest installation is probably by using conda (see below). 
However, on the DKRZ computer cluster Mistral one make use of pre-installed python libraries.

Mistral environment
^^^^^^^^^^^^^^^^^^^

The easiest way to use pyicon on Mistral is to use a pre-installed environment.
In this case, you do not need to read further but you can simply type::

  source pyicon/tools/conda_act_mistral_pyicon_env.sh

If this was successful, you can jump to `Installation`_ below.

Note: The script ``conda_act_mistral_pyicon_env.sh`` is expanding the python path and normal path and activating a suitable conda python3 environment::

.. code-block::

 export PYTHONPATH="/home/mpim/m300602/python/pytbx/mypy:/home/mpim/m300602/pyicon"
 export PATH="/home/mpim/m300602/miniconda2/bin:$PATH"
 source activate myenv_py3


Requirements
^^^^^^^^^^^^

pyicon is developed for python 3.7. Other versions might work as well but are not supported so far.
Furthermore, the following modules are required:

  * numpy, scipy (calculations)
  * matplotlib, cartopy (plotting)
  * netcdf4 (reading netcdf data)
  * ipython, jupyter (for pyicon_view)

A suitable python environment is probably easiest set up by using conda::

  conda env create -f pyicon_env.yml 

with the following yml-file (assumed to be named pyicon_env.yml)::

  name: pyicon_env
  channels:
    - conda-forge
    - defaults
  dependencies:
    - python=3.7
    - numpy
    - scipy
    - netcdf4
    - matplotlib
    - cartopy
    - ipython     # for pyicon_view
    - jupyter     # for pyicon_view
    - jupyterlab  # optional
    - cmocean     # optional
    - mpi4py      # not used by pyicon so far
    - seawater    # not used by pyicon so far
    - xarray      # not used by pyicon so far
    - dask        # not used by pyicon so far

Installation
^^^^^^^^^^^^

Download pyicon by::
  
  git clone git@gitlab.dkrz.de:m300602/pyicon.git

.. So far, the following is not supported yet::

..  cd pyicon
..  python setup.py install

and add pyicon to your PYTHONPATH either by::
  
  export PYTHONPATH="/path/to/pyicon:${PYTHONPATH}"

or by adding the following lines at the beginning of each of your python scripts where you want to import pyicon::
  
  import sys
  sys.path.insert(0,'/path/to/pyicon')
