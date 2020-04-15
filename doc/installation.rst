Installation of pyicon
----------------------

Mistral environment
^^^^^^^^^^^^^^^^^^^

Log into Mistral and go to a directory where you want to install pyicon (this can be in your home directory or on work).
Download pyicon by::
  
  git clone git@gitlab.dkrz.de:m300602/pyicon.git

The easiest way to use pyicon on Mistral is to use a pre-installed environment. 
Therefore execute the following command (do not forget to change ``/path/to/pyicon/`` to the path where you downloaded pyicon)::

  source /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh

Note: The script ``conda_act_mistral_pyicon_env.sh`` is expanding the python path to contain the pyicon module. 
It also expands the normal path variable that enables the use of conda. 
Finally, a suitable python3 environment is activated by conda.

If everything was successful, you should now be ready to use pyicon and you can skipt further installation instructions.

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

General installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Download pyicon by::
  
  git clone git@gitlab.dkrz.de:m300602/pyicon.git

.. So far, the following is not supported yet::

..  cd pyicon
..  python setup.py install

and add pyicon to your PYTHONPATH either by (do not forget to change ``/path/to/pyicon`` to the path where you downloaded pyicon)::
  
  export PYTHONPATH="/path/to/pyicon:${PYTHONPATH}"

or by adding the following lines at the beginning of each of your python scripts where you want to import pyicon (do not forget to change ``/path/to/pyicon``)::
  
  import sys
  sys.path.insert(0,'/path/to/pyicon')

Pyicon depends on a number of python libraries typically used for geoscience data evaluation and plotting. 
The easiest installation is probably by using conda (see below). 
On the DKRZ Mistral computer cluster, it is possible to make use of pre-installed python libraries.

