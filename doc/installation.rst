Installation of pyicon
======================

Mistral environment
-------------------

Log into Mistral and go to a directory where you want to install pyicon (this can be in your home directory or on work).
Download pyicon by::
  
  git clone https://gitlab.dkrz.de/m300602/pyicon.git

The easiest way to use pyicon on Mistral is to use a pre-installed environment. 
Therefore execute the following command (do not forget to change ``/path/to/pyicon/`` to the path where you downloaded pyicon)::

  source /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh

Note: The script ``conda_act_mistral_pyicon_env.sh`` is expanding the python path to contain the pyicon module. 
It also expands the normal path variable that enables the use of conda. 
Finally, a suitable python3 environment is activated by conda.

If everything was successful, you should now be ready to use pyicon and you can skipt further installation instructions.

Requirements
------------

pyicon is developed for python 3.8. Other versions might work as well but are not supported so far.
Furthermore, the following modules are required as a minimum:

  * numpy, scipy (calculations)
  * matplotlib, cartopy (plotting)
  * netcdf4, xarray (reading / writing netcdf data)
  * ipython, jupyter (for running python)
  * mpi4py, dask (for parallel computing and distributed memory)

A suitable python environment is probably easiest set up by using conda (for an installation of conda see below)::

  conda env create -f pyicon/ci/requirements_py38.yml

You can eaily activate your conda environment by using the script::

  pyicon/tools/conda_act_mistral_pyicon_env.sh

In this script modify the path of your conda installation in the line starting with ```path_conda_bin=```. After doing this, you can activate your conda environment by executing::

  source pyicon/tools/conda_act_mistral_pyicon_env.sh

Install conda
-------------

A suitable python environment can be easily created by using conda. Therefore, we roughly outline here how to install conad but we refer to more detailed documentations wwhich can be found in the web. We are explaining the installation of conda using miniconda but other conda versions should work as well. 

Go to::

  https://docs.conda.io/en/latest/miniconda.html

Pick an installer for Python 3.8 and download the file (e.g. right click on the link, "copy link", go to shell type wget and paste the link).

After downloading the script change the rights (chmod 755 <script>) and execute it (./<script>).

If conda is installed successfully, you should see something like '(base)' infront of your shell line.

General installation instructions
---------------------------------

Download pyicon by::
  
  git clone https://gitlab.dkrz.de/m300602/pyicon.git

.. So far, the following is not supported yet::

..  cd pyicon
..  python setup.py install

and add pyicon to your PYTHONPATH either by (do not forget to change ``/path/to/pyicon`` to the path where you downloaded pyicon)::
  
  export PYTHONPATH="/path/to/pyicon:${PYTHONPATH}"

or by adding the following lines at the beginning of each of your python scripts where you want to import pyicon (do not forget to change ``/path/to/pyicon``)::
  
  import sys
  sys.path.insert(0,'/path/to/pyicon')

Pyicon depends on a number of python libraries typically used for geoscience data evaluation and plotting. 
The easiest installation is probably by using conda. A suitable python environment for pyicon can be installed using conda from this file::

  pyicon/ci/requirements_py38.yml

