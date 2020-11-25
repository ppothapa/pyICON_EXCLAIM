Mistral specific
================

.. pyicon compatible python environment
.. ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. 
.. A pyicon compatible python environment can be loaded by::
.. 
..   source /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh
.. 
.. Switch to such a pyicon compatible python environment in every shell session where you want to execute python scripts that use pyicon.

Start Jupyter session on Mistral
--------------------------------

Jupyter without DKRZ software
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One easy way to start a Jupyter session on mistral is like follows:

1. Connect by ssh to mistralpp using port forwording use a <port> which is 8000<port<8887:

  ::

    ssh -L 8888:localhost:<port> <username>@mistralpp.dkrz.de

2. Activate an appropriate python environment and start Jupyter using the above <port>:

  ::
    
    source pyicon/tools/conda_act_mistral_pyicon_env.sh
    jupyter notebook --no-browser --port=<port>

3. Finally open you web browser on you local computer and enter the address:
  
  ::

    localhost:8888

Jupyter with DKRZ software
^^^^^^^^^^^^^^^^^^^^^^^^^^

To use Jupyter on the DKRZ cluster, you can find valuable information `here <https://www.dkrz.de/up/systems/mistral/programming/jupyter-notebook>`_.

DKRZ also provides a `script <https://gitlab.dkrz.de/k202009/ssh_scripts/raw/master/start-jupyter?inline=false>`_ to easily setup a Jupyter session.

Download this script and use it with::

  ./start-jupyter -u username -i /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh

JupyterLab session can be started with ``-c lab`` option (note that pyicon_view does not work with JupyterLab)::

  ./start-jupyter -u username -c lab -i /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh

Grid files on Mistral
---------------------

At the moment, a collection of grid files and ckdtrees for different ICON grids can be found here::

  /mnt/lustre01/work/mh0033/m300602/icon/grids 

In general, ocean triangular grid files are archived here::

  /pool/data/ICON/oes/input/

and atmospheric grid files are archived here::
 
 /pool/data/ICON/grids/public/mpim/ 

An even larger collection of ocean grids can be found here::

  /pool/data/ICON/oes/grids/

E.g. ICON SMT grids can be found in::

  /pool/data/ICON/oes/grids/OceanOnly

Parallel computing with python on Mistral
-----------------------------------------

By using the ``mpi4py`` module, it is possible to run tasks of a script in parallel.
To make use of simple parallelization e.g. for creating animations, a python script needs to be modified in the following way (!todo! improve example)::

  # header of python script 
  import sys
  import matplotlib
  if len(sys.argv)>1 and sys.argv[1]=='--no_backend':
    print('apply: matplotlib.use(\'Agg\')')
    matplotlib.use('Agg')

  ...

  # === mpi4py ===
  try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    npro = comm.Get_size()
  except:
    print('::: Warning: Proceeding without mpi4py! :::')
    rank = 0
    npro = 1
  print('proc %d/%d: Hello world!' % (rank, npro))

  # ajust `steps` index list to only contain reduced number of indices
  list_all_pros = [0]*npro
  for nn in range(npro):
    list_all_pros[nn] = steps[nn::npro]
  steps = list_all_pros[rank]

Such a python script (assuming the name ``parallel_animation.py``) can be run in parallel using the following slurm-script:

.. code-block:: bash

  #! /bin/bash
  #SBATCH --job-name=pyhur
  #SBATCH --time=02:00:00
  #SBATCH --output=log.o-%j.out
  #SBATCH --error=log.o-%j.out
  #SBATCH --ntasks=4
  #SBATCH --partition=compute,compute2
  #SBATCH --account=mh0033
  set -x
  
  #module load python/2.7.12
  module list
  
  source /home/mpim/m300602/bin/myactcondenv.sh
  which python
  
  startdate=`date +%Y-%m-%d\ %H:%M:%S`
  mpirun -np 4 python -u parallel_animation.py --no_backend
  enddate=`date +%Y-%m-%d\ %H:%M:%S`
  echo "Started at ${startdate}"
  echo "Ended at   ${enddate}"
