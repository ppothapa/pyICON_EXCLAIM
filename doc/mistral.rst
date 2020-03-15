Mistral specific
----------------

pyicon compatible python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A pyicon compatible python environment can be loaded by::

  source pyicon/tools/conda_act_mistral_pyicon_env.sh

Switch to pyicon compatible python environment

Start Jupyter session for Mistral
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use Jupyter on the DKRZ cluster you find valuable information `here <https://www.dkrz.de/up/systems/mistral/programming/jupyter-notebook>`_.

DKRZ also provides a `script <https://gitlab.dkrz.de/k202009/ssh_scripts/raw/master/start-jupyter?inline=false>`_ to easily setup a Jupyter session.

Download this script and use it with::

  ./start-jupyter -u username -i /home/mpim/m300602/bin/myactcondenv.sh

JupyterLab session can be started with ``-c lab`` option::

  ./start-jupyter -u username -c lab -i /home/mpim/m300602/bin/myactcondenv.sh

Grid files on Mistral
^^^^^^^^^^^^^^^^^^^^^

At the moment, a collection of grid files and ckdtrees for different ICON grids can be found here::

  /mnt/lustre01/work/mh0033/m300602/icon/grids 

In general, grid files will be archived here::

  /pool/data/ICON/oes/input/r0003/

An even larger collection of grids can be found here::

  /pool/data/ICON/oes/grids/

ICON SMT grids are in::

  /pool/data/ICON/oes/grids/OceanOnly

Parallel computing with python on Mistral
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
