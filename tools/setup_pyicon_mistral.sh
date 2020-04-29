#!/bin/bash

# ... needs to be tested ...

# create a conda environment to run pyicon, including interactive plots in jupyterlab

# use anaconda module on mistral
module load anaconda3/bleeding_edge

# create environment from yml file
conda env create -f ../ci/pyicon_env.yml

# activate the environment to install pyicon package and labextensions
source activate	pyicon_env

# install pyicon package
conda-develop ../pyicon

# install labextension that enables interactive plots in jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

### optional: rename lab to show custom tab name, which is useful if you have a local and remote jupyterlab running
#jupyter lab build --name='JupyterLab pyicon MISTRAL'
