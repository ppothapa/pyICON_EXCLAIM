# User guide for pyicon

Pyicon is a python post-processing and visualization toolbox for ICON with a focus on ocean data. The three main features of pyicon are:

* a number of functions to facilitate the every-day script-based plotting of ICON data
* an interactive (ncview-like) plotting GUI for Jupyter notebook
* a monitoring suite for ICON ocean simulations which combines dedicated diagnostic plots of an ICON simulation on a website

Pyicon is developed within the DFG-project TRR181 - Energy Transfers in Atmosphere and Ocean.

The pyicon documentation can be found here: [documentation](https://m300602.gitlab-pages.dkrz.de/pyicon/)

Pyicon is hosted at: (https://gitlab.dkrz.de/m300602/pyicon/)

## Quick start for pyicon on Mistral

Once you have to download pyicon by git:

```bash
git clone git@gitlab.dkrz.de:m300602/pyicon.git
```

After that you have to load the correct python environment and make sure that pyicon is in your search path each time you want to use it. 
The easiest way is to use the following script:

```bash
source /path/to/pyicon/tools/conda_act_mistral_pyicon_env.sh
```

## Quick start for pyicon @DWD (Confluence, only intern)
https://ninjoservices.dwd.de/wiki/display/KUQ/pyICON+for+ICON+with+NWP+physics

## Installing locally

You can also install `pyicon` locally via `pip`. However, due to dependencies of `cartopy` it is advised to install `cartopy` first via `conda`.

```bash
conda install xarray cartopy dask -c conda-forge
```

Once, `cartopy` is installed in your environment:

```bash
pip install git+https://gitlab.dkrz.de/m300602/pyicon.git
```
