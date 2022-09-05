.. _pyicon-core:

pyicon-core
===========

In this section, the most important features of ``pyicon`` should be documented. 
However, we would like to mention that this documentation is not yet (and maybe will never) be exhaustive. 
Instead we would like to refer to the many example notebooks where the most important use cases are introduced.

Common workflow using ``pyicon``
--------------------------------

The common workflow using ``pyicon`` typically consists of the following 5 steps:

  1. Creating an IconData object
  2. Loading the data
  3. Doing some calculations 
  4. Interpolating the data
  5. Plotting the data
  6. (Optional) Saving the data to netcdf

It is important to note that it is very well possible to use both,  Jupyter Notebooks or simple python scripts.

Example for a typical use case
------------------------------

To be continued...

Creating an IconData object
---------------------------

One of the central element of ``pyicon`` is the IconData object. 
This object is a container with very different sorts of information, like e.g. path names to the data or the grid, grid variables, a list of files of the data set, or many coefficients, like e.g. Coriolis parameter or divergence coefficients.
What type of information is stored in it very much depends on how it is initialized.

A standard initialization looks like this::
  
 IcD = pyic.IconData(
    fname        = 'nib0001_????????????????.nc',
    path_data    = path_data,
    path_grid    = path_grid,
    gname        = gname,
    lev          = lev,
    )

``fname`` is a string containing wild cards to identify the set of netcdf files containing the same variables.
Therefore, it is important not to use something like ``fname = 'nib0001*.nc`` since it might contain e.g. the files ``nib00001_????????????????.nc`` but also e.g. ``nib00001_atm2d_????????????????.nc``.
Here it best, to write out the full file name and use a ``?`` for each number of the time string.

``path_data`` is the path to the netcdf files.

``path_grid`` is the path to the rectangular and tripolar grid files.
The following directory structure is assumed: ``path_data/ckdtree/rectgrids/*.npz``.
This is also the standard directory structure used by the ``config_cktree`` scripts (see below).
When the variables ``fpath_tgrid`` and ``fpath_fx`` are not specified, ``pyicon`` assumes that as well the triangular grid file and -- for ocean data -- the fx-file containing the vertical grid are under this directory.
The name convention is: ``f'{gname}_tgrid.nc'`` and ``f'{gname}_{lev}_fx.nc'``.
If these files do not exist, either specify ``fpath_tgrid`` and ``fpath_fx`` or link these files into this directory.

``gname`` and ``lev`` are the short identifying names for the horizontal and the vertical grid. 
Typical names are: ``gname = 'r2b6_oce_r0004'`` and ``lev = 'L64'``.

Some of the information stored in an IconData object can be expensive to derive and take a lot of memory -- in particular for large grids.
Therefore, it is possible to decide what information should be derived when the object is initialized.
A more light wise IconData object can be loaded by specifying the following options::

  do_triangulation      = True,    # if False matplotlib triangulation object is not derived
  load_vertical_grid    = True,    # if False the fx file is not used and no vertical grid is loaded
  load_vgrid_depth      = 'auto',  # if False the depth is not loaded
  load_vgrid_dz         = 'auto',  # if False the level thicknesses are not loaded
  load_vgrid_mask       = 'auto',  # if False no mask is loaded
  load_triangular_grid  = True,    # if False the tipolar grid (clat, clon) is not loaded
  load_rectangular_grid = True,    # if False the rectangular grid for the interpolation is not loaded (it is usually loaded later on anyway)
  load_variable_info    = True,    # if False no information about which variables are storred in file (usually does not saves lots of time)
  load_grid_from_data_file = False,# set only to True if cdo interpolated data is used instead of pure ICON data
  calc_coeff            = True,    # derive coefficients for divergence, curl, etc. can take a long time and needs lots of memory
  calc_coeff_mappings   = False,   # derive coefficients for reconstructions e.g. between edges and center, can take a long time and needs lots of memory
  do_only_timesteps     = False,   # short cut: if True only time step list and variables names are loaded, can be considered as minimalistic IconData initialization

Other important parameters are::

  omit_last_file        = False,   # last file of output file list is omitted, set to True to avoid potential data damage when open netcdf files are read when the simulation is still running
  time_mode             = 'num2dat e', # ususally 'num2date' should be fine however older simulations sometimes need 'float2date'
  model_type            = 'oce',   # choose 'oce' for ocean data or 'atm' for atmospheric data
  output_freq           = 'auto',  # specify e.g. 'yearly', 'monthly', depending on the output frequency of the data, needed for pyic.time_average to give appropriate number to each month / year. Usually autmotic determination works fine, problems can arise when only one time step is in a file.
  verbose               = False,   # writ out some informations, good to infer which steps take up most of the time
  dtype                 = 'float32', # if double precissioin is needed change to 'float64'

Loading the data
----------------

We provide three different possibilities for reading the data.
One is ``pyicon`` intrinsic (pyic.time_average) and in most cases the preferred option.
Sometimes it can also be useful to use the ``netCDF4`` library directly to read the data, in particular, when no averaging is needed, the data set is very large and by no means any unnecessary steps should be done.
While pyic.time_average has also the possibility to load the data as ``xarray`` and chunked by ``dask``, it might be necessary to use ``xarray`` directly to read the data.

Using pyic.time_average
^^^^^^^^^^^^^^^^^^^^^^^

The ``pyicon`` function pyic.time_average can be used to load data and directly average over the data.
It is even possible to load a single snap shot.
This makes this function to a generic tool for loading the data.
After defining an IconData object, a time interval needs to be specified::

  t1 = '2100-02-01'
  t2 = '2150-01-01'

If you want to load a snapshot, simply only define ``t1`` or set ``t1=t2``.
Now, you can read the data as follows::

  to, it_ave   = pyic.time_average(IcD, 'to', t1=t1, t2=t2, iz='all')

With ``iz`` you can specify a single layer e.g. ``iz=0`` or a squence of layers, e.g. ``iz = [0,4,6]`` or all layers ``iz='all'``. 

Using ``netCDF4``
^^^^^^^^^^^^^^^^^

The easiest way to use the ``netCDF4`` library is first to create an IconData object.
This is handy to infer the desired file name, time index within the netcdf file and depth index.

This can be done as follows::

  # --- specify time step
  it = np.argmin(np.abs(IcD.times-np.datetime64('2295-01-01T00:00:00')))
  # --- specify depth level
  iz = np.argmin(np.abs(IcD.depthc-1000.))

After this, the data can be loaded by using the ``netCDF4`` library as follows::

  f = Dataset(IcD.flist_ts[it], 'r')
  to = f.variables['to'][IcD.its[it],iz,:]
  f.close()

Using ``xarray``
^^^^^^^^^^^^^^^^

Normal ``xarray`` syntax can be used to load the data, e.g. by::

  ds = xr.open_dataset(IcD.flist_ts[it])

If data set containing multiple files should be loaded, one can use (note that we only use ``IcD.flist`` here and not ``IcD.flist_ts`` since the later usually contain one file as often as there are time steps within the file)::

  ds = xr.open_mfdataset(IcD.flist, concat_dim='time', data_vars='minimal',
                         coords='minimal', compat='override', join='override')

The different options are experimental and should speed up loading the data set.
However, very often this command is relatively slow and improvements are most likely possible.

In ``xr.open_dataset`` and ``xr.open_mfdataset``, it is possible to enable ``dask`` by specifying chunks. 

After derivations are done and the result should be interpolated and plotted, it could be necessary to transform the data back to ``numpy`` arrays.
This can be achieved either by::

  numpy_var = ds[var].data

if ``ds[var]`` is a xarray or by::

  numpy_var = ds[var].data.compute()

if ``ds[var].data`` is a ``dask`` array. 

However, in any case it is advisable to first try not to convert to ``numpy`` arrays and report potential bugs.
The conversion should only be done as a last option.

Interpolating the data
----------------------

To efficiently make global plots but also for regional plots it is often advisable to interpolate the data before it is plotted.
Interpolating the data speeds up the plotting process a lot and often the loss of accuracy is tolerable in particular for high resolution simulations.
For getting optimal results, consider which resolution is necessary to have a figure on a screen or a paper with the resolution high enough to recognize all important details but try not to simply use the highest resolution since it usually creates unnecessary large figure sizes and computational effort.
For global plot, e.g. often a resolution of 0.3deg is a good compromise between computational effort, figure size and figure quality.

The interpolation philosophy of ``pyicon`` is that interpolation should happen on-the-fly just before plotting. This means calculations should mostly be performed on the original gird and the final variable which should be plotted is only interpolated just before plotting. 
This interpolation is usually done in the computer memory and it is avoided to save the result to disk (although saving the interpolated data to disk is of course possible). 
With this approach, we avoid creating unnecessary interpolation files on disk, however, interpolation needs to be fast in order to get a smooth plotting work flow.
To assure a fast interpolation, it is common in the usage of ``pyicon`` to first create interpolation files for common source and target grids or vertical sections.
These interpolation files are usually created only once and then re-used over and over again.
This procedure allows for a very efficient interpolation even for large grids as the SMT or R2B11 grid.

Create pre-defined interpolation files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For many typical source and target grid combinations, there are already scripts that can generate interpolation files.
These scripts can be found in::

  pyicon/config_ckdtree

To generate the scripts enter the directory, open the desired script, adjust the necessary paths and grid information in the header and execute the script by (first line valid for mistral only)::

  source ../tools/conda_act_mistral_pyicon_env.sh
  ipython --pylab
  %run config_ckdtree_r2b6_oce_r0004.py

Creating the interpolation files can take quite some time (up to several hours for large grids like SMT or R2B11).
Note that maybe other colleagues have already created interpolation files which fit you needs.
It is perfectly fine to use those. They do not even to be copied just the path needs to be set appropriately as discussed below.

Create an own interpolation file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In most cases, it will not be necessary to create a new script for creating interpolation files, since there is already quite a list of example scripts for the most common grids.
However, for more special source/target grids, it might be necessary to create a new interpolation script. 

Therefore, go to::

  cd pyicon/config_ckdtree

Copy one of the example scripts, e.g. ``config_ckdtree_r2b6_oce_r0004.py`` and edit as follows::

  rev:          can be deleted
  tgname:       how you want to call your grid (no spaces in name, better rather short)
  gname:        name of the grid file (see below how to find an appropriate grid file)
  path_tgrid:   path of the grid file
  fname_tgrid:  keep as it is
  path_ckdtree: path where the interpolation files should be saved, take any path which already exists
  path_rgrid:   keep as it is
  path_section: keep as it is

Usually all ICON simulations need a grid file and you probably now which one was used for your simulation (a file containing lots of information about the horizontal grid). 
However, we only need a couple of variables most important clon, clan also sometimes important vlon, vlat, elon, elat. 
So you could use any file which contains these variables. 
In case that you have a file which contains clon, clat but not vlon, vlat, elon, elat you can use this as well. 
However, you can only plot variables which are defined in the center (most of the variables, like ssh, pres, temp, u, v, w) but you cannot plot variables which are defined on vertices (vorticity) or edges (mass_flux). 
If you only want to derive interpolation indices for clon, clat you need to add load_egrid=False, and load_vgrid=False to all ``pyic.ckdtree_hgrid`` and ``pyic.ckdtree_section`` calls in the script. 

Here is an example::

    pyic.ckdtree_hgrid(lon_reg=[-180.,180.], lat_reg=[-90.,90.], res=1.0,
                      fname_tgrid  = fname_tgrid,
                      path_tgrid   = path_tgrid,
                      path_ckdtree = path_rgrid,
                      sname = sname,
                      gname = gname,
                      tgname = tgname,
                      load_egrid=False,
                      load_vgrid=False,
                      )

After modifying and executing the script, new interpolation files are generated for common target grids. 
If the target grids are not sufficient, it is possible to add custom target grids / sections by simply copying and adjusting the existing examples.

Using interpolation files
^^^^^^^^^^^^^^^^^^^^^^^^^

To use the interpolation files in scripts/notebooks, it is necessary to specify the path to the desired interpolation file.
This file needs to be created for the source grid which corresponds to the data. In the following example, we assume a source grid R2B6 revision 4 for the ocean::

  fpath_ckdtree = path_ckdtree + f'rectgrids/r2b6_oce_r0004_res0.30_180W-180E_90S-90N.npz'

Simply exchange the ``res0.30`` by ``res0.10`` to switch from a 0.3 degree target to a 0.1 degree target grid (of course both target grids need to be created before hand as described above).
Finally, you can do the interpolation by the following command using the above defined interpolation file::

  data_interpolated = pyic.interp_to_rectgrid(data, fpath_ckdtree, coordinates='clat clon')

Doing some calculations
-----------------------

Pyicon is designed to let the user concentrate on the actual derivations and manipulation of output data. 
Therefore many aspects like interpolating and plotting are encapsulated in ready-to-use functions that should facilitate the visualizations of the newly derived data.
Regarding the computations themselves, ``pyicon`` supports different ``pyicon`` the usage of certain ``pyicon`` libraries like ``numpy`` and also to a lesser degree (for now) ``dask``.
Many standard derivations for the ocean and some for the atmosphere are already included into ``pyicon``, however the ultimate goal is that users are enabled to easily do their own calculations.
In the following, some libraries to do own calculations and some pre-defined calculations are discussed.

Numpy computations
^^^^^^^^^^^^^^^^^^

Typically all ``pyicon`` arrays are ``numpy`` arrays. 
Therefore, ``numpy`` is the easiest way of doing calculations within ``pyicon``.
However, when performance bottle necks arise for very large data sets it might be advisable to use ``dask`` computations instead.

Dask computations
^^^^^^^^^^^^^^^^^

The support of ``dask`` in ``pyicon`` is still very experimental. 
A more detailed documentation and examples will thus follow (hopefully) soon.

MPI4py computations
^^^^^^^^^^^^^^^^^^^

Sometimes, mpi4py can be used efficiently with ``pyicon`` to speed up repeating tasks by doing tasks in parallel e.g. along the time or vertical coordinate. 
A very common use case for mpi4py is for creating animations.
Examples will follow (hopefully) soon.

Reconstructions
^^^^^^^^^^^^^^^

How to do typical ICON reconstructions (e.g. derive velocities at triangle centres from triangle edges) using the mimetic reconstructions defined in Korn (2017) can be found in the following notebook:

  * ``examp_oce_reconstructions.ipynb``

Some special diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^

We refer to the examples in the notebooks directory to see how some (and more) of the following diagnostics can be derived.

Ocean:

  * overturning streamfunction
  * barotropic streamfunction
  * section transports
  * vertical velocity
  * tracer / heat fluxes 
  * zonal averaging
  * horizontal and vertical velocity / tracer gradients

Atmosphere:

  * height of isobar
  * temperature on isobar 
  * vert. velocity conversion (omega to w)
  * deriving density (equation of state)
  * zonal averaging
  * wind stress curl

Plotting the data
-----------------

Examples for plotting ICON data can probably be found in every ``pyicon`` notebook.
However, some in particularly useful notebooks are:
  
  * ``examp_intro_start.ipynb``
  * ``examp_oce_timeseries.ipynb``
  * ``examp_oceatm_crop_domain.ipynb``
  * ``examp_plotting_arrange_axes.ipynb``
  * ``examp_plotting_map_projections.ipynb``

Saving data as netcdf
---------------------

The following notebook shows how saving data can be achieved:

  * ``examp_oceatm_save_netcdf.ipynb``

