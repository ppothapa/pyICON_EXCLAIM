# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: pyicon_py39
#     language: python
#     name: pyicon_py39
# ---

# # pyic_tave: A pyicon tool for paralell time averaging using dask

# %%time
import argparse
import glob
import xarray as xr
import dask
from dask.diagnostics import ProgressBar
import numpy as np
import pandas as pd
import sys
# import pyicon as pyic
from dask.distributed import Client, progress
from distributed.scheduler import logger
import socket


# ## Preliminaries

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def main():
    debug_jupyter = isnotebook()
    
    # ### Deciphering arguments
    
    help_text = "Some help text."
    
    # +
    # --- read input arguments
    parser = argparse.ArgumentParser(description=help_text, formatter_class=argparse.RawTextHelpFormatter)
    
    # --- necessary arguments
    parser.add_argument('fpath_data', nargs='+', metavar='fpath_data', type=str,
                        help='Path to ICON data files.')
    parser.add_argument('fpath_out', nargs='+', metavar='fpath_out', type=str,
                        help='Path to output file.')
    parser.add_argument('--dontshow', action='store_true', default=False,
                        help='If dontshow is specified, the plot is not shown')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Show many more messages.')
    # parser.add_argument('--debug_jupyter', action='store_true', default=False,
    #                     help='If option is set, interactive mode from Jupyter Notebook is switched on.')
    parser.add_argument('--time_sel', type=str, default=None,
                        help='Time range in between time averageing will be done.')
    parser.add_argument('--time_isel', type=str, default=None,
                        help='Time records in between averageing will be done.')
    parser.add_argument('--vars', type=str, default=None,
                        help='Variables which are averaged')
    parser.add_argument('--dask', type=str, default=None,
                        help='Specify how das is going to be used.')
    # -
    
    if debug_jupyter:
        print('!!! Warning: Take arguments from Jupyter Cell. !!!')
        fpath_data = '/home/m/m300602/work/proj_vmix/icon/icon_23/icon-oes-trr181/experiments/nib2321_c4/nib2321_P1Y_3d*.nc'
        fpath_out = '/home/m/m300602/work/proj_vmix/icon/icon_23/icon-oes-trr181/experiments/nib2321_c4/nib2321_P1Y_3d_<auto>.nc'
        command_line = f"{fpath_data} {fpath_out} --verbose --vars=to,so --time_sel=1958,2020"
        # command_line = f"{fpath_data} {fpath_out} --verbose --vars=to,so --time_sel=1968,1972"
        iopts = parser.parse_args(command_line.split())
        command_line = "pyic_tave.ipynb " + command_line
    else:
        iopts = parser.parse_args()
        command_line = " ".join(sys.argv)
    if iopts.verbose:
        print(command_line)
    
    
    def decipher_list(string):
        if not string:
            return None
        strings = string.split(',')
        for nn, string in enumerate(strings):
            strings[nn] = string.replace(' ', '')
        return strings
    
    
    fpath_data = iopts.fpath_data[0]
    ave_vars = decipher_list(iopts.vars)
    time_sel = decipher_list(iopts.time_sel)
    time_isel = decipher_list(iopts.time_isel)
    if iopts.verbose:
        print(ave_vars, time_sel, time_isel)
    
    # ### Setting up a dask cluster
    
    nodask = False
    username = 'm300602'
    account_name = 'mh0033'
    dask_tmp_dir = '/work/{account_name}/{username}/dask_tmp'
    
    if not iopts.dask:
        print('!!! Warning: No --dask option specified, continue without dask.!!!')
    elif iopts.dask=='simple_client':
        if __name__=="__main__":
            dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})
            client = Client(n_workers=10, threads_per_worker=1, memory_limit='20GB')
            client.run(numcodecs.register_codec, gribscan.RawGribCodec, "gribscan.rawgrib")
    elif iopts.dask=='local':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(#ip="0.0.0.0",
                           #silence_logs=50,
                           n_workers=32,  # half of workers than cores
                           threads_per_worker=1, # 1 is often fastern than 2
                           memory_limit='20G', # per worker number of tot. memory / by worker (a bit lower)
                          )
        client = Client(cluster)
    elif iopts.dask=='mpi':
        from dask_mpi import initialize
        initialize()
        client = Client()
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        logger.info(f"ssh -L 8788:localhost:{port} -J {username}@levante.dkrz.de {username}@{host}")
    elif iopts.dask=='slurm':
        from dask_jobqueue import SLURMCluster
        queue = 'compute' # Name of the partition we want to use
        job_name = 'pyic_dask' # Job name that is submitted via sbatch
        memory = "100GiB" # Max memory per node that is going to be used - this depends on the partition
        cores = 24 # Max number of cores per task that are reserved - also partition dependend
        walltime = '8:00:00' # Walltime - also partition dependent
        cluster = SLURMCluster(memory=memory,
                               cores=cores,
                               project=account_name,
                               walltime=walltime,
                               queue=queue,
                               name=job_name,
                               scheduler_options={'dashboard_address': ':8787'},
                               local_directory=dask_tmp_dir,
                               job_extra=[f'-J {job_name}', 
                                          f'-D {dask_tmp_dir}',
                                          f'--begin=now',
                                          f'--output={dask_tmp_dir}/LOG_cluster.%j.o',
                                          f'--output={dask_tmp_dir}/LOG_cluster.%j.o'
                                         ],
                               interface='ib0')
        cluster.scale(jobs=2)
        client = Client(cluster)
        if iopts.verbose:
            print(cluster.job_script())
    else:
        print(f'!!! Warning: Unknown dask method: {iopts.dask}. Continuing without a dask cluster. !!!')
    
    try:
        host = client.run_on_scheduler(socket.gethostname)
        port = client.scheduler_info()['services']['dashboard']
        print(f"ssh -L 8788:localhost:{port} -J {username}@levante.dkrz.de {username}@{host}")
        
        client
    except:
        print('!!! Warning: Continuing without dask. !!!')
        nodask = True
    
    # ## Loading the data
    
    flist = glob.glob(fpath_data)
    flist.sort()
    if iopts.verbose:
        print("\n".join(flist))
    
    mfdset_kwargs = dict(combine='nested', concat_dim='time', 
                         data_vars='minimal', coords='minimal', compat='override', join='override',
                         parallel=True,
                        )
    
    chunks = dict(time=1)
    
    if len(flist)==0:
      raise ValueError(f"::: Error: Could not find any file {fpath_data}! :::")
    dso = xr.open_mfdataset(flist, **mfdset_kwargs, chunks=chunks)
    ds = dso.copy()
    
    # ## Appy selections
    
    # +
    if ave_vars:
        if iopts.verbose:
            print('apply ave_vars')
        old_ave_vars = ave_vars.copy()
        ave_vars = []
        varlist = list(ds)
        for var in old_ave_vars:
            if var in varlist:
              ave_vars += [var]
            else:
              print(f'!!! Warning: Variable \'{var}\' is not in data set. !!!')
        ds = ds[ave_vars]
    else:
        ave_vars = list(ds)
        
    if time_isel:
        if iopts.verbose:
            print('apply time_isel')
        ds = ds.isel(time=slice(int(time_isel[0]), int(time_isel[1])))
        time_sel = [ds.time.isel(time=time_isel[0]), ds.time.isel(time=time_isel[1])]
    
    if time_sel:
        if iopts.verbose:
            print('apply time_sel')
        ds = ds.sel(time=slice(time_sel[0], time_sel[1]))
    # -
    
    print(f"--- Averaging variables:\n    {ave_vars}")
    print(f"--- Averaging time records:\n    {ds.time.data}")
    
    # ## Time average
    
    ds_ave = ds.mean(dim='time', keep_attrs=True)
    
    # ## Meta data
    
    # +
    ds_ave['time_records_ave'] = ds.time.rename(time='time_records_ave')
    ds_ave['t1'] = time_sel[0]
    ds_ave['t2'] = time_sel[1]
    ds_ave['time_bnds'] = xr.DataArray(pd.to_datetime(time_sel), dims=['time_bnds'])
    
    ds_ave = ds_ave.assign_attrs({'pyicon': f'{command_line}'})
    # -
    
    if iopts.verbose:
        print(ds_ave)
    
    # ## Saving to netcdf file
    
    # %%time
    fpath_out = iopts.fpath_out[0]
    fpath_out = fpath_out.replace('<auto>',f'{time_sel[0]}_{time_sel[1]}') 
    print(f'Saving netcdf file {fpath_out}')
    write_job = ds_ave.to_netcdf(fpath_out, compute=False)
    if nodask:
        with ProgressBar():
            write_job.compute()
    else:
        x = write_job.persist()
        progress(x)
    
    print('All done!')
    return

if __name__=='__main__':
    main()
