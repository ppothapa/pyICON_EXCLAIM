"""
pyic.Simulation is still experimental and was just copied from ~/proj_vmix/simulation_list.ipynb. 
Therefore, it is not working out of the box yet.

It is tested in ~/proj_vmix/simulation_list.ipynb
"""
import glob
import xarray as xr
import numpy as np
import sys
from .pyicon_tb import identify_grid

class Simulation(object):
    def __init__(self, run, path_data='auto', fpath_fx='auto', fpath_tgrid='auto', fpath_ckdtree='auto', name=None, ):
        self.run = run
        if name:
            self.name = name
        else:
            self.name = self.run
        self.path_data = path_data
        self.fpath_fx = fpath_fx
        self.fpath_tgrid = fpath_tgrid
        self.fpath_ckdtree = fpath_ckdtree
        self.DGridMapping = get_grid_uuid()

        path_grid = '/work/mh0033/m300602/icon/grids/'
        if self.fpath_tgrid=='auto':
            print('\'fpath\' needs to be specified: we get an error.')
            Dgrid = identify_grid(path_grid, fpath)
            self.fpath_tgrid = Dgrid['fpath_grid']
        if self.fpath_fx=='auto':
            print('\'lev\' needs to be specified: we get an error.')
            self.fpath_fx = f'{path_grid}/{gname}/{gname}_{lev}_fx.nc'
        return
    def _get_info_str(self):
        info = f"""{self.run}:
  path_data: {self.path_data}
  fpath_fx: {self.fpath_fx}
  fpath_tgrid: {self.fpath_tgrid}
  fpath_ckdtree: {self.fpath_ckdtree}
        """
        return info
    def __str__(self):
        return self._get_info_str()
    def __repr__(self):
        return self._get_info_str()
    def _update_Dinfo(self):
        self.Dinfo = {
            'run': self.run,
            'name': self.name,
            'path_data': self.path_data,
            'fpath_fx': self.fpath_fx,
            'fpath_tgrid': self.fpath_tgrid,
            'fpath_ckdtree': self.fpath_ckdtree,
        }
        return
    def to_database(self, path_database='/home/m/m300602/icon_simulation_database/'):
        fpath = f'{path_database}/{self.run}.json'
        print(f'Writing {fpath}.')
        self._update_Dinfo()
        with open(fpath, 'w') as fj:
            json.dump(self.Dinfo, fj, sort_keys=True, indent=4)
        return
    def get_fx_from_path(self):
        fpath_fx = glob.glob(f'{self.path_data}/*fx*.nc')
        if len(fpath_fx)<1:
            print(f'::: Warning for {self.run}: No fx-file found! :::')
        else:
            self.fpath_fx = fpath_fx[0]
        return
    def get_tgrid_from_path(self):
        fpath_tgrid = glob.glob(f'{self.path_data}/*ocean-grid.nc')
        if len(fpath_tgrid)<1:
            print(f'::: Warning for {self.run}: No tgrid-file found! :::')
        else:
            self.fpath_tgrid = os.path.realpath(fpath_tgrid[0])
        return
    def get_ckdtree_from_tgrid(self, path_grid='/work/mh0033/m300602/icon/grids/'):
        ds = xr.open_dataset(self.fpath_tgrid)
        uuid = ds.attrs['uuidOfHGrid']
        self.path_grid = path_grid
        self.gname = self.DGridMapping[uuid]
        self.fpath_ckdtree = f"{self.path_grid}/{self.gname}/ckdtree/rectgrids/{self.gname}_res0.30_180W-180E_90S-90N.nc"
        return

def get_grid_uuid(fpath_tgrid='/home/m/m300602/work/icon/grids/*/*_tgrid.nc'):
    flist = glob.glob(fpath_tgrid)
    flist.sort()
    DGridMapping = dict()
    for fname in flist:
        gname = fname.split('/')[-1].split('_tgrid')[0]
        ds = xr.open_dataset(fname)
        uuid = ds.attrs['uuidOfHGrid']
        DGridMapping[uuid] = gname
        # print(f"'{ds.attrs['uuidOfHGrid']}': '{gname}'")
    """
    DGridMapping = {
        'e85b34ae-6577-11eb-81a9-93127e10b90d': 'r2b10_atm_r0039'
        '99c03a3c-6578-11eb-8e42-f565c1ad0089': 'r2b10_oce'
        '9b323b62-6704-11eb-85af-dd9149101e1a': 'r2b11_oce'
        '5bd948e8-ac1a-11ea-a6b1-d317264fdca9': 'r2b4_oce_r0004'
        'f4ed57f6-b2ea-11e9-ae92-c52a3fa37d96': 'r2b6_oce_r0004'
        '66c2eb2c-9bd9-11e8-97bc-e1d6091d8653': 'r2b8_oce_r0004'
        '0f1e7d66-637e-11e8-913b-51232bb4d8f9': 'r2b9_atm_r0015'
        '375cb0cc-637e-11e8-9d6f-8f41a9b9ff4b': 'r2b9_oce_r0004'
        '0d39853e-c26b-11e9-8454-0b16a6d45f73': 'smt'
    }
    """
    return DGridMapping

def inquire_simulation_details(path_data):
    if path_data.endswith('/'):
        path_data = path_data[:-1]
    S = Simulation(path_data.split('/')[-1], path_data=path_data)
    S.get_fx_from_path()
    S.get_tgrid_from_path()
    S.get_ckdtree_from_tgrid()
    return S
    
class SimulationList(object):
    def __init__(self, namelist=[], simlist=[]):
        self.list = []
        for name in namelist:
            self.add_sim(name)
        for S in simlist:
            self.add_sim(S)
        self.dict = dict()
        return
    def add(self, S):
        if isinstance(S, str):
              S = Simulation(S)
        self.list.append(S)
        try:
            self.dict[S.run] = S
        except:
            print("::: Warning: Could not add {S.run} to SimulationList.dict! :::")
    def __iter__(self):
        return SimulationIterator(self)
    def __getitem__(self, item):
         return self.dict[item]
    def __str__(self):
        info = ""
        for S in self.list:
            info += S.__str__() + "\n"
        return info
    def __repr__(self):
        info = ""
        for S in self.list:
            info += S.__str__() + "\n"
        return info
    def from_database(self, path_database='/home/m/m300602/icon_simulation_database/'):
        flist = glob.glob(f'{path_database}/*.json')
        flist.sort()
        for fpath in flist:
            with open(fpath, 'r') as fj:
                Dsim = json.load(fj)
            S = Simulation(**Dsim)
            self.add(S)
        return
    def to_database(self, path_database='/home/m/m300602/icon_simulation_database/'):
        for nn, S in enumerate(self):
            S.to_database()
        return
    
class SimulationIterator:
    def __init__(self, simulationList):
        self._simulationList = simulationList
        self._index = 0
    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < (len(self._simulationList.list)):
            result = self._simulationList.list[self._index]
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration
