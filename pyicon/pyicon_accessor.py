import xarray as xr
import pyicon as pyic

@xr.register_dataarray_accessor("pyic")
class pyiconDataArray:
    def __init__(self, xarray_obj, path_grids='/work/mh0033/m300602/icon/grids/'):
        self._obj = xarray_obj
        self._gname = None
        self.path_grids = path_grids

    @property
    def gname(self):
        if self._gname is None:
          self._gname = pyic.identify_grid(self.path_grids, self._obj)
        return self._gname
  
    def plot(self, **kwargs):
        da = self._obj
        pyic.plot(da, **kwargs)
        return 

    def plot_sec(self, **kwargs):
        da = self._obj
        pyic.plot_sec(da, **kwargs)
        return 

    def interp(self, **kwargs):
        da = self._obj
        dai = pyic.interp_to_rectgrid_xr(da, **kwargs)
        return dai
