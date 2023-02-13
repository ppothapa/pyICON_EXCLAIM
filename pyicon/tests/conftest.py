import pytest
from pathlib import Path
import xarray as xr
import pyicon as pyic


@pytest.fixture()
def raw_grid():
    cur_dir = Path(__file__).parent.resolve()
    grid_path = cur_dir / "test_data/icon_grid_0014_R02B04_O.nc"

    if not grid_path.exists():
        import requests
        grid_download_link = "http://icon-downloads.mpimet.mpg.de/grids/public/mpim/0014/icon_grid_0014_R02B04_O.nc"
        try:
            r = requests.get(grid_download_link, allow_redirects=True)
            with open(grid_path, "wb") as grid_file:
                grid_file.write(r.content)
        except:
            raise FileNotFoundError("{grid_path} does not exist and unable to \
                download it")

    ds_grid = xr.open_dataset(grid_path)
    return ds_grid


@pytest.fixture()
def processed_tgrid(raw_grid):
    return pyic.convert_tgrid_data(raw_grid)
