from itertools import product
import pytest
import numpy as np
import pyicon as pyic
from .conftest import raw_grid, processed_tgrid


def test_convert_tgrid_data(raw_grid):
    converted_tgrid = pyic.convert_tgrid_data(raw_grid)

    # Check conversion to pythonic indexing of neighbour info has worked
    neighbour_information = [
        "vertex_of_cell",
        "edge_of_cell",
        "vertices_of_vertex",
        "edges_of_vertex",
        "edge_vertices",
        "adjacent_cell_of_edge",
        "cells_of_vertex",
    ]
    for info in neighbour_information:
        assert converted_tgrid[info].min().values == 0 or -1

        if info.startswith("v") or info.startswith("edge_vertices"):
            assert converted_tgrid[info].max().values == \
                converted_tgrid.dims["vertex"] - 1
        elif info.startswith("e"):
            assert converted_tgrid[info].max().values == \
                converted_tgrid.dims["edge"] - 1
        elif info.startswith("c") or info.startswith("a"):
            assert converted_tgrid[info].max().values == \
                converted_tgrid.dims["cell"] - 1

    # Conversion of ecv lat and lon to degrees
    for point, dim in product("ecv", ("lat", "lon")):
        coord = point + dim
        assert converted_tgrid[coord].attrs["units"] == "degrees"

    # Converted tgrid attribute is there
    assert converted_tgrid.attrs["converted_tgrid"]

    # Check we can't convert a converted grid
    with pytest.raises(ValueError):
        pyic.convert_tgrid_data(converted_tgrid)

    # Dimension ncells is not present and cell is
    assert "ncells" not in converted_tgrid.dims
    assert "cell" in converted_tgrid.dims


@pytest.mark.parametrize("tgrid", ["raw_grid", "processed_tgrid"])
def test_xr_crop_tgrid(tgrid, request):
    # Set ireg_c and crop the grid
    tgrid = request.getfixturevalue(tgrid)

    for point, dim in product("cev", ["lon", "lat"]):
        coord = point + dim
        if tgrid[coord].units == "radian":
            tgrid[coord] = np.degrees(tgrid[coord])

    ireg_c = tgrid["cell"].where(
        (tgrid["clon"] > -5) & (tgrid["clon"] < 5)
        & (tgrid["clat"] > -5) & (tgrid["clat"] < 5),
        drop=True).astype("int32")

    # This checks ireg_c is as expected
    assert ireg_c.sum() == 301614
    assert ireg_c.prod() == -8253145384319188992

    cropped_tgrid = pyic.xr_crop_tgrid(tgrid, ireg_c)

    # Check ireg_[cev] is present
    for point in "cev":
        assert f"ireg_{point}" in cropped_tgrid.keys()

    # Check ncells == len(ireg_c)
    assert cropped_tgrid.dims["cell"] == ireg_c.sizes["cell"]

    # Check ireg_[cev] is correct
    # Ideally we would hash the array and compare, but this will probably do
    assert cropped_tgrid["ireg_c"].sum() == 301614
    assert cropped_tgrid["ireg_c"].prod() == -8253145384319188992

    assert cropped_tgrid["ireg_e"].sum() == 839941
    assert cropped_tgrid["ireg_e"].prod() == 0

    assert cropped_tgrid["ireg_v"].sum() == 135385
    assert cropped_tgrid["ireg_v"].prod() == -1427286351937536000


def test_nabla_funcs(processed_tgrid):
    # Want to check curl of a gradient
    gradient = pyic.xr_calc_grad(processed_tgrid, processed_tgrid["clon"])
    curl_of_grad = pyic.xr_calc_curl(processed_tgrid, gradient)
    assert np.allclose(curl_of_grad, 0)

    # Should include other tests in the future if any refactoring is done
