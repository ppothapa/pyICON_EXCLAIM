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

    # Dimension ncells is not present and cell is
    assert "ncells" not in converted_tgrid.dims
    assert "cell" in converted_tgrid.dims


@pytest.mark.parametrize("tgrid", ["raw_grid"])
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

    cropped_tgrid = pyic.xr_crop_tgrid(tgrid, ireg_c)

    # Check ireg_[cev] is present
    for point in "ev":
        assert f"ireg_{point}" in cropped_tgrid.keys()

    # Check ncells == len(ireg_c)
    assert cropped_tgrid.dims["cell"] == ireg_c.sizes["cell"]

    # Check ireg_[ev] is correct
    assert cropped_tgrid["ireg_e"].sum() == 839941
    assert cropped_tgrid["ireg_e"].prod() == 0

    assert cropped_tgrid["ireg_v"].sum() == 135385
    assert cropped_tgrid["ireg_v"].prod() == -1427286351937536000
