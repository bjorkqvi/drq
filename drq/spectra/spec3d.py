from geo_skeletons import PointSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_frequency,
    add_coord,
)
import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@add_datavar(name="spec")
@add_coord(name="kx")
@add_coord(name="ky")
@add_frequency()
class F3D(PointSkeleton):
    pass
