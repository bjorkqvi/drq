from geo_skeletons import PointSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_frequency,
    add_coord,
    add_direction,
)
import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@add_datavar(name="spec")
@add_direction()
@add_coord(name="v")
class QvTheta(PointSkeleton):
    pass


@add_datavar(name="spec")
@add_direction()
@add_coord(name="k")
class FkTheta(PointSkeleton):
    pass


@add_datavar(name="spec")
@add_coord(name="k")
@add_frequency()
class Fkf(PointSkeleton):
    pass


@add_datavar(name="spec")
@add_coord(name="kx")
@add_coord(name="ky")
class Fkxy(PointSkeleton):
    pass


@add_datavar(name="spec")
@add_direction()
@add_frequency()
class EfTheta(PointSkeleton):
    pass
