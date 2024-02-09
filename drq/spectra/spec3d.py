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
@add_frequency()
@add_coord(name="ky")
@add_coord(name="kx")
class F3D(PointSkeleton):
    def m(self, moment: float) -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .integrate(coord="freq")
            .values[0]
        )
