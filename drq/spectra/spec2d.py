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
from scipy.fft import fft2, fftshift


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
    @classmethod
    def from_ds(
        cls,
        ds: xr.Dataset,
        x: tuple[float, float] = (-35.0, 25.0),
        y: tuple[float, float] = (-180.0, -120.0),
    ):
        ds = ds.sel(x=slice(x[0], x[1]), y=slice(y[0], y[1]))
        F = np.abs(fftshift(fft2(ds.eta.values))) ** 2
        Fm = np.nanmean(F, axis=0)
        kx = np.linspace(
            -2 * np.pi * 2.0, 2 * np.pi * 2.0, 121
        )  # Wrong but close wnough for pre-implementation
        spec = cls(x=np.mean(x), y=np.mean(y), kx=kx, ky=kx)
        spec.set_spec(Fm / (kx[0] ** 2), allow_reshape=True)
        start_time = pd.to_datetime(ds.time.values[0])
        spec.set_metadata(
            {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                "hs": 4 * np.nanstd(ds.eta.values),
            }
        )
        return spec

    def m(self, moment: float) -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .values[0]
        )

    def plot(self):
        fig, ax = plt.subplots()
        kx, ky = np.meshgrid(self.kx(), self.ky())
        ax.pcolormesh(kx, ky, np.log(self.spec(squeeze=True)))
        fig.show()


@add_datavar(name="spec")
@add_direction()
@add_frequency()
class EfTheta(PointSkeleton):
    pass
