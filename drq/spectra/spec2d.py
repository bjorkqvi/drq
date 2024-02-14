from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec3d import F3D

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

from scipy.interpolate import griddata


class Spec2d:
    def m0(self):
        return self.m(moment=0)

    def hs(self):
        return 4 * np.sqrt(self.m0())


@add_datavar(name="spec")
@add_direction()
@add_coord(name="v")
class QvTheta(PointSkeleton, Spec2d):
    pass


@add_datavar(name="spec")
@add_coord(name="theta")
@add_coord(name="k")
class FkTheta(PointSkeleton, Spec2d):
    @classmethod
    def from_fkxy(cls, fkxy: Fkxy) -> FkTheta:
        dk = np.median(np.diff(fkxy.kx()))

        kx, ky = np.meshgrid(fkxy.kx(), fkxy.ky())
        kxy = (kx**2 + ky**2) ** 0.5
        theta = np.arctan2(kx, ky)
        mask = theta < 0
        theta[mask] = theta[mask] + 2 * np.pi

        theta_vec = np.deg2rad(np.arange(0, 360, 5))
        k_vec = np.arange(dk, np.max(fkxy.kx()), dk)
        theta_grid, k_grid = np.meshgrid(
            theta_vec,
            k_vec,
        )

        spec = griddata(
            (theta.ravel(), kxy.ravel()),
            fkxy.spec().ravel(),
            (theta_grid, k_grid),
            method="nearest",
        )
        spec[np.isnan(spec)] = 0
        fktheta = cls(x=fkxy.x()[0], y=fkxy.y()[0], theta=theta_vec, k=k_vec)
        fktheta.set_spec(spec, allow_reshape=True)
        return fktheta

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, projection="3d")
        dirs, k = np.meshgrid(self.theta(), self.k())
        ax.plot_surface(
            k,
            dirs,
            self.spec(squeeze=True),
            cmap="viridis",
            edgecolor="white",
            vmin=self.spec().min(),
            vmax=self.spec().max(),
            alpha=0.8,
        )

        # ax.pcolormesh(dirs, k, np.log(self.spec(squeeze=True)))
        fig.show()

    def m(self, moment: float) -> float:
        spec = self.spec()
        dtheta = np.median(np.diff(self.theta()))
        k = self.k()
        dk = np.median(np.diff(k))
        return np.sum(np.sum(spec, 2) * dtheta * k ** (moment + 1)) * dk


@add_datavar(name="spec")
@add_coord(name="k")
@add_frequency()
class Fkf(PointSkeleton, Spec2d):
    pass


@add_datavar(name="spec")
@add_coord(name="kx")
@add_coord(name="ky")
class Fkxy(PointSkeleton, Spec2d):
    # @classmethod
    # def from_waveplane(
    #     cls,
    #     ds: xr.Dataset,
    #     x: tuple[float, float] = (-35.0, 25.0),
    #     y: tuple[float, float] = (-180.0, -120.0),
    # ):
    #     ds = ds.sel(x=slice(x[0], x[1]), y=slice(y[0], y[1]))
    #     F = np.abs(fftshift(fft2(ds.eta.values))) ** 2
    #     Fm = np.nanmean(F, axis=0)
    #     kx = np.linspace(
    #         -2 * np.pi * 2.0, 2 * np.pi * 2.0, 121
    #     )  # Wrong but close wnough for pre-implementation
    #     spec = cls(x=np.mean(x), y=np.mean(y), kx=kx, ky=kx)
    #     spec.set_spec(Fm / (kx[0] ** 2), allow_reshape=True)
    #     start_time = pd.to_datetime(ds.time.values[0])
    #     spec.set_metadata(
    #         {
    #             "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
    #             "hs": 4 * np.nanstd(ds.eta.values),
    #         }
    #     )
    #     return spec

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Fkxy:
        spec_ds = f3d.ds().integrate(coord="freq")
        return cls.from_ds(spec_ds)

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
class EfTheta(PointSkeleton, Spec2d):
    pass
