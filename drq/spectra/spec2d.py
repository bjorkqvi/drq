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
from .attributes import SpecAttributes


@add_datavar(name="spec")
@add_direction()
@add_coord(name="v")
class QvTheta(PointSkeleton, SpecAttributes):
    pass


@add_datavar(name="spec")
@add_coord(name="theta")
@add_coord(name="k")
class FkTheta(PointSkeleton, SpecAttributes):
    @classmethod
    def from_f3d(cls, f3d: F3D) -> FkTheta:
        return cls.from_fkxy(Fkxy.from_f3d(f3d))

    @classmethod
    def from_fkxy(cls, fkxy: Fkxy) -> FkTheta:
        dk = 2 * np.pi / 80

        kx, ky = np.meshgrid(fkxy.kx(), fkxy.ky())
        kxy = (kx**2 + ky**2) ** 0.5
        theta = np.arctan2(kx, ky)
        mask = theta < 0
        theta[mask] = theta[mask] + 2 * np.pi

        theta_vec = np.deg2rad(np.arange(0, 360, 5))
        # k_vec = np.arange(dk, np.max(fkxy.kx()), dk)
        k_vec = np.arange(0.01, 7, dk)
        theta_grid, k_grid = np.meshgrid(
            theta_vec,
            k_vec,
        )
        spec_data = fkxy.spec()
        spec = griddata(
            (theta.ravel(), kxy.ravel()),
            spec_data.ravel(),
            (theta_grid, k_grid),
            method="linear",
        )
        spec[np.isnan(spec)] = 0
        fktheta = cls(lon=fkxy.lon(), lat=fkxy.lat(), theta=theta_vec, k=k_vec)
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

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                np.sum(
                    np.sum(self.spec(), 2) * self.dtheta() * self.k() ** (moment + 1)
                )
                * self.dk()
            )
        elif method == "sum":
            return (
                np.sum(
                    np.sum(self.spec(), 2) * self.dtheta() * self.k() ** (moment + 1)
                )
                * self.dk()
            )
        else:
            raise ValueError("Method should be 'integrate' (default) or 'sum'!")


# @add_datavar(name="spec")
# @add_coord(name="k")
# @add_frequency()
# class Fkf(PointSkeleton, SpecAttributes):
#     @classmethod
#     def from_fk3d(cls, f3d: F3D) -> Fkf:
#         kx, ky = np.meshgrid(f3d.kx(), f3d.ky())
#         kxy = (kx**2 + ky**2) ** 0.5
#         theta = np.arctan2(kx, ky)
#         mask = theta < 0
#         theta[mask] = theta[mask] + 2 * np.pi

#         theta_vec = np.deg2rad(np.arange(0, 360, 5))
#         k_vec = np.arange(dk, np.max(fkxy.kx()), dk)
#         theta_grid, k_grid = np.meshgrid(
#             theta_vec,
#             k_vec,
#         )

#         spec = griddata(
#             (theta.ravel(), kxy.ravel()),
#             fkxy.spec().ravel(),
#             (theta_grid, k_grid),
#             method="nearest",
#         )
#         spec[np.isnan(spec)] = 0
#         fktheta = cls(lon=fkxy.lon(), lat=fkxy.lat(), theta=theta_vec, k=k_vec)
#         fktheta.set_spec(spec, allow_reshape=True)
#         return fktheta


@add_datavar(name="spec")
@add_coord(name="kx")
@add_coord(name="ky")
class Fkxy(PointSkeleton, SpecAttributes):
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
class EfTheta(PointSkeleton, SpecAttributes):
    pass
