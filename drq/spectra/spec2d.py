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
import cmocean.cm
from scipy.interpolate import griddata
from .attributes import SpecAttributes
import matplotlib.pyplot as plt
from drq.dispersion import wavenumber


class Spectrum2D(PointSkeleton, SpecAttributes):
    pass


@add_datavar(name="spec")
@add_direction()
@add_coord(name="v")
class QvTheta(Spectrum2D):
    pass


@add_datavar(name="spec")
@add_coord(name="theta")
@add_coord(name="k")
class FkTheta(Spectrum2D):
    @classmethod
    def from_f3d(cls, f3d: F3D) -> FkTheta:
        return cls.from_fkxy(Fkxy.from_f3d(f3d))

    @classmethod
    def from_fkxy(cls, fkxy: Fkxy) -> FkTheta:
        dk = 2 * np.pi / 80
        kx, ky = np.meshgrid(fkxy.kx(), fkxy.ky())
        kxy = (kx**2 + ky**2) ** 0.5
        spec_data = fkxy.spec()
        k_vec = np.arange(0.01, 7, dk)

        # Interpolate using 0 to 2*pi so we are continuous around 180
        dD = 5
        # Captures energy better to use a high dtheta in interpolation
        refine_factor = 50
        theta_south = np.arctan2(kx, ky)
        theta_vec = np.deg2rad(np.arange(0, 360, dD / refine_factor))
        theta_grid_south, k_grid = np.meshgrid(
            theta_vec,
            k_vec,
        )
        spec_south = griddata(
            (theta_south.ravel(), kxy.ravel()),
            spec_data.ravel(),
            (theta_grid_south, k_grid),
            method="linear",
        )

        # Interpolate using -pi to pi so we are continuous around 0
        theta_north = np.arctan2(kx, ky)
        mask = theta_north > np.pi
        theta_north[mask] = theta_north[mask] - 2 * np.pi

        theta_grid_north = np.zeros(theta_grid_south.shape)
        mask = theta_grid_south > np.pi
        theta_grid_north[np.logical_not(mask)] = theta_grid_south[np.logical_not(mask)]
        theta_grid_north[mask] = theta_grid_south[mask] - 2 * np.pi
        spec_north = griddata(
            (theta_north.ravel(), kxy.ravel()),
            spec_data.ravel(),
            (theta_grid_north, k_grid),
            method="linear",
        )

        # Combine north and south spec
        spec = spec_north
        ind0, ind1 = (
            len(theta_vec) // 2 - len(theta_vec) // 4,
            len(theta_vec) // 2 + len(theta_vec) // 4,
        )
        spec[:, ind0:ind1] = spec_south[:, ind0:ind1]

        spec[np.isnan(spec)] = 0

        # Bin average down to dD (e.g. 5) degrees
        val_roll = np.matlib.repmat(spec.T, 3, 1).T
        theta_vec_mean = theta_vec[::refine_factor]
        spec_mean = np.zeros((spec.shape[0], len(theta_vec_mean)))
        for i, n in enumerate(
            range(refine_factor, refine_factor + spec.shape[1], refine_factor)
        ):
            spec_mean[:, i] = np.mean(
                val_roll[:, n - refine_factor // 2 : n + refine_factor // 2], 1
            )

        fktheta = cls(lon=fkxy.lon(), lat=fkxy.lat(), theta=theta_vec_mean, k=k_vec)
        fktheta.set_spec(spec_mean, allow_reshape=True)
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
        ax.set_xlabel("k [rad/m]")
        ax.set_ylabel("theta [rad]")

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


@add_datavar(name="spec")
@add_frequency()
@add_coord(name="kx")
class Fkxf(Spectrum2D):
    @classmethod
    def from_f3d(cls, f3d: F3D) -> Fkxf:
        spec_ds = f3d.ds().sum(dim="ky") * f3d.dky()
        return cls.from_ds(spec_ds)

    def m(self, moment: float, method: str = "integrate") -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="freq")
            .values[0]
        )

    def plot(self):
        fig, ax = plt.subplots()
        kx, freq = np.meshgrid(self.kx(), self.freq())
        ax.pcolormesh(kx, freq, np.log(self.spec(squeeze=True)))
        ax.set_xlabel("kx [rad/m]")
        ax.set_ylabel("freq [Hz]")

        fig.show()


@add_datavar(name="spec")
@add_frequency()
@add_coord(name="ky")
class Fkyf(Spectrum2D):
    @classmethod
    def from_f3d(cls, f3d: F3D) -> Fkyf:
        spec_ds = f3d.ds().sum(dim="kx") * f3d.dkx()
        return cls.from_ds(spec_ds)

    def m(self, moment: float, method: str = "integrate") -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="ky")
            .integrate(coord="freq")
            .values[0]
        )

    def plot(self):
        fig, ax = plt.subplots()
        ky, freq = np.meshgrid(self.ky(), self.freq())
        ax.pcolormesh(ky, freq, np.log(self.spec(squeeze=True)))
        ax.set_xlabel("ky [rad/m]")
        ax.set_ylabel("freq [Hz]")

        fig.show()


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

    def m(self, moment: float, method: str = "integrate") -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .values[0]
        )

    def plot(self, ax=None, fig=None, log=True, vmin=None, vmax=None, cbar=True):
        if ax is None:
            fig, ax = plt.subplots()
            show_fig = True
        else:
            show_fig = False
        kx, ky = np.meshgrid(self.kx(), self.ky())
        spec = self.spec(squeeze=True)
        if vmin is None:
            vmin = np.log(np.nanmin(spec))
        if vmax is None:
            vmax = np.log(np.nanmax(spec))

        if log:
            spec = np.log(spec)
            vmin = np.log(vmin)
            vmax = np.log(vmax)
        cont = ax.pcolormesh(kx, ky, spec, cmap=cmocean.cm.haline, vmin=vmin, vmax=vmax)
        if cbar:
            fig.colorbar(cont)
        ax.set_xlabel("kx [rad/m]")
        ax.set_ylabel("ky [rad/m]")
        if hasattr(self.ds(), "freq"):

            ax.set_title(f"f = {self.ds().freq:.5f}")
            k_lin = wavenumber(2 * np.pi * self.ds().freq)
            theta = np.linspace(0, 2 * np.pi, 36)
            ax.plot(np.cos(theta) * k_lin, np.sin(theta) * k_lin, "--k")
        if show_fig:
            fig.show()


@add_datavar(name="spec")
@add_direction()
@add_frequency()
class EfTheta(PointSkeleton, SpecAttributes):
    pass
