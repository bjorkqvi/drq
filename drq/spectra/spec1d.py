from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .spec3d import F3D

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

from drq import dispersion

from drq.waveplane.waveplane import WavePlane


class Spec1d:
    def m0(self):
        return self.m(moment=0)

    def hs(self):
        return 4 * np.sqrt(self.m0())


@add_datavar(name="spec")
@add_frequency()
class Ef(PointSkeleton, Spec1d):
    @classmethod
    def from_waveplane(cls, wp: WavePlane) -> Ef:
        ts = wp.ds().interpolate_na(dim="time")
        dt = int(np.mean(np.diff(ts.time))) / 1_000_000_000
        n_seg = 8
        f, E = sp.signal.welch(
            np.squeeze(ts.eta.values), fs=1 / dt, nperseg=len(ts.eta) // n_seg
        )
        spec = cls(x=ts.x.values[0], y=ts.y.values[0], freq=f[1:])
        spec.set_spec(E[1:], allow_reshape=True)
        start_time = pd.to_datetime(ts.time.values[0])
        spec.set_metadata(
            {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                "hs_raw": 4 * np.std(ts.eta.values),
            }
        )
        return spec

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Ef:
        spec_ds = f3d.ds().integrate(coord="kx").integrate(coord="ky")
        return cls.from_ds(spec_ds)

    def m(self, moment: float) -> float:
        return (
            (self.spec(data_array=True) * self.freq() ** moment)
            .integrate(coord="freq")
            .values[0]
        )

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.freq(), self.spec(squeeze=True))
        ax = self._set_plot_text(ax)
        fig.show()

    def loglog(self, power: float = None, line: float | bool = None) -> None:
        fig, ax = plt.subplots()
        if power is not None:
            spec = (self.spec(squeeze=True) / 2 / np.pi) * self.freq(
                angular=True
            ) ** power
        else:
            spec = self.spec(squeeze=True)
        ax.loglog(self.freq(), spec)
        if line is not None:
            if line is True:
                line = 10**-2 * 9.81**2
            ax.loglog(self.freq(angular=True), np.ones(len(self.freq())) * line)
        ax = self._set_plot_text(ax, power=power)
        fig.show()

    def _set_plot_text(self, ax, power: float = None):
        try:
            ax.set_title(
                f"x={self.x()[0]}, y={self.y()[0]}, start_time: {self.ds().start_time} UTC"
            )
        except:
            pass

        if power is None:
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("E(f) [m^2/Hz]")
        else:
            ax.set_xlabel("w [rad*s]")
            ax.set_ylabel(f"S(w)*w^{power:.1f}")
        return ax


@add_datavar(name="spec")
@add_coord(name="k")
class Fk(PointSkeleton, Spec1d):
    @classmethod
    def from_fkxy(cls, fkxy) -> Fk:
        kx = fkxy.kx()
        ky = fkxy.ky()
        k = (kx[kx > 0] ** 2 + ky[ky > 0] ** 2) ** 0.5
        k = k[k < max(fkxy.kx())]

        theta = np.linspace(0, 2 * np.pi, len(kx * 2))
        dtheta = np.mean(np.diff(theta))

        new_spec = np.zeros(len(k))
        for n in range(len(k)):
            kx = k[n] * np.sin(theta)
            ky = k[n] * np.cos(theta)
            print(n)
            new_spec[n] = (
                np.sum(
                    np.array(
                        [
                            fkxy.spec(data_array=True)
                            .interp(kx=x, ky=y, method="nearest")
                            .values[0]
                            for (x, y) in zip(kx, ky)
                        ]
                    )
                )
                * dtheta
            )

        spec = cls(lon=fkxy.lon(), lat=fkxy.lat(), k=k)
        spec.set_spec(new_spec, allow_reshape=True)
        return spec

    @classmethod
    def from_ef(cls, ef: Ef):
        k = dispersion.wavenumber(w=ef.freq(angular=True))
        cg = dispersion.group_speed(f=ef.freq())

        F = cg * ef.spec() / 2 / np.pi

        spec = cls(x=ef.x(), y=ef.y(), k=k)
        spec.set_spec(F, allow_reshape=True)
        start_time = ef.ds().start_time
        spec.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        return spec

    def m(self, moment: float) -> float:
        return (
            (self.spec(data_array=True) * self.k() ** moment)
            .integrate(coord="k")
            .values[0]
        )

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.k(), self.spec(squeeze=True))
        ax = self._set_plot_text(ax)
        fig.show()

    def loglog(self, power: float = None, line: float | bool = None) -> None:
        fig, ax = plt.subplots()
        if power is not None:
            spec = self.spec(squeeze=True) * self.k() ** power
        else:
            spec = self.spec(squeeze=True)
        ax.loglog(self.k(), spec)
        if line is not None:
            if line is True:
                line = 0.5 * 10**-2
            ax.loglog(self.k(), np.ones(len(self.k())) * line)
        ax = self._set_plot_text(ax, power=power)
        fig.show()

    def _set_plot_text(self, ax, power: float = None):
        try:
            ax.set_title(
                f"x={self.x()[0]}, y={self.y()[0]}, start_time: {self.ds().start_time} UTC"
            )
        except:
            pass
        if power is None:
            ax.set_xlabel("k [rad/m]")
            ax.set_ylabel("F(k) [m^3]")
        else:
            ax.set_xlabel("k [rad/m]")
            ax.set_ylabel(f"F(k)*k^{power:.1f}")
        return ax


@add_datavar(name="spec")
@add_coord(name="v")
class Qv(PointSkeleton, Spec1d):
    @classmethod
    def from_ef(cls, ef: Ef):
        c = dispersion.phase_speed(f=ef.freq())

        Q = 9.81 * ef.spec() / 2 / np.pi  # dw/dv = g

        spec = cls(x=ef.x(), y=ef.y(), v=1 / c)
        spec.set_spec(Q, allow_reshape=True)
        start_time = ef.ds().start_time
        spec.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        return spec

    def m(self, moment: float) -> float:
        return (
            (self.spec(data_array=True) * self.v() ** moment)
            .integrate(coord="v")
            .values[0]
        )

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.v(), self.spec(squeeze=True))
        ax = self._set_plot_text(ax)
        fig.show()

    def loglog(self, power: float = None, line: float | bool = None) -> None:
        fig, ax = plt.subplots()
        if power is not None:
            spec = self.spec(squeeze=True) * self.v() ** power
        else:
            spec = self.spec(squeeze=True)
        ax.loglog(self.v(), spec)
        if line is not None:
            if line is True:
                line = 9.81**-2 * 0.5 * 10**-2  ## ???? Check the values here
            ax.loglog(self.v(), np.ones(len(self.v())) * line)
        ax = self._set_plot_text(ax, power=power)
        fig.show()

    def _set_plot_text(self, ax, power: float = None):
        ax.set_title(
            f"x={self.x()[0]}, y={self.y()[0]}, start_time: {self.ds().start_time} UTC"
        )

        if power is None:
            ax.set_xlabel("v [s/m]")
            ax.set_ylabel("Q(v) [m^3/s]")
        else:
            ax.set_xlabel("v [s/m]")
            ax.set_ylabel(f"Q(v)*v^{power:.1f}")
        return ax
