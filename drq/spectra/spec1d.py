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


@add_datavar(name="spec")
@add_coord(name="v")
class Qv(PointSkeleton):
    pass


@add_datavar(name="spec")
@add_frequency()
class Ef(PointSkeleton):
    @classmethod
    def from_ds(cls, ds: xr.Dataset, x: int = 0, y: int = -100):
        ts = ds.sel(x=x, y=y).interpolate_na(dim="time")
        dt = int(np.mean(np.diff(ts.time))) / 1_000_000_000
        f, E = sp.signal.welch(ts.eta.values, fs=1 / dt, nperseg=len(ts.eta) / 8)
        spec = cls(x=x, y=y, freq=f[1:])
        spec.set_spec(E[1:], allow_reshape=True)
        start_time = pd.to_datetime(ts.time.values[0])
        spec.set_metadata({"start_time": start_time.strftime("%Y-%m-%d %H:%M")})
        return spec

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
        ax.set_title(
            f"x={self.x()[0]}, y={self.y()[0]}, start_time: {self.ds().start_time} UTC"
        )

        if power is None:
            ax.set_xlabel("f [Hz]")
            ax.set_ylabel("E(f) [m^2/Hz]")
        else:
            ax.set_xlabel("w [rad*s]")
            ax.set_ylabel(f"S(w)*w^{power:.1f}")
        return ax


@add_datavar(name="spec")
@add_coord(name="k")
class Fk(PointSkeleton):
    @classmethod
    def from_ef(cls, ef: Ef):
        k = dispersion.wavenumber(w=ef.freq(angular=True))
        cg = dispersion.group_speed(f=ef.freq())

        F = cg * ef.spec() / 2 / np.pi

        spec = cls(x=ef.x(), y=ef.y(), k=k)
        spec.set_spec(F, allow_reshape=True)
        start_time = ef.ds().start_time
        spec.set_metadata({"start_time": start_time})
        return spec

    def plot(self) -> None:
        fig, ax = plt.subplots()
        ax.plot(self.k(), self.spec(squeeze=True))
        ax = self._set_plot_text(ax)
        fig.show()

    def loglog(self, power: float = None, line: float | bool = None) -> None:
        fig, ax = plt.subplots()
        if power is not None:
            spec = self.spec(squeeze=True, angular=True) * self.k(angular=True) ** power
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
        ax.set_title(
            f"x={self.x()[0]}, y={self.y()[0]}, start_time: {self.ds().start_time} UTC"
        )

        if power is None:
            ax.set_xlabel("k [rad/m]")
            ax.set_ylabel("F(k) [m^3]")
        else:
            ax.set_xlabel("k [rad/m]")
            ax.set_ylabel(f"F(k)*k^{power:.1f}")
        return ax
