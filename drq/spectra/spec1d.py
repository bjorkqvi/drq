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
from drq.spectra.spec2d import FkTheta, Fkxy
import pandas as pd

from drq import dispersion

from drq.waveplane.waveplane import WavePlane

from .attributes import SpecAttributes
from .plotting import Plotting1D


@add_datavar(name="spec")
@add_frequency()
class Ef(PointSkeleton, SpecAttributes, Plotting1D):
    x_label = "freq"
    y_label = "E(f)"
    x_unit = "Hz"
    y_unit = "m^2/Hz"

    @classmethod
    def from_waveplane(cls, wp: WavePlane) -> Ef:
        ts = wp.ds().interpolate_na(dim="time")
        dt = int(np.mean(np.diff(ts.time))) / 1_000_000_000
        n_seg = 8
        f, E = sp.signal.welch(
            np.squeeze(ts.eta.values), fs=1 / dt, nperseg=len(ts.eta) // n_seg
        )
        lon, lat = wp.metadata().get(lon, 0), wp.metadata().get(lat, 0)
        spec = cls(lon=lon, lat=lat, freq=f[1:])
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

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                (self.spec(data_array=True) * self.freq() ** moment)
                .integrate(coord="freq")
                .values[0]
            )
        elif method == "sum":
            return (self.spec(data_array=True) * self.freq() ** moment).sum(
                dim="freq"
            ).values[0] * self.df()
        else:
            raise ValueError("Method should be 'integrate' (default) or 'sum'!")


@add_datavar(name="spec")
@add_coord(name="k")
class Fk(PointSkeleton, SpecAttributes, Plotting1D):
    x_label = "k"
    y_label = "F(k)"
    y_unit = "m^3"
    x_unit = "rad/m"

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Fk:
        return cls.from_fkxy(Fkxy.from_f3d(f3d))

    @classmethod
    def from_fkxy(cls, fkxy) -> Fk:
        return cls.from_fktheta(FkTheta.from_fkxy(fkxy))

    @classmethod
    def from_fktheta(cls, fktheta) -> Fk:
        spec_ds = fktheta.ds().sum(dim="theta") * fktheta.dtheta()
        fk = cls.from_ds(spec_ds)
        spec = fk.spec(squeeze=True)
        fk.set_spec(spec * fk.k(), allow_reshape=True)
        return fk

    @classmethod
    def from_ef(cls, ef: Ef):
        k = dispersion.wavenumber(w=ef.freq(angular=True))
        cg = dispersion.group_speed(f=ef.freq())
        F = cg * ef.spec() / 2 / np.pi

        spec = cls(lon=ef.lon(), lat=ef.lat(), k=k)
        spec.set_spec(F, allow_reshape=True)
        # start_time = ef.ds().start_time
        # spec.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        return spec

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "sum":
            return (self.spec(data_array=True) * self.k() ** moment).sum(
                dim="k"
            ).values[0] * self.dk()
        elif method == "integrate":
            return (
                (self.spec(data_array=True) * self.k() ** moment)
                .integrate(coord="k")
                .values[0]
            )
        else:
            raise ValueError("Method should be 'integrate' (default) or 'sum'!")


@add_datavar(name="spec")
@add_coord(name="kx")
class Fkx(PointSkeleton, SpecAttributes, Plotting1D):
    x_label = "kx"
    y_label = "Fx(k)"
    y_unit = "m^3"
    x_unit = "rad/m"

    @classmethod
    def from_fkxy(cls, fkxy) -> Fkx:
        spec_ds = fkxy.ds().sum(dir="ky") * fkxy.ky()
        return cls.from_ds(spec_ds)

    @classmethod
    def from_f3d(cls, f3d) -> Fkx:
        fkxy = Fkxy.from_f3d(f3d)
        spec_ds = fkxy.ds().sum(dir="ky") * fkxy.ky()
        return cls.from_ds(spec_ds)


@add_datavar(name="spec")
@add_coord(name="ky")
class Fky(PointSkeleton, SpecAttributes, Plotting1D):
    x_label = "ky"
    y_label = "Fy(k)"
    y_unit = "m^3"
    x_unit = "rad/m"

    @classmethod
    def from_fkxy(cls, fkxy) -> Fkx:
        spec_ds = fkxy.ds().sum(dir="kx") * fkxy.kx()
        return cls.from_ds(spec_ds)

    @classmethod
    def from_f3d(cls, f3d) -> Fkx:
        fkxy = Fkxy.from_f3d(f3d)
        spec_ds = fkxy.ds().sum(dir="kx") * fkxy.kx()
        return cls.from_ds(spec_ds)


@add_datavar(name="spec")
@add_coord(name="v")
class Qv(PointSkeleton, SpecAttributes, Plotting1D):
    x_label = "v"
    y_label = "Q(v)"
    y_unit = "m^3/s"
    x_unit = "s/m"

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
