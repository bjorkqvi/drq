from __future__ import annotations
from typing import TYPE_CHECKING

from .spec2d import FkTheta, Fkxy, Fkxf, Fkyf
from geo_skeletons import PointSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_frequency,
    add_coord,
)
from .spec3d import F3D
import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from copy import deepcopy
from drq import dispersion

from drq.waveplane.waveplane import WavePlane

from .attributes import SpecAttributes
from .plotting import Plotting1D

from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator


class Spectrum1D(PointSkeleton, SpecAttributes, Plotting1D):
    pass


@add_datavar(name="spec")
@add_frequency()
class Ef(Spectrum1D):
    x_label = "freq"
    y_label = "E(f)"
    x_unit = "Hz"
    y_unit = "m^2/Hz"

    @classmethod
    def from_waveplane(cls, wp: WavePlane) -> Ef:
        return cls.from_f3d(F3D.from_waveplane(wp))

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
class Fk(Spectrum1D):
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
        print(f"Using linear dispersion to convert f -> k!!!")
        k = dispersion.wavenumber(w=ef.freq(angular=True))
        cg = dispersion.group_speed(f=ef.freq())
        F = cg * ef.spec() / 2 / np.pi

        spec = cls(lon=ef.lon(), lat=ef.lat(), k=k)
        spec.set_spec(F, allow_reshape=True)
        # start_time = ef.ds().start_time
        # spec.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        spec.name = spec.get_name() + "_linear"
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
class Fkx(Spectrum1D):
    x_label = "kx"
    y_label = "Fx(k)"
    y_unit = "m^3"
    x_unit = "rad/m"

    @classmethod
    def from_fkxy(cls, fkxy: Fkxy) -> Fkx:
        XX, YY = np.meshgrid(fkxy.kx(), fkxy.ky())
        theta = np.arctan2(YY, XX)
        fkxy2 = deepcopy(fkxy)
        fkxy2.set_spec(
            fkxy.spec(squeeze=True) * np.abs(np.cos(theta)), allow_reshape=True
        )
        spec_ds = fkxy2.ds().sum(dim="ky") * fkxy2.dky()
        return cls.from_ds(spec_ds)

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Fkx:
        fkxy = Fkxy.from_f3d(f3d)
        return cls.from_fkxy(fkxy)

    @classmethod
    def from_fkxf(cls, fkxf: Fkxf) -> Fkx:
        spec_ds = fkxf.ds().integrate(coord="freq")
        return cls.from_ds(spec_ds)

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "sum":
            return (self.spec(data_array=True) * self.kx() ** moment).sum(
                dim="kx"
            ).values[0] * self.dkx()
        elif method == "integrate":
            return (
                (self.spec(data_array=True) * self.kx() ** moment)
                .integrate(coord="kx")
                .values[0]
            )
        else:
            raise ValueError("Method should be 'integrate' (default) or 'sum'!")


@add_datavar(name="spec")
@add_coord(name="ky")
class Fky(Spectrum1D):
    x_label = "ky"
    y_label = "Fy(k)"
    y_unit = "m^3"
    x_unit = "rad/m"

    @classmethod
    def from_fkxy(cls, fkxy) -> Fkx:
        XX, YY = np.meshgrid(fkxy.kx(), fkxy.ky())
        theta = np.arctan2(YY, XX)
        fkxy2 = deepcopy(fkxy)
        fkxy2.set_spec(
            fkxy.spec(squeeze=True) * np.abs(np.sin(theta)), allow_reshape=True
        )
        spec_ds = fkxy2.ds().sum(dim="kx") * fkxy2.dkx()
        return cls.from_ds(spec_ds)

    @classmethod
    def from_f3d(cls, f3d) -> Fkx:
        fkxy = Fkxy.from_f3d(f3d)
        return cls.from_fkxy(fkxy)

    @classmethod
    def from_fkyf(cls, fkxf: Fkyf) -> Fky:
        spec_ds = fkxf.ds().integrate(coord="freq")
        return cls.from_ds(spec_ds)

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "sum":
            return (self.spec(data_array=True) * self.ky() ** moment).sum(
                dim="ky"
            ).values[0] * self.dky()
        elif method == "integrate":
            return (
                (self.spec(data_array=True) * self.ky() ** moment)
                .integrate(coord="ky")
                .values[0]
            )
        else:
            raise ValueError("Method should be 'integrate' (default) or 'sum'!")


@add_datavar(name="spec")
@add_coord(name="vx")
class Qvx(Spectrum1D):
    x_label = "vx"
    y_label = "Qx(v)"
    y_unit = "m^3/s"
    x_unit = "s/m"

    @classmethod
    def from_fkxf(cls, fkxf: Fkxf, method: str = "bin") -> Qvx:
        spec = fkxf.spec(squeeze=True) / 2 / np.pi
        wvec, kxvec = fkxf.freq(angular=True), fkxf.kx()
        # interp = RegularGridInterpolator((kxvec, wvec), spec, bounds_error=False)

        if method == "bin":
            vx, qspec = bin_qspec(k=kxvec, w=wvec, spec=spec, dv=0.1)
        # else:
        #     dw = np.median(np.diff(wvec))
        #     for n, v in enumerate(vxvec):
        #         kvec = wvec * v
        #         dk = dw * v
        #         # qspec[n] = np.nansum(interp((kvec, wvec))) * dw * dk / dv
        #         qspec[n] = (
        #             np.nansum(interp((kvec, wvec), method="nearest")) * dw * dk / dv
        #         )
        qvx = cls(
            lon=fkxf.lon(strict=True),
            lat=fkxf.lat(strict=True),
            x=fkxf.x(strict=True),
            y=fkxf.y(strict=True),
            vx=vx,
        )
        qvx.set_spec(qspec, allow_reshape=True)
        return qvx

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Qvx:
        return cls.from_fkxf(Fkxf.from_f3d(f3d))

    @classmethod
    def from_fkx(cls, fkx: Fkx) -> Qvx:
        print(f"Using linear dispersion to convert kx -> vx!!!")
        vx = dispersion.inverse_phase_speed(k=fkx.kx())
        jacobian = 2 * np.sqrt(9.81 * np.abs(fkx.kx()))  # dk/dv
        spec = jacobian * fkx.spec(squeeze=True)

        qvx = cls(
            lon=fkx.lon(strict=True),
            lat=fkx.lat(strict=True),
            x=fkx.x(strict=True),
            y=fkx.y(strict=True),
            vx=vx,
        )
        qvx.set_spec(spec, allow_reshape=True)
        # start_time = ef.ds().start_time
        # qvx.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        qvx.name = qvx.get_name() + "_linear"
        return qvx

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                (self.spec(data_array=True) * self.vx() ** moment)
                .integrate(coord="vx")
                .values[0]
            )


@add_datavar(name="spec")
@add_coord(name="vy")
class Qvy(Spectrum1D):
    x_label = "vy"
    y_label = "Qy(v)"
    y_unit = "m^3/s"
    x_unit = "s/m"

    @classmethod
    def from_fkxf(cls, fkyf: Fkyf, method: str = "bin") -> Qvy:
        spec = fkyf.spec(squeeze=True) / 2 / np.pi
        wvec, kyvec = fkyf.freq(angular=True), fkyf.kx()
        # interp = RegularGridInterpolator((kyvec, wvec), spec, bounds_error=False)

        if method == "bin":
            vy, qspec = bin_qspec(k=kyvec, w=wvec, spec=spec, dv=0.1)
        # else:
        #     dw = np.median(np.diff(wvec))
        #     for n, v in enumerate(vxvec):
        #         kvec = wvec * v
        #         dk = dw * v
        #         # qspec[n] = np.nansum(interp((kvec, wvec))) * dw * dk / dv
        #         qspec[n] = (
        #             np.nansum(interp((kvec, wvec), method="nearest")) * dw * dk / dv
        #         )
        qvy = cls(
            lon=fkyf.lon(strict=True),
            lat=fkyf.lat(strict=True),
            x=fkyf.x(strict=True),
            y=fkyf.y(strict=True),
            vy=vy,
        )
        qvy.set_spec(qspec, allow_reshape=True)
        return qvy

    @classmethod
    def from_fky(cls, fky: Fky) -> Qvx:
        print(f"Using linear dispersion to convert ky -> vy!!!")
        vy = dispersion.inverse_phase_speed(k=fky.ky())
        jacobian = 2 * np.sqrt(9.81 * np.abs(fky.ky()))  # dk/dv
        spec = jacobian * fky.spec(squeeze=True)

        qvy = cls(
            lon=fky.lon(strict=True),
            lat=fky.lat(strict=True),
            x=fky.x(strict=True),
            y=fky.y(strict=True),
            vy=vy,
        )
        qvy.set_spec(spec, allow_reshape=True)
        # start_time = ef.ds().start_time
        # qvx.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        qvy.name = qvy.get_name() + "_linear"
        return qvy

    @classmethod
    def from_f3d(cls, f3d: F3D) -> Qvy:
        return cls.from_fkxf(Fkxf.from_f3d(f3d))

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                (self.spec(data_array=True) * self.vy() ** moment)
                .integrate(coord="vy")
                .values[0]
            )


@add_datavar(name="spec")
@add_coord(name="v")
class Qv(Spectrum1D):
    x_label = "v"
    y_label = "Q(v)"
    y_unit = "m^3/s"
    x_unit = "s/m"

    @classmethod
    def from_ef(cls, ef: Ef):
        print(f"Using linear dispersion to convert f -> v!!!")
        c = dispersion.phase_speed(f=ef.freq())

        Q = 9.81 * ef.spec() / 2 / np.pi  # dw/dv = g

        spec = cls(x=ef.x(), y=ef.y(), v=1 / c)
        spec.set_spec(Q, allow_reshape=True)
        start_time = ef.ds().start_time
        spec.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        spec.name = spec.get_name() + "_linear"
        return spec

    @classmethod
    def from_fk(cls, fk: Fk) -> Qvx:
        print(f"Using linear dispersion to convert k -> v!!!")
        v = dispersion.inverse_phase_speed(k=fk.k())
        jacobian = 2 * np.sqrt(9.81 * fk.k())  # dk/dv
        spec = jacobian * fk.spec(squeeze=True)

        qv = cls(
            lon=fk.lon(strict=True),
            lat=fk.lat(strict=True),
            x=fk.x(strict=True),
            y=fk.y(strict=True),
            v=v,
        )
        qv.set_spec(spec, allow_reshape=True)
        # start_time = ef.ds().start_time
        # qvx.set_metadata({"start_time": start_time, "hs": ef.ds().hs})
        qv.name = qv.get_name() + "_linear"
        return qv

    def m(self, moment: float, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                (self.spec(data_array=True) * self.v() ** moment)
                .integrate(coord="v")
                .values[0]
            )


def bin_qspec(
    k: np.ndarray, w: np.ndarray, spec: np.ndarray, dv: float = 0.1, vmax: float = None
):
    assert spec.shape == (len(k), len(w))
    vmax = vmax or 1 / dv

    vmin = -vmax if np.min(k) < 0 else dv
    vvec = np.arange(vmin, vmax + dv, dv)

    qspec = np.zeros(vvec.shape)

    dw = np.median(np.diff(w))
    dkx = np.median(np.diff(k))
    for n, kval in enumerate(k):
        for m, wval in enumerate(w):
            ind = np.where(
                np.round((vvec / dv)).astype(int)
                == np.round(kval / wval / dv).astype(int)
            )[0]
            if not ind:
                continue
            qspec[ind] += spec[n, m] * dw * dkx / dv

    return vvec, qspec
