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
import numpy.matlib

from drq.fft_funcs import welch_3d
from drq.waveplane.waveplane import WavePlane


@add_datavar(name="spec")
@add_frequency()
@add_coord(name="ky")
@add_coord(name="kx")
class F3D(PointSkeleton):
    @classmethod
    def from_waveplane(
        cls,
        waveplane: WavePlane,
        x: tuple[float, float] = (-35.0, 25.0),
        y: tuple[float, float] = (-180.0, -120.0),
    ):

        wp = waveplane.sel(x=slice(x[0], x[1]), y=slice(y[0], y[1]))
        spec, kx, ky, f = welch_3d(
            wp.eta(),
            wp.time(),
            wp.y(),
            wp.x(),
            n_seg=8,
        )
        df = np.mean(np.diff(f))
        dk = np.mean(np.diff(kx))
        eta = wp.eta()
        eta[np.isnan(eta)] = 0
        print(f"Spectral Hs: {4 * np.sqrt(np.sum(spec) * df * dk * dk)}")
        print(f"Raw Hs: {4*np.std(eta - np.mean(eta))}")
        breakpoint()

    def m(self, moment: float) -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .integrate(coord="freq")
            .values[0]
        )
