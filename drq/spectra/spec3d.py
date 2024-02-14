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
        wp: WavePlane,
    ):
        spec, kx, ky, f = welch_3d(
            wp.eta(),
            wp.time(),
            wp.y(),
            wp.x(),
            n_seg=8,
        )
        f3d = cls(x=np.mean(wp.x()), y=np.mean(wp.y()), freq=f, kx=kx, ky=ky)
        f3d.set_spec(spec, allow_reshape=True)
        return f3d

    def m0(self) -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .integrate(coord="freq")
            .values[0]
        )

    def hs(self):
        return 4 * np.sqrt(self.m0())
