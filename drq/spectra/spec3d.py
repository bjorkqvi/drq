from geo_skeletons import PointSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_frequency,
    add_coord,
)
import xarray as xr
import pandas as pd
import numpy as np
from drq.fft_funcs import welch_3d
from drq.waveplane.waveplane import WavePlane
import xarray as xr

from .attributes import SpecAttributes

from .spec2d import Fkxy
from drq.dispersion import wavenumber
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

import matplotlib.pyplot as plt


@add_datavar(name="spec")
@add_frequency()
@add_coord(name="ky")
@add_coord(name="kx")
class F3D(PointSkeleton, SpecAttributes):
    @classmethod
    def from_netcdf(cls, filename: str) -> "F3D":
        return cls.from_ds(xr.open_dataset(filename))

    @classmethod
    def from_waveplane(cls, wp: WavePlane, window: str = "hann") -> "F3D":
        """Calculates 3D spectrum using 3D FFT of surface data

        window: 'hann' (3D hann-window) or 'tukey' (2D tukey in space and hann in time)
        """
        spec, kx, ky, f = welch_3d(
            wp.eta(),
            wp.time(),
            wp.y(),
            wp.x(),
            nperseg=len(wp.time()) // 8,
            window=window,
        )
        lon, lat = wp.metadata().get("lon", 0), wp.metadata().get("lat", 0)
        f3d = cls(lon=lon, lat=lat, freq=f, kx=kx, ky=ky)
        f3d.set_spec(spec, allow_reshape=True)

        start_time = pd.to_datetime(wp.time()[0])
        f3d.set_metadata(
            {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                "hs_raw": wp.hs(),
            }
        )
        return f3d

    def slice(self, method="nearest", **kwargs):
        """Slices the 3D spectrum in frequency"""
        if kwargs.get("freq") is not None:
            f0 = np.atleast_1d(kwargs.get("freq"))
            assert len(f0) == 1
            sliced_spec_ds = self.sel(freq=f0, method=method).ds()
            f0 = sliced_spec_ds.freq.values[0]
            fkxy = Fkxy.from_ds(sliced_spec_ds.squeeze(drop=True))
            fkxy.set_metadata({"freq": f0}, append=True)

            return fkxy

    def plot(self, f_ind: int = 0, vmin: float = None, vmax: float = None):
        """Plots the 3D spectrum with a slider as a function of frequency"""

        def update_plot(val):
            ax.cla()
            spec_slice = self.slice(freq=self.freq()[val])
            spec_slice.plot(
                ax, fig, log=True, vmin=vmin, vmax=vmax, cbar=self._add_cbar
            )

            if self._add_cbar:
                self._add_cbar = False

        fig, ax = plt.subplots(1)

        self._add_cbar = True

        if vmin is None:
            vmin = np.nanmin(self.spec())
        if vmax is None:
            vmax = np.nanmax(self.spec())
        if len(self.freq()) > 1:
            ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
            freq_slider = Slider(
                ax_slider,
                "freq_index",
                0,
                len(self.freq()) - 1,
                valinit=f_ind,
                valstep=1,
            )
            freq_slider.on_changed(update_plot)

        update_plot(0)

        plt.show(block=True)

    def m(self, moment: int, method: str = "integrate") -> float:
        if method == "integrate":
            return (
                self.spec(data_array=True)
                .integrate(coord="kx")
                .integrate(coord="ky")
                .integrate(coord="freq")
                .values[0]
            )
        elif method == "sum":
            return np.sum(self.spec()) * self.dkx() * self.dky() * self.df()
