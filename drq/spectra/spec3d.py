from skeletons.geo_skeletons import PointSkeleton
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
from drq import msg

@add_datavar(name="spec")
@add_frequency()
@add_coord(name="ky")
@add_coord(name="kx")
class F3D(PointSkeleton, SpecAttributes):
    @classmethod
    def from_netcdf(cls, filename: str) -> "F3D":
        return cls.from_ds(xr.open_dataset(filename, chunks="auto"))

    @classmethod
    def from_waveplane(cls, wp: WavePlane, window: str = "hann") -> "F3D":
        """Calculates 3D spectrum using 3D FFT of surface data

        window: 'hann' (3D hann-window) or 'tukey' (2D tukey in space and hann in time)
        """
        msg.start("Calculating 3D spectrum from WavePlane")
        nperseg = len(wp.time()) // 8
        msg.plain(
            f'{f" Welch method (window = '{window}', points/segment = {nperseg}) ":-^80}'
        )
        spec, kx, ky, f = welch_3d(
            wp.eta(),
            wp.time(),
            wp.y(),
            wp.x(),
            nperseg=nperseg,
            window=window,
        )

        lon, lat = wp.metadata().get("lon", 0), wp.metadata().get("lat", 0)
        f3d = cls(lon=lon, lat=lat, freq=f, kx=kx, ky=ky)
        f3d.set_spec(spec)

        msg.middle('Setting metadata')
        # print("Hs from WavePlane for metadata (takes time do decode dask-array)...")
        # hs_raw = np.around(wp.hs(dask=False), 2)
        # print(f"{hs_raw}m")

        start_time = pd.to_datetime(wp.time()[0])
        metadata = wp.metadata().copy()
        if metadata.get("lon") is not None:
            del metadata["lon"]
        if metadata.get("lat") is not None:
            del metadata["lat"]
        f3d.set_metadata(metadata)
        msg.plain(metadata)
        T = (wp.time()[-1] - wp.time()[0]).seconds / 60
        dx = np.diff(wp.edges("x"))[0]
        dy = np.diff(wp.edges("y"))[0]
        fs = 10**9 / np.mean(np.diff(wp.time())).astype(int)
        new_metadata =  {
                "start_time": start_time.strftime("%Y-%m-%d %H:%M"),
                # "hs_raw": hs_raw,
                "duration_min": np.around(T, 1),
                "dx_m": dx,
                "dy_m": dy,
                "sampling_freq_Hz": fs,
            }
        msg.plain(new_metadata)
        f3d.set_metadata(
           new_metadata,
            append=True,
        )
        msg.stop()
        return f3d

    def to_netcdf(self, filename: str = None) -> None:
        """Writes 3D spectral data to netcdf.
        Default filename: 'F3D_{station}_{start_time}_{dt}min_{dx}m_{dy}m.nc'"""
        if filename is None:
            station = self.metadata().get("station", "")
            station += "_"
            T = (self.time()[-1] - self.time()[0]).seconds / 60
            dx = np.diff(self.edges("x"))[0]
            dy = np.diff(self.edges("y"))[0]
            df = 10**9 / np.mean(np.diff(self.time())).astype(int)
            filename = f"WASS_{station}{self.time()[0]:%Y%m%d_%H%M}_{T:.0f}min_{df:.1f}Hz_{dx}m_{dy}m.nc"

        msg.start('Writing to file')
        msg.plain(filename)
        self.ds().to_netcdf(filename)
        msg.stop()

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
                .values
            )
        elif method == "sum":
            return np.sum(self.spec()) * self.dkx() * self.dky() * self.df()
