from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_time,
)
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm
import pandas as pd
from matplotlib.widgets import Slider

import xarray as xr
from copy import deepcopy


@add_datavar(name="eta")
@add_time()
class WavePlane(GriddedSkeleton):
    @classmethod
    def from_netcdf(cls, filename: str) -> "WavePlane":
        return cls.from_ds(xr.open_dataset(filename))

    @classmethod
    def from_ekofisk(cls, filename: str) -> "WavePlane":
        """Reads netcdf file of Ekofisk data and applies station specific settings"""
        wp = cls.from_netcdf(filename).flip_yaxis().detrend()
        wp.set_station(lon=3.21, lat=56.55)
        wp.set_box(x=(-45.0, 34.0), y=(120.0, 199.0))
        return wp.cut_to_box()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_box(x=self.edges("x"), y=self.edges("y"), init=True)

    def xgrid(self) -> np.ndarray:
        """Returns a meshgrid of x-values"""
        X, _ = np.meshgrid(self.x(), self.y())
        return X

    def ygrid(self) -> np.ndarray:
        """Returns a meshgrid of y-values"""
        _, Y = np.meshgrid(self.x(), self.y())
        return Y

    def flip_yaxis(self) -> "WavePlane":
        """Flips the y-axis values fron positive to negative (or vice versa)"""
        wp_flip = WavePlane(x=self.x(), y=np.flip(-self.y()), time=self.time())
        wp_flip.set_eta(np.flip(self.eta(), axis=1))
        wp_flip.set_box(x=self.box_x, y=np.flip(-self.box_y), init=True)
        return wp_flip

    def detrend(self) -> "WavePlane":
        """Removes the mean of the data from all surfaes"""
        wp_detrended = deepcopy(self)
        wp_detrended.set_eta(self.eta() - np.nanmean(self.eta()))
        return wp_detrended

    def set_station(self, lon: float, lat: float) -> None:
        """Sets longitude and latitue of station that data is from"""
        self.set_metadata({"lon": lon, "lat": lat}, append=True)

    def set_box(
        self, x: tuple[float, float], y: tuple[float, float], init: bool = False
    ):
        """Sets the box to mark data of interest"""
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)

        if len(x) == 1:
            x = (x[0], x[0])
        if len(y) == 1:
            y = (y[0], y[0])
        self.box_x = np.array(x)
        self.box_y = np.array(y)

        if not init:
            print(
                f"Setting box: x = {x[0]}-{x[1]} ({x[1]-x[0]} m), x = {y[0]}-{y[1]} ({y[1]-y[0]} m)"
            )

            if not self.box_squared():
                print("Box is not a square!")

    def box_squared(self) -> bool:
        """Checks if the set box is squared"""
        x, y = self.box_x, self.box_y
        return np.isclose(np.abs(y[1] - y[0]), np.abs(x[1] - x[0]))

    def cut_to_box(self) -> "WavePlane":
        "Cuts the WavePlane to the ser box"
        x, y = self.box_x, self.box_y
        cut_wp = WavePlane.from_ds(
            self.ds().sel(x=slice(np.min(x), np.max(x)), y=slice(np.min(y), np.max(y)))
        )
        return cut_wp

    def plot(self, time_ind: int = 0, vmin: float = None, vmax: float = None):
        """Plots the surface data with a slider as a function of time"""

        def update_plot(val):
            ax.cla()
            cont = ax.pcolormesh(
                self.xgrid(),
                self.ygrid(),
                self.eta(data_array=True).isel(time=val),
                cmap=cmocean.cm.diff,
                vmin=vmin,
                vmax=vmax,
            )
            time = pd.to_datetime(self.time()[val])
            if self._add_cbar:
                self._add_cbar = False
                cbar = fig.colorbar(cont)
            ax.plot(x, [y[0], y[0]], "r", linestyle="dashed")
            ax.plot(x, [y[1], y[1]], "r", linestyle="dashed")
            ax.plot([x[0], x[0]], y, "r", linestyle="dashed")
            ax.plot([x[1], x[1]], y, "r", linestyle="dashed")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(time.strftime("%Y-%m-%d %H:%M:%S"))

        fig, ax = plt.subplots(1)

        self._add_cbar = True

        if vmin is None or vmax is None:
            min_val = np.nanmin(self.eta())
            max_val = np.nanmax(self.eta())
            max_val = np.maximum(np.abs(min_val), np.abs(max_val))
        if vmin is None:
            vmin = -max_val
        if vmax is None:
            vmax = max_val
        x, y = self.box_x, self.box_y
        if len(self.time()) > 1:
            ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
            time_slider = Slider(
                ax_slider,
                "time_index",
                0,
                len(self.time()) - 1,
                valinit=time_ind,
                valstep=1,
            )
            time_slider.on_changed(update_plot)

        update_plot(0)

        plt.show(block=True)

    def m0(self, nan_to_zero: bool = True):
        eta = self.eta()
        if nan_to_zero:
            eta[np.isnan(eta)] = 0
            return np.var(eta)
        else:
            return np.nanvar(eta)

    def hs(self, nan_to_zero: bool = True):
        return 4 * np.sqrt(self.m0(nan_to_zero=nan_to_zero))
