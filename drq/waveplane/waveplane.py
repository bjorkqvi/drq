from geo_skeletons import GriddedSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_time,
)
import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm
import pandas as pd
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox

import xarray as xr


@add_datavar(name="eta")
@add_time()
class WavePlane(GriddedSkeleton):
    @classmethod
    def from_netcdf(cls, filename: str) -> "F3D":
        return cls.from_ds(xr.open_dataset(filename))

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.set_box(x=self.edges("x"), y=self.edges("y"), init=True)

    def xgrid(self):
        X, _ = np.meshgrid(self.x(), self.y())
        return X

    def ygrid(self):
        _, Y = np.meshgrid(self.x(), self.y())
        return Y

    def flip_yaxis(self) -> "WavePlane":
        wp_flip = WavePlane(x=self.x(), y=np.flip(-self.y()), time=self.time())
        wp_flip.set_eta(np.flip(self.eta(), axis=1))
        wp_flip.set_box(x=self.box_x, y=np.flip(-self.box_y))
        return wp_flip

    def set_station(self, lon: float, lat: float) -> None:
        self.set_metadata({"lon": lon, "lat": lat}, append=True)

    def set_box(
        self, x: tuple[float, float], y: tuple[float, float], init: bool = False
    ):
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
        x, y = self.box_x, self.box_y
        return np.isclose(np.abs(y[1] - y[0]), np.abs(x[1] - x[0]))

    def cut_to_box(self) -> "WavePlane":
        x, y = self.box_x, self.box_y
        cut_wp = WavePlane.from_ds(
            self.ds().sel(x=slice(np.min(x), np.max(x)), y=slice(np.min(y), np.max(y)))
        )
        return cut_wp

    def plot(self):
        def update_plot(val):
            ax.cla()
            ax.pcolormesh(
                self.xgrid(),
                self.ygrid(),
                self.eta(data_array=True).isel(time=val),
                cmap=cmocean.cm.diff,
            )
            time = pd.to_datetime(self.time()[val])
            ax.plot(x, [y[0], y[0]], "r", linestyle="dashed")
            ax.plot(x, [y[1], y[1]], "r", linestyle="dashed")
            ax.plot([x[0], x[0]], y, "r", linestyle="dashed")
            ax.plot([x[1], x[1]], y, "r", linestyle="dashed")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_title(time.strftime("%Y-%m-%d %H:%M:%S"))

        fig, ax = plt.subplots(1)
        x, y = self.box_x, self.box_y
        if len(self.time()) > 1:
            ax_slider = plt.axes([0.17, 0.05, 0.65, 0.03])
            time_slider = Slider(
                ax_slider,
                "time_index",
                0,
                len(self.time()) - 1,
                valinit=0,
                valstep=1,
            )
            time_slider.on_changed(update_plot)

        update_plot(0)

        plt.show(block=True)

        #

        # fig.show()

    def m0(self, nan_to_zero: bool = True):
        eta = self.eta()
        if nan_to_zero:
            eta[np.isnan(eta)] = 0
            return np.var(eta)
        else:
            return np.nanvar(eta)

    def hs(self, nan_to_zero: bool = True):
        return 4 * np.sqrt(self.m0(nan_to_zero=nan_to_zero))
