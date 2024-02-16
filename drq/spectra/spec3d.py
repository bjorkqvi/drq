from geo_skeletons import PointSkeleton
from geo_skeletons.decorators import (
    add_datavar,
    add_frequency,
    add_coord,
)
import xarray as xr
import pandas as pd

from drq.fft_funcs import welch_3d
from drq.waveplane.waveplane import WavePlane
import xarray as xr

from .attributes import SpecAttributes


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
        """window: 'hann' (3D hann-window) or 'tukey' (2D tukey in space and hann in time)"""
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

    def m(self, moment: int, method: str = "integrate") -> float:
        return (
            self.spec(data_array=True)
            .integrate(coord="kx")
            .integrate(coord="ky")
            .integrate(coord="freq")
            .values[0]
        )
