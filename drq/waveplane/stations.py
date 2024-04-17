from .waveplane import WavePlane
import xarray as xr
import numpy as np
import pandas as pd
import dask
from drq import msg


class AcquaAlta(WavePlane):
    @classmethod
    def from_wass(cls, filename):
        """Reads in WASS from Acqua Alta from a netcdf file. Time is decoded from the filename."""

        wanted_sf = 5  # Downsample to this frequency [Hz]
        time_len = 20  # Minutes

        # with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        msg.start("Reading WASS-file")
        msg.plain(filename)

        with xr.open_dataset(filename, chunks="auto") as ds:

            date_str = filename.split("__")[1].split("_")[0]
            time_str = filename.split("__")[1].split("_")[1]
            start_time = pd.to_datetime(f"{date_str} {time_str}")
            dt = ds.time.values.squeeze()[0]
            time = pd.to_datetime(start_time + ds.time.values.squeeze() - dt)

            msg.plain(
                f"Time period in file identified as {time[0].strftime('%Y-%m-%d %H:%M:%S')} - {time[-1].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            msg.middle("Processing data")
            orig_sf = ds.fps.values[0].astype(float)
            count_end = int(time_len * 60 * orig_sf) - 1
            if count_end < len(time):
                msg.plain(
                    f"Cutting to {time_len} minutes: {time[0].strftime('%Y-%m-%d %H:%M:%S')} - {time[count_end].strftime('%Y-%m-%d %H:%M:%S')}"
                )
                ds = ds.isel(count=slice(0, count_end))
                time = pd.to_datetime(start_time + ds.time.values.squeeze() - dt)
            downsampling_factor = int(orig_sf / wanted_sf)
            new_sr = orig_sf / downsampling_factor
            if downsampling_factor > 1:
                msg.plain(
                    f"Downsampling with a factor {downsampling_factor} from {orig_sf:.2f} Hz to {new_sr:.2f} Hz..."
                )

            ds = ds.isel(count=range(0, len(ds.time.values), downsampling_factor))

            time = pd.to_datetime(start_time + ds.time.values.squeeze() - dt)
            x = ds.X_grid[0, :].values
            y = ds.Y_grid[:, 0].values

            wp = cls(x=x / 1000, y=y / 1000, time=time, name="AcquaAlta")
            msg.plain("Masking out bad areas...")
            mask = ds.mask_Z.data.astype(bool)
            mask_z = np.float16(ds.mask_Z.data)
            mask_z[np.logical_not(mask)] = np.nan
            ds["mask_Z"] = (["X", "Y"], mask_z)
            zz = ds.Z * ds.mask_Z
        wp.set_eta(zz.data / 1000)
        wp.detrend()
        wp.flip_yaxis()

        msg.middle("Setting AcquaAlta-specific info")
        wp.set_box(x=(-19.9, 20.1), y=(25.0, 65.0))
        wp.set_station(lat=45.3142467, lon=12.5082483)
        wp.set_metadata({"station": "AquaAlta"})
        msg.stop()
        return wp


class Ekofisk(WavePlane):
    @classmethod
    def from_wass(cls, filename):
        msg.start("Reading WASS-file")
        msg.plain(filename)

        # with dask.config.set(**{"array.slicing.split_large_chunks": True}):
        wp = cls.from_netcdf(filename)
        msg.middle("Processing data")
        wp.detrend()
        wp.flip_yaxis()

        msg.middle("Setting Ekofisk-specific info")
        wp.set_box(x=(-45.0, 34.0), y=(120.0, 199.0))
        wp.set_station(lon=3.21, lat=56.55)
        wp.set_metadata({"station": "Ekofisk"})
        msg.stop()
        return wp


class BlackSea(WavePlane):
    @classmethod
    def from_wass(cls, filename):
        msg.start("Reading WASS-file")
        msg.plain(filename)
        with xr.open_dataset(filename, chunks="auto") as ds:
            wp = cls(y=ds.X[:, 0], x=ds.Y[0, :], time=ds.time)
        wp.set_eta(ds.Z.data)
        msg.middle("Processing data")
        wp.detrend()
        wp.flip_yaxis()
        msg.middle("Setting BlackSea-specific info")
        wp.set_metadata(
            {
                "station": "BlackSea",
                "camera_distance_m": ds.attrs.get("Cameras distance [m]"),
            }
        )
        return wp
