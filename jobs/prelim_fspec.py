import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm
import numpy as np
import scipy as sp
import pandas as pd


def interp_nan(ts: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolates NaN values from the yanked eta-timeseries and determined dt [s]"""
    number_of_nans = len(np.where(np.isnan(ts.eta.values))[0])
    if number_of_nans == 0:
        print(f"Non Nan-values found!")
    else:
        print(f"Interpolating {number_of_nans} NaN-values...")

    dt = int(np.mean(np.diff(ts.time))) / 1_000_000_000
    z = pd.Series(ts.eta.values).interpolate().tolist()
    return z, dt


def calc_fspec(z: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, float]:
    f, E = sp.signal.welch(z, fs=fs, nperseg=len(z) / 8)
    df = np.mean(np.diff(f))
    return f, E, df


def plot_spec(ax, f, E, loglog: bool = False, label: str = "_"):
    if loglog:
        ax.loglog(f, E)
    else:
        ax.plot(f, E, label=label)
        ax.set_xlim([0, 0.5])
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("E(f) (m^2/Hz)")

    return ax


def plot_saturation_spec(ax, f, E, power: float = 5):
    w = f * 2 * np.pi
    S = E / 2 / np.pi
    ax.loglog(w, S * w**power, label="_")
    ax.set_xlabel("w (rad/s)")
    ax.set_ylabel(f"S(w)*w^{power:.1f}")

    return ax


ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")
# ds = xr.open_dataset("sv_data/ekofisk/xygrid_50cm_20191209_1200_plane_sub.nc")
fig, ax = plt.subplots(2, 2)
for x in range(49, 52):
    for y in range(49, 52):
        ts = ds.isel(x=x, y=y)

        z, dt = interp_nan(ts)
        f, E, df = calc_fspec(z, 1 / dt)

        ax[0, 0] = plot_spec(
            ax[0, 0],
            f,
            E,
            label=f"x {ts.x.values}, y {ts.y.values} / Hs_raw/m0 = {4*np.sqrt(np.var(z)):.2f}/{4*np.sqrt(np.sum(E)*df):.2f}",
        )
        ax[0, 1] = plot_spec(ax[0, 1], f, E, loglog=True)

        ax[1, 0] = plot_saturation_spec(ax[1, 0], f, E, power=4)
        ax[1, 1] = plot_saturation_spec(ax[1, 1], f, E, power=5)


ax[1, 1].loglog(
    2 * np.pi * f, np.ones(len(f)) * 9.81**2 * 10**-2, "k--", label="g^2*10^-2"
)
ax[1, 1].legend()
ax[0, 0].legend()
fig.show()
