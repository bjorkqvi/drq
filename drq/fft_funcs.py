import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib


def hann_3d(Nx: int, Ny: int, Nt: int) -> tuple[dict]:
    hannt = sp.signal.windows.hann(Nt)
    hanny = sp.signal.windows.hann(Ny)
    hannx = sp.signal.windows.hann(Nx)

    hannxy = np.matlib.repmat(hannx, Ny, 1)
    hannyx = np.matlib.repmat(hanny, Nx, 1)
    hanntx = np.matlib.repmat(hannt, Nx, 1)
    hannxyt = np.repeat(hannxy[:, :, np.newaxis], Nt, axis=2)
    hannyxt = np.repeat(hannyx[:, :, np.newaxis], Nt, axis=2)
    hanntxy = np.repeat(hanntx[np.newaxis, :, :], Ny, axis=0)

    hann_3d = hannxyt * hannyxt * hanntxy

    return hann_3d


def welch_3d(
    eta: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    n_seg: int = 8,
    overlap: float = 0.5,
):
    """Takes in surface elevation (time, y, x) and return spectrum (kx,ky,f).
    x- and y-dimensions must be the same (i.e. a square splane)"""
    assert len(x) == len(y)

    eta = eta.T
    ## Spectrum will be (kx,ky,f)
    assert eta.shape == (len(x), len(y), len(time))
    n_seg_long = (
        n_seg - (n_seg - 1) * overlap
    )  # Timeseries is this many segments long when accounting for overlap

    # One window segment is this many time steps long
    Nt = np.floor(len(time) / n_seg_long).astype(int)

    # Cut out any data that is a partial window segment long
    eta = eta[:, :, 0 : int(Nt * n_seg_long)]

    # In physical space
    Nx, Ny = len(x), len(y)
    dt = int(np.mean(np.diff(time))) / 1_000_000_000  # [s]
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    # In spectral space
    f = sp.fft.fftshift(sp.fft.fftfreq(Nt, dt))
    df = np.median(np.diff(f))
    kx = sp.fft.fftshift(sp.fft.fftfreq(Nx, dx))
    dkx = np.median(np.diff(kx))
    ky = sp.fft.fftshift(sp.fft.fftfreq(Ny, dy))
    dky = np.median(np.diff(ky))

    hann = hann_3d(Nx, Ny, Nt)
    # wc = 1 / np.mean(hann**2)
    wc = 1 / np.sum(hann**2)
    # hann, wc = np.ones((gi.Nx, gi.Ny, gi.Nt)), 1
    KFspec_all = np.zeros((Nx, Ny, Nt))
    t0 = int(-np.floor(Nt * overlap))
    for n in range(n_seg):
        t0 += int(np.floor(Nt * overlap))
        t1 = int(t0 + Nt)
        print(f"t0={t0}, t1={t1}")
        z_3d = eta[:, :, t0:t1]
        z_3d[np.isnan(z_3d)] = 0
        zw = (z_3d - np.mean(z_3d)) * hann

        zf = sp.fft.fftshift(sp.fft.fftn(zw))
        KFspec = (np.abs(zf) ** 2 / (dkx * dky * df) / ((Nt * Nx * Ny))) * wc
        KFspec_all += KFspec

    # Return one-sided spectrum
    mask = f > 0
    return 2 * KFspec_all[:, :, mask] / n_seg, kx, ky, f[mask]
