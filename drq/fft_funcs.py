import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib


def window_1d_to_3d(winx: np.ndarray, winy: np.ndarray, wint: np.ndarray) -> np.ndarray:
    Nx, Ny, Nt = len(winx), len(winy), len(wint)
    winxy = np.matlib.repmat(winx, Ny, 1)
    winyx = np.matlib.repmat(winy, Nx, 1).T
    wintx = np.matlib.repmat(wint, Nx, 1)
    winxyt = np.repeat(winxy[:, :, np.newaxis], Nt, axis=2)
    winyxt = np.repeat(winyx[:, :, np.newaxis], Nt, axis=2)
    wintxy = np.repeat(wintx[np.newaxis, :, :], Ny, axis=0)

    win_3d = winxyt * winyxt * wintxy
    return win_3d


def hann_3d(Nx: int, Ny: int, Nt: int) -> np.ndarray:
    hannt = sp.signal.windows.hann(Nt)
    hanny = sp.signal.windows.hann(Ny)
    hannx = sp.signal.windows.hann(Nx)

    hann_3d = window_1d_to_3d(hannx, hanny, hannt)

    return hann_3d


def tukey_3d(Nx: int, Ny: int, Nt: int) -> np.ndarray:
    hannt = sp.signal.windows.hann(Nt)
    tukeyy = sp.signal.windows.tukey(Ny)
    tukeyx = sp.signal.windows.tukey(Nx)

    tukey_3d = window_1d_to_3d(tukeyx, tukeyy, hannt)

    return tukey_3d


def welch_3d(
    eta: np.ndarray,
    time: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    nperseg: int = 256,
    noverlap: int = None,
    window: str = "hann",
):
    """Takes in surface elevation (time, y, x) and return spectrum (kx,ky,f).
    x- and y-dimensions must be the same (i.e. a square splane)

    window: 'hann' (3D hann-window) or 'tukey' (2D tukey in space and hann in time)"""
    assert len(x) == len(y)

    eta = eta.T
    ## Spectrum will be (kx,ky,f)
    assert eta.shape == (len(x), len(y), len(time))
    if noverlap is None:
        noverlap = nperseg // 2

    # Create block indeces
    start_inds = range(0, len(time) - nperseg, nperseg - noverlap)

    # One window segment is this many time steps long
    Nt = nperseg
    # Cut out any data that is a partial window segment long
    eta = eta[:, :, 0 : start_inds[-1] + nperseg]

    # In physical space
    Nx, Ny = len(x), len(y)
    dt = int(np.mean(np.diff(time))) / 1_000_000_000  # [s]
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))

    # In spectral space
    f = sp.fft.fftshift(sp.fft.fftfreq(Nt, dt))
    df = np.median(np.diff(f))
    kx = 2 * np.pi * sp.fft.fftshift(sp.fft.fftfreq(Nx, dx))
    dkx = np.median(np.diff(kx))
    ky = 2 * np.pi * sp.fft.fftshift(sp.fft.fftfreq(Ny, dy))
    dky = np.median(np.diff(ky))

    if window == "hann":
        window_3d = hann_3d(Nx, Ny, Nt)
    elif window == "tukey":
        window_3d = tukey_3d(Nx, Ny, Nt)

    # wc = 1 / np.mean(hann**2)
    wc = 1 / np.sum(window_3d**2)
    # hann, wc = np.ones((gi.Nx, gi.Ny, gi.Nt)), 1
    KFspec_all = np.zeros((Nx, Ny, Nt))
    for n0 in start_inds:
        z_3d = eta[:, :, n0 : n0 + nperseg]
        z_3d[np.isnan(z_3d)] = 0
        zw = (z_3d - np.mean(z_3d)) * window_3d

        zf = sp.fft.fftshift(sp.fft.fftn(zw))
        KFspec = (np.abs(zf) ** 2 / (dkx * dky * df) / ((Nt * Nx * Ny))) * wc
        KFspec_all += KFspec

    # Return one-sided spectrum
    mask = f > 0
    return 2 * KFspec_all[:, :, mask] / len(start_inds), kx, ky, f[mask]
