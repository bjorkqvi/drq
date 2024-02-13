import xarray as xr
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.matlib
from dataclasses import dataclass


@dataclass
class GridInfo:
    dx: float
    dy: float
    dt: float
    Nx: int
    Ny: int
    Nt: int
    Lx: float
    Ly: float
    dkx: float
    dky: float
    df: float
    kx: np.ndarray
    ky: np.ndarray
    f: np.ndarray
    pass


def grid_info_from_arrays(
    time: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[dict]:
    # In physical space
    Nt, Nx, Ny = len(time), len(x), len(y)

    dt = int(np.mean(np.diff(time))) / 1_000_000_000  # [s]
    dx = np.mean(np.diff(x))
    dy = np.mean(np.diff(y))
    Lx = np.max(x) - np.min(x)
    Ly = np.max(y) - np.min(y)

    # In spectral space
    df = 1 / (Nt * dt)
    f = np.linspace(
        -np.floor(Nt / 2) * df,
        np.floor(Nt / 2) * df,
        Nt,
    )

    dkx, dky = (
        np.pi * 2 / Lx,
        np.pi * 2 / Ly,
    )

    if np.mod(Nx, 2) == 0 or np.mod(Ny, 2) == 0:
        raise NotImplementedError("Not implemented for even Nx/Ny yet...")
    else:
        kx = np.linspace(
            -np.floor(Nx / 2) * dkx,
            np.floor(Nx / 2) * dkx,
            Nx,
        )
        ky = np.linspace(
            -np.floor(Ny / 2) * dky,
            np.floor(Ny / 2) * dky,
            Ny,
        )

    grid_info = GridInfo(
        Nt=Nt,
        Nx=Nx,
        Ny=Ny,
        dx=dx,
        dy=dy,
        dt=dt,
        Lx=Lx,
        Ly=Ly,
        df=df,
        dkx=dkx,
        dky=dky,
        kx=kx,
        ky=ky,
        f=f,
    )
    return grid_info


def hann_3d(Nx: int, Ny: int, Nt: int) -> tuple[dict]:
    hannx = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, Nx - 1, Nx) / (Nx - 1)))
    hanny = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, Ny - 1, Ny) / (Ny - 1)))
    hannt = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, Nt - 1, Nt) / (Nt - 1)))

    hannxy = np.matlib.repmat(hannx, Ny, 1)
    hannyx = np.matlib.repmat(hanny, Nx, 1)
    hanntx = np.matlib.repmat(hannt, Nx, 1)
    hannxyt = np.repeat(hannxy[:, :, np.newaxis], Nt, axis=2)
    hannyxt = np.repeat(hannyx[:, :, np.newaxis], Nt, axis=2)
    hanntxy = np.repeat(hanntx[np.newaxis, :, :], Ny, axis=0)

    # hann_2d = hannxy * hannyx
    hann_3d = hannxyt * hannyxt * hanntxy

    # wcx = 1 / np.mean(hannx**2)
    # wcy = 1 / np.mean(hanny**2)
    # wct = 1 / np.mean(hannt**2)
    # # wcx = 1 / np.sum(hannx**2) ** 2
    # # wcy = 1 / np.sum(hanny**2) ** 2
    # # wct = 1 / np.sum(hannt**2) ** 2
    # wcxy = wcx * wcy
    # wcxyt = wcxy * wct
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
    if np.mod(Nt, 2) == 0:
        Nt = Nt - 1

    # Cut out any data that is a partial window segment long
    eta = eta[:, :, 0 : int(Nt * n_seg_long)]
    gi = grid_info_from_arrays(time[0:Nt], x, y)
    hann = hann_3d(gi.Nx, gi.Ny, gi.Nt)
    # wc = 1 / np.mean(hann**2)
    wc = 1 / np.sum(hann**2)
    # hann, wc = np.ones((gi.Nx, gi.Ny, gi.Nt)), 1
    KFspec_all = np.zeros((gi.Nx, gi.Ny, gi.Nt))
    t0 = int(-np.floor(Nt * overlap))
    for n in range(n_seg):
        t0 += int(np.floor(Nt * overlap))
        t1 = int(t0 + Nt)
        print(f"t0={t0}, t1={t1}")
        z_3d = eta[:, :, t0:t1]
        z_3d[np.isnan(z_3d)] = 0
        zw = (z_3d - np.mean(z_3d)) * hann

        zf = sp.fft.fftshift(sp.fft.fftn(zw))
        KFspec = (
            np.abs(zf) ** 2 / (gi.dkx * gi.dky * gi.df) / ((gi.Nt * gi.Nx * gi.Ny))
        ) * wc
        KFspec_all += KFspec
        # print(f"Spectral Hs: {4 * np.sqrt(np.sum(KFspec) * gi.df * gi.dkx * gi.dky)}")
        # print(f"Raw Hs: {4*np.std(z_3d - np.mean(z_3d))}")

    return KFspec_all / n_seg, gi.kx, gi.ky, gi.f
