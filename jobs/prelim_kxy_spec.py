from scipy.fftpack import fft2, fftshift
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm
import numpy as np

ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")
x = [-35, 25]
y = [-180, -120]
ds = ds.sel(x=slice(x[0], x[1]), y=slice(y[0], y[1]))
fig, ax = plt.subplots(1)
ax.pcolormesh(ds.xgrid, ds.ygrid, ds.isel(time=4869).eta, cmap=cmocean.cm.diff)
ax.plot(x[0], y[0], "ro")
ax.plot(x[1], y[1], "ro")
fig.show()


F = np.abs(fftshift(fft2(ds.eta.values)))
fig, ax = plt.subplots(1)
Fm = np.nanmean(F, axis=0)
ax.pcolormesh(np.log(Fm))
fig.show()
