import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm
import numpy as np

ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")
# ds = xr.open_dataset("sv_data/ekofisk/xygrid_50cm_20191209_1200_plane_sub.nc")
ds["ygrid"] = np.abs(ds.ygrid)
ds["y"] = np.abs(ds.y)

fig, ax = plt.subplots(1)
ax.pcolormesh(ds.xgrid, ds.ygrid, ds.isel(time=4869).eta, cmap=cmocean.cm.diff)
fig.show()
