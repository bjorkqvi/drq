import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm
import numpy as np
from drq.waveplane.waveplane import WavePlane
from drq.spectra.spec3d import F3D

ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")
# ds = xr.open_dataset("sv_data/ekofisk/xygrid_50cm_20191209_1200_plane_sub.nc")
wp = WavePlane.from_ds(ds).flip_yaxis()
# x: tuple[float, float] = (-35.0, 25.0),
# y: tuple[float, float] = (120.0, 180.0),
wp.set_box(x=(-35.0, 25.0), y=(120.0, 180.0))

f3d = F3D.from_waveplane(wp.cut_to_box())
# breakpoint()
# ds["ygrid"] = np.abs(ds.ygrid)
# ds["y"] = np.abs(ds.y)

# fig, ax = plt.subplots(2)
# ax[0].pcolormesh(ds.xgrid, ds.ygrid, ds.isel(time=1000).eta, cmap=cmocean.cm.diff)

# ax[1].pcolormesh(
#     wp.xgrid(), wp.ygrid(), wp.ds().isel(time=1000).eta, cmap=cmocean.cm.diff
# )
# fig.show()


# fig, ax = plt.subplots(2)
# ax[0].pcolormesh(ds.isel(time=1000).eta, cmap=cmocean.cm.diff)
# ax[1].pcolormesh(wp.ds().isel(time=1000).eta, cmap=cmocean.cm.diff)
# fig.show()
