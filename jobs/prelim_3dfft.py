import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm
import numpy as np
from drq.spectra.spec3d import F3D
from drq.waveplane.waveplane import WavePlane

# ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")
ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20191209_1200_plane_sub.nc")
waveplane = WavePlane.from_ds(ds)

f3d = F3D.from_waveplane(waveplane)
