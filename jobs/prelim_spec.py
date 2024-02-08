import xarray as xr
from drq.spectra.spec1d import Ef, Fk, Qv

ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")


spec = Ef.from_ds(ds)
kspec = Fk.from_ef(spec)
Q = Qv.from_ef(spec)
