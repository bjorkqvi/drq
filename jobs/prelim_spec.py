import xarray as xr
from drq.spectra.spec1d import Ef, Fk, Qv
from drq.spectra.spec2d import Fkxy
from drq.spectra.spec3d import F3D

ds = xr.open_dataset("../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc")


spec = Ef.from_waveplane(ds)
kspec = Fk.from_ef(spec)
Q = Qv.from_ef(spec)
F = Fkxy.from_waveplane(ds)

fn = "wass__20140310_094000_step03_smoothTS_lowess08"
f3d = F3D.from_ds(xr.open_dataset(f"../spec_data/acqua_alta/{fn}.nc"))
