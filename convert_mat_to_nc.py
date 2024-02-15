import scipy
from drq.spectra.spec3d import F3D

fn = "wass__20140310_094000_step03_smoothTS_lowess08"
mat = scipy.io.loadmat(f"spec_data/acqua_alta/{fn}.mat")


spec = F3D(
    lon=12.5082483,
    lat=45.3142467,
    kx=mat.get("kxs").squeeze(),
    ky=mat.get("kys").squeeze(),
    freq=mat.get("fs").squeeze(),
)

spec.set_spec(mat.get("KFspec") * 2, allow_reshape=True)
spec.ds().sel(freq=slice(0.00000001, 10_000)).to_netcdf(f"spec_data/acqua_alta/{fn}.nc")
