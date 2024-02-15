import drq

filename = "../spec_data/acqua_alta/wass__20140310_094000_step03_smoothTS_lowess08.nc"

f3d = drq.F3D.from_netcdf(filename)
fkt = drq.FkTheta.from_f3d(f3d)
ef = drq.Ef.from_f3d(f3d)
fkxy = drq.Fkxy.from_f3d(f3d)
fk = drq.Fk.from_fktheta(fkt)
fk2 = drq.Fk.from_ef(ef)


fig, ax = fk.loglog()

ax = fk2.loglog(ax)
fig.show()
