import drq
import matplotlib.pyplot as plt

#
# filename = "../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc"
filename = "../sv_data/ekofisk/xygrid_50cm_20200328_1500_plane_sub.nc"

wp = drq.WavePlane.from_netcdf(filename).flip_yaxis()
wp.set_station(lon=3.21, lat=56.55)
wp.set_box(x=(-45.0, 34.0), y=(120.0, 199.0))
# wp.plot()
print(wp.cut_to_box().hs())
f3d = drq.F3D.from_waveplane(wp.cut_to_box())
fkxy = drq.Fkxy.from_f3d(f3d)


ef = drq.Ef.from_f3d(f3d)
fk = drq.Fk.from_f3d(f3d)
fk_theory = drq.Fk.from_ef(ef)
fkt = drq.FkTheta.from_f3d(f3d)
print(f3d.hs())
# fig, ax = fk.loglog(return_handle=True)
# ax = fk_theory.loglog(ax)
# fig.show()
