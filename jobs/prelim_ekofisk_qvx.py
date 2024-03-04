import drq
import matplotlib.pyplot as plt

#
# filename = "../sv_data/ekofisk/xygrid_50cm_20231124_1300_plane_sub.nc"
# filename = "../sv_data/ekofisk/xygrid_50cm_20200328_1500_plane_sub.nc"
filename = "../sv_data/ekofisk/xygrid_50cm_20210305_1200_plane_sub.nc"

wp = drq.WavePlane.from_netcdf(filename).flip_yaxis()
wp.set_station(lon=3.21, lat=56.55)
wp.set_box(x=(-45.0, 34.0), y=(120.0, 199.0))
# wp.plot()
print(wp.cut_to_box().hs())
f3d = drq.F3D.from_waveplane(wp.cut_to_box())
fkxf = drq.Fkxf.from_f3d(f3d)
qvx = drq.Qvx.from_fkxf(fkxf)
fkx = drq.Fkx.from_f3d(f3d)
qvy = drq.Qvy.from_f3d(f3d)
fky = drq.Fky.from_f3d(f3d)
qvx2 = drq.Qvx.from_fkx(fkx)
qvx2.name = "linear"
drq.plot([qvx, qvx2])

fk = drq.Fk.from_f3d(f3d)
qv = drq.Qv.from_fk(fk)
qv.plot()
breakpoint()
