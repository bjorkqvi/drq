import drq
from drq.fft_funcs import hann_3d
import matplotlib.pyplot as plt

filename = "../sv_data/ekofisk/xygrid_50cm_20200328_1500_plane_sub.nc"

wp = drq.WavePlane.from_netcdf(filename).flip_yaxis().detrend()
wp.set_station(lon=3.21, lat=56.55)
# wp.plot(time_ind=2333)
wp.set_box(x=(-45.0, 34.0), y=(120.0, 199.0))
wp = wp.cut_to_box()

# wp = wp.cut_to_box()
# hann = hann_3d(Nx=len(wp.x()), Ny=len(wp.y()), Nt=len(wp.time()))
# wp.set_eta(wp.eta() * hann.T)
