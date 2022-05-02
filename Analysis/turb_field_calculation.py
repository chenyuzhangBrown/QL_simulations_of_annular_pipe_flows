# Load python packages and Dedalus
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py

#from dedalus import public as de
#from dedalus.extras import flow_tools
#from dedalus.extras import plot_tools
import subprocess
#from dedalus.tools import post
import matplotlib.colors as colors
from numba import jit,njit, literal_unroll
import sys
import csv


start_time = time.time()

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
#dpdz =  -0.01799970941 # ret81 
#dpdz = -0.01543184963 # ret 75
#dpdz = -0.03950553506 # ret 120
dpdz = -0.01982135353 # ret 65
r_in = 0.5
r_out = 1
N_r = 100
N_th = 128
N_z = 512
total_area = N_r * N_th * N_z
#threshold = 0.15
#v_m = 0.993799968

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
#This maps minimum, midpoint and maximum values to 0, 0.5 and 1 respectively i.e.



# set a velocity scale according to the laminar flow to nondimensionalize the parameter
# laminar flow equation
def get_laminar(r, dpdz, nu):
  return dpdz/(4*nu)*(r**2 - (3/(4*np.log(2))*np.log(r))-1)


def turb_field(u, v, v_scale,threshold):
    q = np.sqrt(u*u + v*v)
    q_b = q > threshold*v_scale
    turb_field = q_b.astype(int)
    print(turb_field.shape, flush = True)
    print(turb_field, flush = True)
    print(np.sum(turb_field), flush = True)
    return turb_field



filelist = ["/users/czhang54/data/czhang54/Saved_results/ret85_512_nl/ret85_11_nl_g/ret85_11_nl_g_s1.h5"]
namelist = ["26_s1_{}.png","2_s1_{}.png","3_s1_{}.png","4_s1_{}.png","5_s1_{}.png","5_s2_{}.png","6_s1_{}.png","6_s2_{}.png","7_s1_{}.png","7_s2_{}.png"]
#name = namelist[j]
#print(name)
v_scale = -(1/(nu))* dpdz
tur_frac_array = np.array([])
time_array = np.array([])


# Plot

for j in range(14,15):
    if j == 26:
        series = [1, 2, 3]
    else:
        series = [1]
    for s in series:
        if j == 0:
            path = "/users/czhang54/data/czhang54/Saved_results/checkpoints9/restart.h5"
        else:
            path = "/users/czhang54/data/czhang54/Saved_results/ret85_512_nl/ret85_{}_nl_g/ret85_{}_nl_g_s{}.h5".format(j,j,s)

            file = h5py.File(path, mode='r')
            t = file['scales']['sim_time']
            r = file['scales']['r']['1.0']
            z = file['scales']['z']['1.0']
            th = file['scales']['th']['1.0']
            th = np.asarray(th)
            z = np.asarray(z)
            r = np.asarray(r)
            t = np.asarray(t)

            # ul = file['tasks']['ul']
            # uh = file['tasks']['uh']
            # vl = file['tasks']['vl']
            # vh = file['tasks']['vh']

            # ul = np.asarray(ul)
            # uh = np.asarray(uh)
            # vl = np.asarray(vl)
            # vh = np.asarray(vh)

            u = file['tasks']['u']
            v = file['tasks']['v']

            u = np.asarray(u)
            v = np.asarray(v)

            # calculate turbulence fraction and save to hdf5 files
            f1 = h5py.File("/users/scratch/czhang54/NL_analysis/ret85_nl_turb_field_{}_s_{}.hdf5".format(j,s), "w")
            g1 = f1.create_group('scales')
            g1.create_dataset('t',data=t)
            g1.create_dataset('r',data = r)
            g1.create_dataset('th', data = th)
            g1.create_dataset('z', data = z)
            g2 = f1.create_group('fields')
            for i, threshold in enumerate([0.0001, 0.0002,0.0003,0.0004,0.0005]):
                #f1 = h5py.File("/users/scratch/ret81_analysis/ret81_turb_field_t{}.hdf5".format(threshold), "w")
                #turb_f = turb_field((ul+uh),(vl+vh),v_scale,threshold)
                turb_f = turb_field(u, v ,v_scale,threshold)
                dset = g2.create_dataset("threshold_{}".format(i), (turb_f.shape), dtype='f', data=turb_f)
                print(threshold)
                dset.attrs['threshold'] = threshold
# time_array = time_array * v_m/(r_out - r_in)
# fig = plt.figure(figsize = (4.8,3.6))
# ax = fig.add_subplot(111)
# p = plt.plot(time_array, tur_frac_array, label = 'turbulent fraction', color = 'black')
# ax.grid()
# plt.title('Turbulent Fraction over time')
# plt.xlabel('time')
# plt.ylabel('turbulent fraction')
# plt.savefig("/users/czhang54/scratch/ret120/ret120_tur_frac_zoom.pdf", bbox_inches = "tight")
# plt.close()
# print('plt saved')
end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
            
            

    
    
