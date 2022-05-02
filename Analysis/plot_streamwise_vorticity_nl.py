# Load python packages and Dedalus
import numpy as np  
import time
import matplotlib.pyplot as plt
import h5py

import matplotlib.colors as colors
#from numba import jit,njit, literal_unroll
import sys


start_time = time.time()

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
v_m = 0.7826174748   # unit velocity

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


#@njit(fastmath = True)
def vor(r,th,z,v):
    vor = np.zeros(v.shape)
    for i, r_ in enumerate (r):
            for j, th_ in enumerate (th):
                for k, z_ in enumerate (z):
                    vor[k,j,i] = v[k,j,i]/r[i]
    return vor


#j = int(float(sys.argv[1])) # looping j to determine which file to read/plot
pre_path = "/users/czhang54/data/czhang54/Saved_results/ret85_512_nl/"
post_path = "_s1.h5"

folderlist = ["ret85_14_nl_g"]
#folderlist = ["ret85_9_nl_g","ret85_10_nl_g","ret85_11_nl_g", "ret85_12_nl_g"]
namelist = ["14_s1_{}.png", "11_s1_{}.png","12_s1_{}.png","13_s1_{}.png","14_s1_{}.png","6_s1_{}.png","7_s1_{}.png","11_s1_{}.png","6_s1_{}.png","6_s2_{}.png","6_s3_{}.png"]
#name = namelist[j]




# Plot
for j in range(len(folderlist)):
    name = namelist[j]
    folder = folderlist[j]
    path = pre_path + folder + '/' + folder + post_path
    with h5py.File(path, mode='r') as file:

        t = file['scales']['sim_time']
        r = file['scales']['r']['1.0']
        z = file['scales']['z']['1.0']
        th = file['scales']['th']['1.0']
        t1 = t[:]

        th = np.asarray(th)
        z = np.asarray(z)
        r = np.asarray(r)
        r1=r[:]
        z1=z[:]
        #v_m = 1.0372661594647228 # Avg z-direction velocity calculated beforhand to normalize the field here

        r_,z_=np.meshgrid(z1,r1)

        w_array = np.zeros(r_.shape)

        for index in range(54,55):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            u = file['tasks']['u'][index,:,:,:]
            v = file['tasks']['v'][index,:,:,:]
            w = file['tasks']['w'][index,:,:,:]

            vel = np.asarray([u,v,w])

            for u in vel:
                u = np.asarray(u)

            u = u/v_m
            v = v/v_m
            w = w/v_m

            # Define streamwise vorticity xi
            dvdr = np.gradient(v,axis = 2)
            dudth = np.gradient(u, axis = 1)
            xi = dvdr + vor(r,th,z,v) + vor(r,th,z,dudth)
            xi_show = xi[:,0,:]


            v_min = xi.min()
            v_max = xi.max()
            mid_val = 0

            fig = plt.figure(figsize = (34,4))
            ax = fig.add_subplot(111)

            # p = plt.plot(z1, xi[:,0,50])
            # plt.xlabel('z', fontsize = 17)
            # plt.ylabel(r'$\xi_z', fontsize = 17)
            #log scale
            p=plt.pcolormesh(r_,z_,np.transpose(xi_show),cmap='RdBu_r',clim=(v_min, v_max), norm=colors.SymLogNorm(linthresh=0.05, vmin = v_min, vmax = v_max))

            #plt.xlabel('z', fontsize = 17)
            #plt.ylabel('r', fontsize = 17)
            #plt.title("Streamwise Vorticity at "+ r'$\theta = 0$'+" and t = {:3.1f}".format(t[index]*v_m/0.5), fontsize = 17)

            ax.set_aspect('1')
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            #cbar = plt.colorbar(orientation='horizontal', pad = 0.2)
            #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation = -90)
            #cbar.ax.tick_params(labelsize = 'large')

            # linear scale
            # p=plt.pcolormesh(r_,z_,np.transpose(xi_show),cmap='RdBu_r',clim=(v_min, v_max), norm= MidpointNormalize(midpoint=mid_val, vmin = v_min, vmax = v_max))

            # plt.xlabel('z', fontsize = 17)
            # plt.ylabel('r', fontsize = 17)
            # plt.title("Streamwise Vorticity at "+ r'$\theta = 0$'+" and t = {:3.1f}".format(t[index]*v_m/0.5), fontsize = 17)

            # ax.set_aspect('12')
            # cbar = plt.colorbar(orientation='horizontal', pad = 0.2)
            # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation = -90)
            # cbar.ax.tick_params(labelsize = 'large')

            plt.savefig("/users/czhang54/scratch/NL_analysis/a1_xi_85_" +name.format(index), bbox_inches = "tight")
            print('good')
            plt.close()

        end_time = time.time()
        print('Run time: {}'.format(end_time - start_time))

    
    
