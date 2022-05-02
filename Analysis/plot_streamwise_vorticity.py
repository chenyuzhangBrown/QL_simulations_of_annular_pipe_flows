# Load python packages and Dedalus
import numpy as np 
from mpi4py import MPI 
import time
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from numpy import arange
import h5py
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from dedalus import public as de 
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools
import subprocess
from dedalus.tools import post
import matplotlib.colors as colors
from numba import jit,njit, literal_unroll
import sys
import csv

start_time = time.time()

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
v_m = 1.204527773474421

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


@njit(fastmath = True)
def vor(r,th,z,v):
    vor = np.zeros(v.shape)
    for i, r_ in enumerate (r):
            for j, th_ in enumerate (th):
                for k, z_ in enumerate (z):
                    vor[k,j,i] = v[k,j,i]/r[i]
    return vor

#def dudth_or(r,th,z,v):
#    dudth_or = np.zeros(v.shape)
#    for i, r_ in enumerate (r):
#            for j, th_ in enumerate (th):
#                for k, z_ in enumerate (z):
#                    dudth_or[k,j,i] = dudth[k,j,i]/r[i]

#    return dudth_or
j = int(float(sys.argv[1])) # looping j to determine which file to read/plot

#filelist = ["/users/czhang54/scratch/ret81/ret81_g/ret81_g_s1.h5","/users/czhang54/scratch/ret81/ret81_2_g/ret81_2_g_s1.h5","/users/czhang54/scratch/ret81/ret81_3_g/ret81_3_g_s1.h5","/users/czhang54/scratch/ret81/ret81_4_g/ret81_4_g_s1.h5","/users/czhang54/scratch/ret81/ret81_5_g/ret81_5_g_s1.h5","/users/czhang54/scratch/ret81/ret81_6_g/ret81_6_g_s1.h5","/users/czhang54/scratch/ret81/ret81_7_g/ret81_7_g_s1.h5","/users/czhang54/scratch/ret81/ret81_8_g/ret81_8_g_s1.h5","/users/czhang54/scratch/ret81/ret81_9_g/ret81_9_g_s1.h5","/users/czhang54/scratch/ret81/ret81_10_g/ret81_10_g_s1.h5","/users/czhang54/scratch/ret81/ret81_11_g/ret81_11_g_s1.h5","/users/czhang54/scratch/ret81/ret81_12_g/ret81_12_g_s1.h5","/users/czhang54/scratch/ret81/ret81_13_g/ret81_13_g_s1.h5","/users/czhang54/scratch/ret81/ret81_14_g/ret81_14_g_s1.h5","/users/czhang54/scratch/ret81/ret81_15_g/ret81_15_g_s1.h5","/users/czhang54/scratch/ret81/ret81_16_g/ret81_16_g_s1.h5","/users/czhang54/scratch/ret81/ret81_17_g/ret81_17_g_s1.h5","/users/czhang54/scratch/ret81/ret81_18_g/ret81_18_g_s1.h5"]
filelist = ["/users/czhang54/data/czhang54/Saved_results/ret160_r3/ret160_r3_1_g/ret160_r3_1_g_s9.h5"]
#filelist = ["/users/czhang54/scratch/ret81_r2_2_g/ret81_r2_2_g_s18.h5", \
#            "/users/czhang54/scratch/ret81_r2_2_g/ret81_r2_2_g_s19.h5", \
#            "/users/czhang54/scratch/ret81_r2_2_g/ret81_r2_2_g_s20.h5", \
#            "/users/czhang54/scratch/ret81_r2_2_g/ret81_r2_2_g_s21.h5"]
namelist = ["_1_s7_{}.png","1_s19_{}.png","1_s20_{}.png","1_s21_{}.png","2_s5_{}.png", "2_s6_{}.png","2_s7_{}.png", "2_s8_{}.png", "2_s9_{}.png", "2_s10_{}.png", "2_s11_{}.png", "11_s2_{}.png", "12_s1_{}.png", "13_s1_{}.png", "14_s1_{}.png", "15_s1_{}.png", "16_s1_{}.pdf", "17_s1_{}.pdf", "18_s1_{}.pdf"] 
#filelist = ["/users/czhang54/scratch/ret85_nl/ret85_2_nl_g/ret85_2_nl_g_s1.h5", "/users/czhang54/scratch/ret85_nl/ret85_3_nl_g/ret85_3_nl_g_s1.h5", "/users/czhang54/scratch/ret85_nl/ret85_3_nl_g/ret85_3_nl_g_s2.h5", "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s1.h5", "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s2.h5", "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s3.h5", "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s1.h5", "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s2.h5", "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s3.h5", "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s1.h5", "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s2.h5", "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s3.h5"]
filelist = np.asarray(filelist)
name = namelist[j]



#post.merge_process_files("/users/czhang54/scratch/state_variables25", cleanup=True)
#print(subprocess.check_output("find /users/czhang54/scratch/state_variables21", shell=True).decode())
with h5py.File(filelist[j], mode='r') as file:
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

    r_,z_=np.meshgrid(z1,r1)
    r_w = np.reshape(r_, len(z1)*len(r1))
    z_w = np.reshape(z_, len(z1)* len(r1))
    
    for index in range(len(t1)):
        #fig = plt.figure()
        ul = file['tasks']['ul'][index,:,:,:]
        uh = file['tasks']['uh'][index,:,:,:]
        vl = file['tasks']['vl'][index,:,:,:]
        vh = file['tasks']['vh'][index,:,:,:]
        wl = file['tasks']['wl'][index,:,:,:]
        wh = file['tasks']['wh'][index,:,:,:]
        print(wh.shape)
        
        vel = np.asarray([ul,uh,vl,vh,wl,wh])

        for u in vel:
            u = np.asarray(u)

        u = ul + uh
        v = vl + vh
        w = wl + wh

        u = u/v_m
        v = v/v_m
        w = w/v_m

        # Define streamwise vorticity xi
        dvdr = np.gradient(v,axis = 2)
        #vor = np.zeros(v.shape)
        dudth = np.gradient(u, axis = 1)
        xi = dvdr + vor(r,th,z,v) + vor(r,th,z,dudth)
        xi_show = xi[:,0,:]


        v_min = xi.min()
        v_max = xi.max()
        mid_val = 0
        #fig = plt.figure()
        fig = plt.figure(figsize = (34,4))
        ax = fig.add_subplot(111)

        p=plt.pcolormesh(r_,z_,np.transpose(xi_show),cmap='RdBu_r',clim=(v_min, v_max), norm=colors.SymLogNorm(linthresh=0.03, vmin = v_min, vmax = v_max))


        #plt.xlabel('z', fontsize = 17)
        #plt.ylabel('r', fontsize = 17)
        #plt.xticks(fontsize = 17)
        #plt.yticks(fontsize = 17)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #plt.title("Streamwise Vorticity at "+ r'$\theta = 0$, QL simulation', fontsize = 17)
        #print('title printed')
        ax.set_aspect('1')

        #plt.colorbar(p)
        #cbar = plt.colorbar(orientation='horizontal',pad = 0.2)
        #cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation = -90)
        #cbar.ax.set_xticklabels(ticks = [v_min, 0, v_max])
        #ticks = [v_min, 0, v_max]
        ##cbar.ax.tick_params(labelsize = 'large')
        #cbar.ax.locator_params(nbins = 5)
        #cbar.formatter.set_powerlimits((0, 0))
        #font = matplotlib.font_manager.FontProperties(family='times new roman', size=16)
        #text.set_font_properties(font)
        plt.savefig("/users/czhang54/scratch/xi_ret160_r3/a1_xi"+name.format(index),bbox_inches = "tight")
        #print("/users/czhang54/scratch/ret120/xi/xi_"+name.format(index), " saved")

        plt.close()

        # save data
        # header = ['z_', 'r_', 'xi']
        # xi_w = np.reshape(xi_show, len(z1)*len(r1))
        # data = np.transpose([z_w,r_w,xi_w])
        # with open('/users/czhang54/scratch/xi_75/a1_xi_75_19_s1_{}.csv'.format(index), 'w') as f:
        #     w = csv.writer(f)
        #     w.writerow(header)
        #     w.writerows(data)

end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
        
        
