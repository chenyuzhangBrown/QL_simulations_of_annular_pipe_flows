# Load python packages and Dedalus
import numpy as np 
from mpi4py import MPI 
import time
import matplotlib.pyplot as plt
import h5py

from dedalus import public as de 
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools
import subprocess
from dedalus.tools import post
import matplotlib.colors as colors

# Parameters
nu = 0.00020704166
dpdz = -0.01543184963
r_in = 0.5
r_out = 1
v_m = 1.204527773474421    # unit velocity



# Run it for the first time to check the output path. Once it's done this can be commented out.

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



# laminar flow equation
def get_laminar(r, th, dpdz, nu):
    laminar_f = np.zeros(len(r))
    for i, R in enumerate(r):
        laminar_f[i] = (-dpdz)/(4*nu)*(r_in**2 - R**2 ) + (-dpdz/(4*nu))*(r_out**2-r_in**2)*(np.log(R/r_in)/np.log(r_out/r_in))
    #print(laminar_f)
    print('laminar', max(laminar_f), min(laminar_f))
    return laminar_f


#post.merge_process_files("/users/czhang54/scratch/state_variables25", cleanup=True)
#print(subprocess.check_output("find /users/czhang54/scratch/state_variables21", shell=True).decode())
#path = "/users/czhang54/data/czhang54/Saved_results/ret65_512_nl/ret65_11_nl_g/ret65_11_nl_g_s1.h5"
path = "/users/czhang54/data/czhang54/Saved_results/ret75/ret75_19_g/ret75_19_g_s1.h5"

# Plot
with h5py.File(path, mode='r') as file:
    t = file['scales']['sim_time']
    ##u = file['tasks']['u']
    ##v = file['tasks']['v']
    ##p = file['tasks']['p']
    ##xi_z=file['tasks']['xi_z']
    r = file['scales']['r']['1.0']
    z = file['scales']['z']['1.0']
    th = file['scales']['th']['1.0']
    t1 = t[:]
    r1=r[:]
    th1=th[:]
    
    
    
    for index in range(len(t1) - 1,len(t1)):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='polar')
        # QL
        v = file['tasks']['vl'][index,256,:,:]
        # v = np.sum(v, axis = 0)
        # v_avg = v/(len(z))
        # laminar_f = get_laminar(r1,th1,dpdz,nu)
        # print(laminar_f.shape)
        # v_laminar = []
        # for i in range(len(th)):
        #     v_laminar.append(laminar_f)
        # v_laminar = np.array(v_laminar)
        # v = (v_avg - v_laminar)/v_m

        # FNL
        # v = file['tasks']['u'][index,256,:,:]
        # v_sum = np.sum(v, axis = 0)
        # v_avg = v_sum/(len(th))
        # print('avg', max(v_avg), min(v_avg))
        # v_avg = v_avg - get_laminar(r1,th1,dpdz,nu)
        # v = []
        # for i in range(len(th)):
        #     v.append(v_avg)
        v = np.array(v)/v_m
        theta_,r_=plot_tools.quad_mesh(th1,r1)
        #print(theta_.shape)
        #print(v.shape)
        v_min = v.min()
        v_max = v.max()
        mid_val = 0

        #p=plt.pcolormesh(theta_,r_,np.transpose(v),cmap='RdBu_r',clim=(v_min, v_max), norm=colors.SymLogNorm(linthresh=0.03, vmin = v_min, vmax = v_max))
        p=plt.pcolormesh(theta_,r_,np.transpose(v),cmap='RdBu_r',clim=(v_min, v_max), norm=MidpointNormalize(midpoint = mid_val, vmin = v_min, vmax = v_max))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        #plt.title(r'$\bar{v}_\theta$' +" at z = 0 and t = {:3.1f}".format(t[index]*v_m/0.5))
        #ax.set_aspect('equal')
        plt.colorbar(p)
        plt.savefig("/users/czhang54/scratch/ret75_analysis/vl_z_256_polar.png", bbox_inches = 'tight')
        plt.close()
        ##plt.pcolormesh(r_,z_,xi,cmap='PuOr')
       
    
    
