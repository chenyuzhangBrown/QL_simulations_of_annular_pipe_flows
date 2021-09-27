# Load python packages and Dedalus
import numpy as np 
from mpi4py import MPI 
import time
import matplotlib.pyplot as plt
import h5py
import cmath

from dedalus import public as de 
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools
import subprocess
from dedalus.tools import post
import matplotlib.colors as colors


# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
vm = 1.046422915


# This is to set the zero velocity to be white
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


# Because the kth order in data is like (0,1,2,3,-3,-2,-1), this is to sort it so that it's in the order (-3,-2,-1,0,1,2,3)
def sort(arr):
    arr2 = arr.copy()
    a = int((len(arr)-1)/2)
    for i in range(0,a):
        arr[i] = arr2[a+i+1]
        arr[a+i] = arr2[i]
    arr[len(arr)-1]=arr2[a]
    return arr







#post.merge_process_files("/users/czhang54/scratch/state_variables25", cleanup=True)
#print(subprocess.check_output("find /users/czhang54/scratch/state_variables21", shell=True).decode())

# Plot
with h5py.File("/users/czhang54/scratch/ret95/ret95_1_c/ret95_1_c_s1.h5", mode='r') as file:

    t = file['scales']['sim_time']
    kth = file['scales']['kth']
    kz = file['scales']['kz']
    Tr = file['scales']['Tr']

    kth1 = kth[:]
    kz1 = kz[:]
    Tr1 = Tr[:]
    kth1 = sort(kth1)
    t1 = t[:]

    print(kth1)
    print(kth1.shape)
    

    for index in range(0,len(t1),5):
        print('index: ',index)
        ul = file['tasks']['ul'][index,:,:,:]
        uh = file['tasks']['uh'][index,:,:,:]
        vl = file['tasks']['vl'][index,:,:,:]
        vh = file['tasks']['vh'][index,:,:,:]
        wl = file['tasks']['wl'][index,:,:,:]
        wh = file['tasks']['wh'][index,:,:,:]
        
        fields = [ul,uh,vl,vh,wl,wh]
        power = np.array([np.multiply(x,np.conj(x)) for x in fields])
        print("power shape", power.shape)
        p_array = np.zeros(Tr1.shape)

        for i in range(len(power)):
            for j in range(len(kth)):
                for k in range(len(kz)):
                    p_array = p_array + power[i][k][j]
        #print('p_array: ',p_array)
        fig = plt.figure()
        p = plt.plot(Tr1,p_array)
        #plt.plot(kth1,y_list,label='slope = -5/3')
        
        #p=plt.pcolormesh(r_,z_,np.transpose(v),cmap='RdBu_r')
        #p=plt.pcolormesh(r_,z_,w0,cmap='RdBu_r')




        plt.xlabel('Tr')
        plt.ylabel('log (kinetic energy)')
        plt.yscale('log')
        #plt.axis([60,177,0,1E-9])
        plt.title("kinetic energy vs $T_r$ at t = {}".format(round(t[index]*vm/0.5,3)))

        #ax.set_aspect('equal')
        #plt.colorbar(p)
        plt.savefig("/users/czhang54/scratch/ret95/power/power_vs_Tr_log_li_1_{}_.png".format(index))
        #plt.savefig("/users/czhang54/data/czhang54/3DPipe/Extractingr\=0/snapshots1/u_1_{}.png".format(index))

        #plt.close()

        
        ##plt.pcolormesh(r_,z_,xi,cmap='PuOr')
        ##plt.colorbar()
        ##plt.title("xi_z field at t = {} s".format(round(t[index],2)))
        ##plt.savefig('/users/czhang54/scratch/diagnostics8/xiz_{}.png'.format(index))
        ##plt.close()


    
    #for index in range(index):
        #plot_tools.plot_bot_3d(u, title="U-velocity field at t = {} s".format(round(t[index],2)), normal_axis=0,normal_index=index)
        #plt.savefig('/users/czhang54/scratch/state_variables7/zero_noise_u_{}.png'.format(index))

        #plot_tools.plot_bot(w,image_axes=(0,3),data_slices=(1,2,3,4),title="W-velocity field at t = {} s".format(round(t[index],2)))
        #plt.savefig('/Users/chenyuzhang/Documents/Codes/3DOutputs/w_{}.png'.format(index))
    
        #plot_tools.plot_bot_3d(v, title="V-velocity field at t = {} s".format(round(t[index],2)), normal_axis=0,normal_index=index)
        #plt.savefig('/users/czhang54/scratch/state_variables7/zero_noise_v_{}.png'.format(index))

        #plot_tools.plot_bot_3d(p, title="Pressure field at t = {} s".format(round(t[index],2)), normal_axis=0,normal_index=index)
        #plt.savefig('/users/czhang54/scratch/state_variables7/zero_noise_p_{}.png'.format(index))

    
    
