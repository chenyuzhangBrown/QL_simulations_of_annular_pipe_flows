# Load python packages and Dedalus
from ast import Return
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
threshold  = 0.0003



def get_turb_field(u, v, v_scale,threshold):
    q = np.sqrt(u*u + v*v)
    q_b = q > threshold*v_scale
    turb_field = q_b.astype(int)
    return turb_field

def get_tf_per_t(turb_field, r, th, z):
    r_out = 1
    r_in = 0.5
    dth = th[1] - th[0]
    dz = z[1] - z[0]
    tf_r = np.sum(turb_field, axis = (0,1))
    tf = 0
    for R in range(len(r)):
        if R == len(r) - 1:
            rdr = r[R]*(1 - r[R])
        else:
            rdr = r[R]*(r[R + 1] - r[R])
        tf += tf_r[R]*rdr
    tf = tf*dth*dz/(np.pi * (r_out**2 - r_in**2)*34)
    #return tf*dz*dth*dt/(34*T*(np.pi * (r_out**2 - r_in**2)))
    return tf

def get_filelist(ret):
    filelist = []
    if ret == 70:
        for j in range(7,8):
            if j == 5:
                series = [1,2,3,4,5]
            elif j == 6:
                series = [1,2]
            else:
                series = [1]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret70/ret70_{}_g/ret70_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)
    elif ret == 72:
        for j in range(3,4):
            series = [4]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret72/ret72_{}_g/ret72_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)

    elif ret == 75:
        for j in range(10,20):
            series = [1]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret75/ret75_{}_g/ret75_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)
    elif ret == 79:
        for j in range(16,28):
            if j == 26:
                series = [1,2,3]
            else:
                series = [1]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret79/ret79_{}_g/ret79_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)
                #print(filelist)
    elif ret == 81:
        for j in range(15,25):
            if j == 19:
                series = [1,2]
            else:
                series = [1]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret81/ret81_{}_g/ret81_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)
    elif ret == 95:
        for j in range(6,8):
            series = [1,2]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret95/ret95_{}_g/ret95_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)

    else:
        for j in range(8,16):
            series = [1,2]
            for s in series:
                path = "/users/czhang54/data/czhang54/Saved_results/ret120/ret120_{}_g/ret120_{}_g_s{}.h5".format(j,j,s)
                filelist.append(path)
    return filelist
filelist = ["/users/czhang54/data/czhang54/Saved_results/ret79/ret79_26_g/ret79_26_g_s3.h5"]

#v_scale = -(1/(nu))* dpdz
tur_frac_array = np.array([])
time_array = np.array([])


# Plot
def main(ret, filelist, threshold, dpdz, nu):
    header = ['Time', 'Turbulence Fraction']
    csvfile = open('/users/czhang54/scratch/tf_analysis/tf_vs_t_ret_{}_0_003.csv'.format(ret), 'w')
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for j in range(len(filelist)):
        print(len(filelist), j, flush = True)
        #name = namelist[j]
        print(filelist[j], flush = True)
        f = h5py.File(filelist[j], mode='r')
        t = f['scales']['sim_time'][:]
        r = f['scales']['r']['1.0'][:]
        z = f['scales']['z']['1.0'][:]
        th = f['scales']['th']['1.0'][:]
        

        th = np.array(th)
        z = np.array(z)
        r = np.array(r)
        t = np.array(t)
        v_scale = (1/(nu))* dpdz
        for i, T in enumerate(t):
            ul = f['tasks']['ul'][i,:,:,:]
            uh = f['tasks']['uh'][i,:,:,:]
            vl = f['tasks']['vl'][i,:,:,:]
            vh = f['tasks']['vh'][i,:,:,:]
            u = ul + uh
            v = vl + vh
            tur_field = get_turb_field(ul + uh, vl + vh, v_scale,threshold)
            tur_frac = get_tf_per_t(tur_field, r, th, z)
            writer.writerow([T,tur_frac])
            print(T,tur_frac, flush = True)


ret_list = [72]
dpdz_list = [4*(2*nu*i/0.5)**2 for i in ret_list]

for i, ret in enumerate(ret_list):
    print(ret)
    main(ret, get_filelist(ret), threshold, dpdz_list[i], nu)


end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
            
            

    
    
