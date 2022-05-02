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
threshold  = 0.0003
#dpdz_list = [4*(2*nu*i/0.5)**2 for i in [70,75,79,81,120]]
dpdz_list = [4*(2*nu*i/0.5)**2 for i in [95]]


def get_turb_field(u, v, v_scale,threshold):
    q = np.sqrt(u*u + v*v)
    q_b = q > threshold*v_scale
    turb_field = q_b.astype(int)
    return turb_field

def get_tf(turb_field, r, th, z, t):
    r_out = 1
    r_in = 0.5
    dth = th[1] - th[0]
    dz = z[1] - z[0]
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    tf_r = np.sum(turb_field, axis = (0,1,2))
    tf = 0
    for R in range(len(r)):
        if R == len(r) - 1:
            rdr = r[R]*(1 - r[R])
        else:
            rdr = r[R]*(r[R + 1] - r[R])
        tf += tf_r[R]*rdr
    tf = tf*dth*dz/(len(t)*np.pi * (r_out**2 - r_in**2)*34)
    #return tf*dz*dth*dt/(34*T*(np.pi * (r_out**2 - r_in**2)))
    return tf

# Calculate average tf
def avg_tf(filelist, threshold, dpdz):
    tur_frac_array = np.array([])
    for j in range(len(filelist)):
        print(len(filelist), j, flush = True)
        #name = namelist[j]
        print(filelist[j], flush = True)
        f = h5py.File(filelist[j], mode='r')
        t = f['scales']['sim_time']
        r = f['scales']['r']['1.0']
        z = f['scales']['z']['1.0']
        th = f['scales']['th']['1.0']
        

        th = np.array(th)
        z = np.array(z)
        r = np.array(r)
        t = np.array(t)
        

        ul = f['tasks']['ul'][:,:,:,:]
        uh = f['tasks']['uh'][:,:,:,:]
        vl = f['tasks']['vl'][:,:,:,:]
        vh = f['tasks']['vh'][:,:,:,:]
        u = ul + uh
        v = vl + vh
        #dpdz = f['scales']['dpdz']
        print('imported, time = {}'.format(time.time() - start_time), flush = True)
        v_scale = (1/(nu))* dpdz
        #print('v_scale, time = {}'.format(time.time() - start_time), v_scale, flush = True)
        #ul = np.array(ul)
        print('max u', u.max(), flush = True)
        print('max v', v.max(), flush = True)
        #print('ul, time = {}'.format(time.time() - start_time), flush = True)
        #uh = np.array(uh)
        #print('uh, time = {}'.format(time.time() - start_time), flush = True)
        #vl = np.array(vl)
        #print('vl, time = {}'.format(time.time() - start_time), flush = True)
        #vh = np.array(vh)
        #print('vh,time = {}'.format(time.time() - start_time), flush = True)
        print('wait for t_field', flush = True)
        t_field = get_turb_field(ul + uh, vl + vh, v_scale,threshold)
        tur_frac = get_tf(t_field, r, th, z, t)
        tur_frac_array = np.append(tur_frac_array, tur_frac)
    tur_frac_avg = np.sum(tur_frac_array)/len(tur_frac_array)
    return tur_frac_avg

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
#filelist_list = [filelist_70, filelist_75, filelist_79, filelist_80, 
#                filelist_81, filelist_81, filelist_95, filelist_120]
#ret_list = [70, 75, 79, 81, 120]
ret_list = [95]
avg_tur_frac_list = []
for i, ret in enumerate(ret_list):
    print('ret: ', ret, i, flush = True)
    avg_tur_frac_list.append(avg_tf(get_filelist(ret), threshold, dpdz_list[i]))
    print(avg_tur_frac_list)
    # append dpdz here


# save data
header = ['dpdz','Tubulence Fraction']
data = np.transpose([dpdz_list, avg_tur_frac_list])
with open('/users/czhang54/scratch/tur_frac_list.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(data)


fig = plt.figure(figsize = (4.8,3.6))
ax = fig.add_subplot(111)
print('avg_tur_frac_list', avg_tur_frac_list)
print('dpdz_list', dpdz_list)
p = plt.plot(dpdz_list, avg_tur_frac_list, label = 'time-averaged turbulent fraction', color = 'black')
ax.grid()
plt.title('Time-averaged Turbulent Fraction over Constant Pressure Gradient')
plt.xlabel('Constant Pressure Gradient')
plt.ylabel('Time-averaged Turbulent Fraction')
plt.savefig("/users/czhang54/scratch/ql_tur_frac.pdf", bbox_inches = "tight")
plt.close()
print('plt saved')
end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
            
            

    
    
