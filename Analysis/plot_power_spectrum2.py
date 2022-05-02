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
import sys
import csv

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
v_m = 0.993799968


# Because the kth order in data is like (0,1,2,3,-3,-2,-1), this is to sort it so that it's in the order (-3,-2,-1,0,1,2,3)
def arg(arr):
    arr2 = np.zeros(len(arr)+1)
    a = int((len(arr))/2)
    for i in range(1,a):
        arr2[i] = arr[len(arr)-a+i]
        arr2[a+i] = arr[i]
    arr2[0] = arr[a]
    arr2[len(arr)] = arr[a]
    arr2[a] = arr[0]
    return arr2


j = int(float(sys.argv[1])) # looping j to determine which file to read/plot
#k = sys.argv[2] == 'true' # looping k to determine which axis to plot against
#print(k)
# filelist = ["/users/czhang54/scratch/ret120_1_g/ret120_1_g_s1.h5","/users/czhang54/scratch/ret120_2_g/ret120_2_g_s1.h5",
#             "/users/czhang54/scratch/ret120_3_g/ret120_3_g_s1.h5", "/users/czhang54/scratch/ret120_4_g/ret120_4_g_s1.h5", 
#             "/users/czhang54/scratch/ret120_5_g/ret120_5_g_s1.h5", "/users/czhang54/scratch/ret120_6_g/ret120_6_g_s1.h5",
#             "/users/czhang54/scratch/ret120_7_g/ret120_7_g_s1.h5", "/users/czhang54/scratch/ret120_8_g/ret120_8_g_s1.h5", 
#             "/users/czhang54/scratch/ret120_9_g/ret120_9_g_s1.h5", "/users/czhang54/scratch/ret120_10_g/ret120_10_g_s1.h5",
#            "/users/czhang54/scratch/ret120_11_g/ret120_11_g_s1.h5", "/users/czhang54/scratch/ret120_11_g/ret120_11_g_s2.h5"]
#filelist = ["/users/czhang54/scratch/ret70_6_g/ret70_6_g_s2.h5","/users/czhang54/scratch/ret70_6_g/ret70_6_g_s2.h5","/users/czhang54/scratch/ret81/ret81_3_g/ret81_3_g_s1.h5","/users/czhang54/scratch/ret81/ret81_4_g/ret81_4_g_s1.h5","/users/czhang54/scratch/ret81/ret81_5_g/ret81_5_g_s1.h5","/users/czhang54/scratch/ret81/ret81_6_g/ret81_6_g_s1.h5","/users/czhang54/scratch/ret81/ret81_7_g/ret81_7_g_s1.h5","/users/czhang54/scratch/ret81/ret81_8_g/ret81_8_g_s1.h5","/users/czhang54/scratch/ret81/ret81_9_g/ret81_9_g_s1.h5","/users/czhang54/scratch/ret81/ret81_10_g/ret81_10_g_s1.h5","/users/czhang54/scratch/ret81/ret81_11_g/ret81_11_g_s1.h5","/users/czhang54/scratch/ret81/ret81_12_g/ret81_12_g_s1.h5","/users/czhang54/scratch/ret81/ret81_13_g/ret81_13_g_s1.h5","/users/czhang54/scratch/ret81/ret81_14_g/ret81_14_g_s1.h5","/users/czhang54/scratch/ret81/ret81_15_g/ret81_15_g_s1.h5","/users/czhang54/scratch/ret81/ret81_16_g/ret81_16_g_s1.h5","/users/czhang54/scratch/ret81/ret81_17_g/ret81_17_g_s1.h5","/users/czhang54/scratch/ret81/ret81_18_g/ret81_18_g_s1.h5"]
filelist = ["/users/czhang54/scratch/ret81_r_g/ret81_r_g_s1.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s2.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s3.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s4.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s5.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s6.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s7.h5", "/users/czhang54/scratch/ret81_r_g/ret81_r_g_s8.h5"]
filelist = np.asarray(filelist)
namelist = ["1_s1_{}.png","1_s2_{}.png","1_s3_{}.png","1_s4_{}.png","1_s5_{}.png", "1_s6_{}.png","1_s7_{}.png", "1_s8_{}.png", "9_s1_{}.png", "10_s1_{}.png", "11_s1_{}.png", "11_s2_{}.png", "12_s1_{}.png", "13_s1_{}.png", "14_s1_{}.png", "15_s1_{}.png", "16_s1_{}.pdf", "17_s1_{}.pdf", "18_s1_{}.pdf"]
name = namelist[j]

#ath = k #boolean variable
ath = 3 == 3
print(ath)


#post.merge_process_files("/users/czhang54/scratch/state_variables25", cleanup=True)
#print(subprocess.check_output("find /users/czhang54/scratch/state_variables21", shell=True).decode())

# Plot
#with h5py.File( "/users/czhang54/scratch/ret120_3_g/ret120_3_g_s1.h5", mode='r') as file:
#with h5py.File("/users/czhang54/data/czhang54/3DPipe/Extractingr=0/snapshots2/snapshots2_s1/snapshots2_s1_p0.h5", mode='r') as file:
with h5py.File(filelist[j], mode='r') as file:


    t = file['scales']['sim_time']
    #u1 = file['tasks']['u']
    ##v = file['tasks']['v']
    ##p = file['tasks']['p']
    ##xi_z=file['tasks']['xi_z']
    th = file['scales']['th']['1.0']
    z = file['scales']['z']['1.0']
    r = file['scales']['r']['1.0']
    t1 = t[:]
    print(len(t1))

    th_ = th[:]
    z_ = z[:]
    r_ = r[:]

    l = int(len(th_)/2)
    lz = int(len(z_)/2)
    kth1 = np.arange(-l+1,l,step = 1)   # excluding the zero mode
    kz1 = np.arange(-lz+1,lz,step = 1)
    ath = int((len(kth1)-1)/2)
    az = int((len(kz1)-1)/2)


        
    if 2==2 :
        print('yes')
        for index in range(0,len(t1)):
            p_array = np.zeros(kth1.shape)

            ul = file['tasks']['ul'][index,:,:,:]
            uh = file['tasks']['uh'][index,:,:,:]
            vl = file['tasks']['vl'][index,:,:,:]
            vh = file['tasks']['vh'][index,:,:,:]
            wl = file['tasks']['wl'][index,:,:,:]
            wh = file['tasks']['wh'][index,:,:,:]

            u = ul + uh
            v = vl + vh
            w = wl + wh

            u = u/v_m
            v = v/v_m
            w = w/v_m

            # FFT on the theta axis(axes = 1)
            ut = np.fft.fft(u,axis = 1)
            vt = np.fft.fft(v,axis = 1)
            wt = np.fft.fft(w,axis = 1)

        
            #Take the absolute value of the arrays (so they are real)
            uta = np.abs(ut)
            vta = np.abs(vt)
            wta = np.abs(wt)

            #Take square of each to get power
            utp = np.square(uta)
            vtp = np.square(vta)
            wtp = np.square(wta)

            # Spatially average the power
            p_array = np.zeros(len(th_))
            p_array = np.sum(utp + vtp + wtp, axis = (0, 3))
            # for NL
            # for i in range(0,len(th_)):
            #     for j in range(0,len(z_)):
            #         for k in range(0,len(r_)):
            #             p_array[i] += utp[j,i,k] + vtp[j,i,k] + wtp[j,i,k]
        
            final_p = arg(p_array)
            print('index: ', index)
            #print('final p_array')
            #print(final_p)
            p_copy = final_p[1:len(final_p)-1]
            print('p_copy shape: ', len(p_copy))
            law = [0 if kth == 0 else np.sign(kth)*(np.abs(kth) ** (-5/3)) for kth in kth1]
                  
            fig = plt.figure(figsize = (4.8,3.6))

            #y_list = np.power(kth1,-5/3)                
            ax = fig.add_subplot(111)
            p = plt.plot(kth1[ath:],p_copy[ath:], label = 'spectrum', color = 'red')
            p2 = plt.plot(kth1[ath:], law[ath:], label = 'Kolmogorov scaling', color = 'blue', linewidth = 2, linestyle = 'dashed')
            ax.grid()

            plt.xlabel(r'log($k_\theta$)', fontsize = 15)
            plt.ylabel('log(kinetic energy $(U^2)$)', fontsize = 14)
            plt.yscale('log')
            plt.xscale('log')
            plt.legend()
            #plt.title("kinetic energy spectrum over k_th at t = {:3.1f}".format(t[index]*v_m/0.5))
            plt.title("kinetic energy spectrum over " + r'$k_\theta$'+" at t = {:3.1f}".format(t[index]*v_m/0.5))

            plt.savefig("/users/czhang54/scratch/ret81_r_g/power/power_vs_kth_log_log_"+name.format(index),bbox_inches="tight")
            #plt.savefig("/users/czhang54/scratch/ret120/power/power_vs_kth_log_li_3_s1_{}.pdf".format(index),bbox_inches="tight")
            print('th figure saved')


            plt.close()

    else:
        print('no')
        for index in range(0,len(t1)):
            print("index: ",index)
            p_array = np.zeros(kth1.shape)

            ul = file['tasks']['ul'][index,:,:,:]
            uh = file['tasks']['uh'][index,:,:,:]
            vl = file['tasks']['vl'][index,:,:,:]
            vh = file['tasks']['vh'][index,:,:,:]
            wl = file['tasks']['wl'][index,:,:,:]
            wh = file['tasks']['wh'][index,:,:,:]

            u = ul + uh
            v = vl + vh
            w = wl + wh

            u = u/v_m
            v = v/v_m
            w = w/v_m

            # FFT on the theta axis(axes = 1)
            ut = np.fft.fft(u,axis = 0)
            vt = np.fft.fft(v,axis = 0)
            wt = np.fft.fft(w,axis = 0)

        
            #Take the absolute value of the arrays (so they are real)
            uta = np.abs(ut)
            vta = np.abs(vt)
            wta = np.abs(wt)

            #Take square of each to get power
            utp = np.square(uta)
            vtp = np.square(vta)
            wtp = np.square(wta)

            # Spatially average the power
            p_array = np.zeros(len(z_))
        
            # for NL
            for i in range(0,len(th_)):
                for j in range(0,len(z_)):
                    for k in range(0,len(r_)):
                        p_array[j] += utp[j,i,k] + vtp[j,i,k] + wtp[j,i,k]
        
            final_p = arg(p_array)
            print('index: ', index)
            #print('final p_array')
            #print(final_p)
            p_copy = final_p[1:len(final_p)-1]
            print('p_copy shape: ', len(p_copy))
            law = [0 if kz == 0 else np.sign(kz) * ((np.abs(kz) ** (-5/3)))* (10**7) for kz in kz1]
            #print(law)
                  
            fig = plt.figure(figsize=(4.8,3.6))

            #y_list = np.power(kth1,-5/3)                
            ax = fig.add_subplot(111)
            p = plt.plot(kz1[az:],p_copy[az:], label = 'spectrum', color = 'red')
            plt.plot(kz1[az:],law[az:], label = 'Kolmogorov scaling', color = 'blue', linewidth = 2, linestyle = 'dashed')
            plt.legend()
            ax.grid()
            plt.xlabel('log($k_z$)', fontsize = 15)
            plt.ylabel('log(kinetic energy$(U^2)$)', fontsize = 14)
            plt.yscale('log')
            plt.xscale('log')
            plt.title("kinetic energy spectrum over $k_z$ at t = {:3.1f}".format(t[index]*v_m/0.5))
            plt.savefig("/users/czhang54/scratch/ret81_r_g/power/power_vs_kz_log_log_"+name.format(index),bbox_inches="tight")
            #plt.savefig("/users/czhang54/scratch/ret120/power/power_vs_kz_log_li_3_s1_{}.pdf".format(index),bbox_inches="tight")
            print("z figure saved")

            plt.close()


    
    
