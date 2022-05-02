# Load python packages and Dedalus
import numpy as np
#from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import h5py
import cmath

from dedalus import public as de 
from dedalus.extras import flow_tools
from dedalus.extras import plot_tools
import subprocess
from dedalus.tools import post
import matplotlib.colors as colors
import sys


# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
v_m = 0.7826174748

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


#filelist = ["/Volumes/BigData/chenyu/ret60_nl_g/ret60_nl_g_s1.h5",
#        "/Volumes/BigData/chenyu/ret60_nl_g/ret60_nl_g_s2.h5",
#        "/Volumes/BigData/chenyu/ret60_2_nl_g/ret60_2_nl_g_s1.h5",
#        "/Volumes/BigData/chenyu/ret60_2_nl_g/ret60_2_nl_g_s2.h5",
#        "/Volumes/BigData/chenyu/ret60_2_nl_g/ret60_2_nl_g_s3.h5",
#        "/Volumes/BigData/chenyu/ret60_3_nl_g/ret60_3_nl_g_s1.h5",
#        "/Volumes/BigData/chenyu/ret60_3_nl_g/ret60_3_nl_g_s2.h5",
#        "/Volumes/BigData/chenyu/ret60_3_nl_g/ret60_3_nl_g_s3.h5",
#        "/Volumes/BigData/chenyu/ret60_3_nl_g/ret60_3_nl_g_s4.h5"]

#filelist = ["/users/czhang54/scratch/ret65_512_nl/ret65_10_nl_g/ret65_10_nl_g_s1.h5"]
pre_path = "/users/czhang54/data/czhang54/Saved_results/ret85_512_nl/"
post_path = "_s1.h5"

folderlist = ["ret85_14_nl_g"]

namelist = ["14_s1_{}.png"]
#ath = k #boolean variable
j = int(float(sys.argv[1]))
#print(ath)

# Plot
for i in range(len(folderlist)):
#for i in range(1):
    name = namelist[i]
    folder = folderlist[i]
    path = pre_path + folder + '/' + folder + post_path
    with h5py.File(path, mode='r') as file:

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
        print('z length', len(z_))
        r_ = r[:]

        l = int(len(th_)/2)
        lz = int(len(z_)/2)
        kth1 = np.arange(-l+1,l,step = 1)  # excluding the zero mode
        kz1 = np.arange(-lz+1,lz,step = 1)
        ath = int((len(kth1)-1)/2)
        az = int((len(kz1)-1)/2)

            
        if j==2:
            print('yes')
            for index in range(0,len(t1),5):
                p_array = np.zeros(kth1.shape)

                u = file['tasks']['u'][index,:,:,:]
                v = file['tasks']['v'][index,:,:,:]
                w = file['tasks']['w'][index,:,:,:]

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
            
                # for NL
                for i in range(0,len(th_)):
                    for j in range(0,len(z_)):
                        for k in range(0,len(r_)):
                            p_array[i] += utp[j,i,k] + vtp[j,i,k] + wtp[j,i,k]
            
                final_p = arg(p_array)
                print('index: ', index)
                #print('final p_array')
                #print(final_p)
                p_copy = final_p[1:len(final_p)-1]
                print('p_copy shape: ', len(p_copy))
                law = [0 if kth == 0 else np.sign(kth)*((np.abs(kth) ** (-5/3)))*(10**7) for kth in kth1]
                      
                fig = plt.figure(figsize = (4.8, 3.6))

                #y_list = np.power(kth1,-5/3)
                ax = fig.add_subplot(111)
                p = plt.plot(kth1[ath:],p_copy[ath:], label = 'spectrum',color = 'red', linewidth = 2)
                p2 = plt.plot(kth1[ath:], law[ath:], label = 'Kolmogorov scaling', color = 'blue', linewidth = 2, linestyle = 'dashed')
                ax.grid()

                plt.xlabel(r'$k_\theta$', fontsize = 15)
                plt.ylabel('log (kinetic energy $(U^2)$)', fontsize = 14)
                plt.yscale('log')
                plt.xscale('log')
                plt.legend()
                #plt.title("kinetic energy spectrum over k_th at t = {:3.1f}".format(t[index]*v_m/0.5))
                print('t[index]:', t[index])
                print('t[index]*v_m/0.5:', t[index]*v_m/0.5)
                plt.title("kinetic energy spectrum over " + r'$k_\theta$' + " at t = {:3.1f}".format(t[index]*v_m/0.5), fontsize = 14)
                print("kinetic energy spectrum over " + r'$k_\theta$' + " at t = {:3.1f}".format(t[index]*v_m/0.5))
                #plt.savefig("/users/czhang54/scratch/state_variables21_g/power/power_vs_kth_log_li_"+name.format(index))
                plt.savefig("/users/czhang54/scratch/power_85_nl_" + name[:2] +"/power_vs_kth_log_log" + name.format(index),bbox_inches="tight")
                print('th figure saved')


                plt.close()

        else:
            print('no')
            for index in range(10,len(t1)):
                print("index: ",index)
                p_array = np.zeros(kth1.shape)

                u = file['tasks']['u'][index,:,:,:]
                v = file['tasks']['v'][index,:,:,:]
                w = file['tasks']['w'][index,:,:,:]

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
                law = [0 if kz == 0 else np.sign(kz) * ((np.abs(kz) ** (-5/3)))*(10**7) for kz in kz1]
                      
                fig = plt.figure(figsize = (4.8,3.6))

                #y_list = np.power(kth1,-5/3)
                ax = fig.add_subplot(111)
                plt.plot(kz1[az:],p_copy[az:], label = 'spectrum', color = 'red', linewidth = 2)
                plt.plot(kz1[az:],law[az:], label = 'Kolmogorov scaling', color = 'blue', linewidth = 2, linestyle = 'dashed')
                ax.grid()
                plt.legend()
                plt.xlabel('$k_z$', fontsize = 15)
                plt.ylabel('log (kinetic energy$(U^2)$)', fontsize = 14)
                plt.yscale('log')
                plt.xscale('log')
                #font = matplotlib.font_manager.FontProperties(family='times new roman', size=20)
                #text.set_font_properties(font)

                plt.title("kinetic energy spectrum over $k_z$ at t = {:3.1f}".format(t[index]*v_m/0.5), fontsize = 14)
                plt.savefig("/users/czhang54/scratch/power_85_nl_" + name[:2] +"/power_vs_kz_log_log" + name.format(index),bbox_inches="tight")
                print("z figure saved")

                plt.close()


    
    
