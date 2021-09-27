# Load python packages and Dedalus
import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
import subprocess
import matplotlib.colors as colors
from numba import jit,njit, literal_unroll
import sys
import pickle


start_time = time.time()

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1
N_r = 96
N_th = 256
N_z = 256
total_area = N_r * N_th * N_z
threshold = 0.1
v_m = 0.5776462314



@njit
def calculate_t_f(u, v, w, total_area, threshold):
    tur_eng = np.zeros([N_z, N_th, N_r])
    tot_eng = np.zeros([N_z, N_th, N_r])
    for vel in [u, v, w]:
        vel_avg_pre = np.sum(vel, axis = 1)/N_th              # avg over theta
        vel_avg = np.zeros([N_z, N_th, N_r])
        for i in range(N_th): vel_avg[:, i, :] = vel_avg_pre
        #print(vel_avg.shape)
        vel_tur = vel - vel_avg
        tur_eng += np.power(vel_tur, 2)                   # sum the turbulent energy over the fields
        tot_eng += np.power(vel, 2)                       # sum the total energy over the fields
    
    tur_eng_frac = tur_eng/tot_eng
    
    tur_area = np.sum(tur_eng_frac >= 0.1)
    tur_frac = tur_area/total_area
    
    return tur_frac

#j = int(float(sys.argv[1])) # looping j to determine which file to read/plot

#filelist = ["/Volumes/BigData/chenyu/ret65_nl_g/ret65_nl_g_s1.h5","/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s1.h5","/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s2.h5","/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s3.h5"]
#filelist = ["/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s1.h5","/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s2.h5","/Volumes/BigData/chenyu/ret65_2_nl_g/ret65_2_nl_g_s3.h5"]
filelist = ["/users/czhang54/scratch/ret85_nl/ret85_2_nl_g/ret85_2_nl_g_s1.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_3_nl_g/ret85_3_nl_g_s1.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s1.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s2.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_4_nl_g/ret85_4_nl_g_s3.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s1.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s2.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_5_nl_g/ret85_5_nl_g_s3.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s1.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s2.h5",
        "/users/czhang54/scratch/ret85_nl/ret85_6_nl_g/ret85_6_nl_g_s3.h5"]
filelist = np.asarray(filelist)
namelist = ["2_s1_{}.pdf", "3_s1_{}.pdf","4_s1_{}.pdf","4_s2_{}.pdf","4_s3_{}.pdf","5_s1_{}.pdf","5_s2_{}.pdf","5_s3_{}.pdf","6_s1_{}.png","6_s2_{}.pdf","6_s3_{}.pdf"]
#namelist = ["2_s1_{}.png","2_s2_{}.png","2_s3_{}.png"]
#namelist = ["1_s1_{}.pdf","1_s2_{}.pdf","2_s1_{}.pdf","2_s2_{}.pdf","2_s3_{}.pdf","3_s1_{}.pdf","3_s2_{}.pdf","3_s3_{}.pdf","3_s4_{}.pdf","s4_5_{}.pdf"]
#name = namelist[j]
tur_frac_array = np.array([])
time_array = np.array([])


# Plot
for j in range(len(filelist)):
    name = namelist[j]
    print(name)
    with h5py.File(filelist[j], mode='r') as file:

        t = file['scales']['sim_time']
        r = file['scales']['r']['1.0']
        z = file['scales']['z']['1.0']
        th = file['scales']['th']['1.0']
        t1 = t[:]
        time_array = np.append(time_array, t1)
        th = np.asarray(th)
        z = np.asarray(z)
        r = np.asarray(r)
        r1=r[:]
        z1=z[:]
        
        for index in range(0,len(t1)):
            u = file['tasks']['u'][index,:,:,:]
            v = file['tasks']['v'][index,:,:,:]
            w = file['tasks']['w'][index,:,:,:]

            # Normalize
            for vel in [u, v, w]:
                vel = vel/v_m
            
            tur_frac = calculate_t_f(u, v, w, total_area, threshold)
            tur_frac_array = np.append(tur_frac_array, tur_frac)

time_array = time_array * v_m/(r_out - r_in)
fig = plt.figure(figsize = (4,3))
ax = fig.add_subplot(111)
p = plt.plot(time_array, tur_frac_array, label = 'turbulent fraction', color = 'black')
plt.title('DNS Turbulent Fraction over Time')
plt.xlabel('time')
plt.ylabel('turbulent fraction')
plt.savefig("/users/czhang54/scratch/ret85_nl/ret60_nl_tur_frac.pdf", bbox_inches = 'tight')
plt.close()
print('plt saved')
end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
            
            

    
    
