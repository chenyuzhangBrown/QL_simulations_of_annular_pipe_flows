# This is a simulation code for annular pipe flows

import numpy as np
import time
import h5py
from mpi4py import MPI 
import matplotlib.pyplot as plt

import dedalus.public as de
from dedalus.extras import flow_tools
import matplotlib.colors as colors
import pathlib

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)


################################################################################################################################################################################

nu = 0.00020704166 # Re_tau = 60
Lp=2*np.pi

# r_in is the radius of the inner pipe, r_out the radius of the outer the pipe, length the length of the pipe

r_in = 0.5
r_out = 1.
length = 34

# pressure gradient across the pipe
#dpdz = -0.0098763844 
#dpdz = -0.03024642527  #Ret = 105
#dpdz = -0.01543184963  #Ret = 75
#dpdz = -0.02475954541   #Ret = 95
#dpdz = -0.01982135353   #Ret = 85
dpdz = -0.01159103372   #Ret = 65
#dpdz = -0.009876383764   #Ret = 60


# Cylindrical coordinates (r, theta, z)
r_basis = de.Chebyshev('r', 100, interval=(r_in, r_out), dealias=3/2)
th_basis = de.Fourier('th',256,interval=(0., 2*np.pi),dealias=3/2)
z_basis = de.Fourier('z', 512, interval=(0., length), dealias=3/2)
domain = de.Domain([z_basis, th_basis, r_basis], grid_dtype=np.float64,mesh=[32,16])




TC = de.IVP(domain, variables=['p', 'u', 'v', 'w', 'ur', 'vr', 'wr'], ncc_cutoff=1e-8)
TC.meta[:]['r']['dirichlet'] = True
TC.parameters['nu'] = nu
TC.parameters['dpdz']=dpdz
TC.parameters['Lp']=Lp
TC.parameters['length']=length
TC.parameters['r_out']=r_out



# Equations of motions. 
TC.add_equation("r*ur + u + dth(v) + r*dz(w) = 0")
TC.add_equation("r*r*dt(u) - r*r*nu*dr(ur) - r*nu*ur - nu*dth(dth(u)) - r*r*nu*dz(dz(u)) + nu*u + 2*nu*dth(v) + r*r*dr(p) = -r*r*u*ur - r*r*w*dz(u) + r*v*v - r*v*dth(u)")
TC.add_equation("r*r*dt(v) - r*r*nu*dr(vr) - r*nu*vr - nu*dth(dth(v)) - r*r*nu*dz(dz(v)) + nu*v - 2*nu*dth(u) + r*dth(p) = -r*r*u*vr - r*r*w*dz(v) - r*v*dth(v) - r*u*v")
TC.add_equation("r*r*dt(w) - r*r*nu*dr(wr) - r*nu*wr - nu*dth(dth(w)) - r*r*nu*dz(dz(w)) + r*r*dz(p) = - r*r*u*wr - r*v*dth(w) - r*r*w*dz(w) - r*r*dpdz")
TC.add_equation("ur - dr(u) = 0")
TC.add_equation("vr - dr(v) = 0")
TC.add_equation("wr - dr(w) = 0")






r = domain.grid(2, scales=domain.dealias)
th = domain.grid(1,scales=domain.dealias)
z = domain.grid(0, scales=domain.dealias)


# boundary conditions
TC.add_bc("right(u) = 0")
TC.add_bc("right(v) = 0")
TC.add_bc("right(w) = 0")
TC.add_bc("left(u) = 0", condition="(nz != 0) or (nth !=0 )")
TC.add_bc("left(v) = 0")
TC.add_bc("left(w) = 0")
#TC.add_bc("left(p) = 0", condition="nz == 0")
TC.add_bc("left(p)=0", condition="(nz == 0) and (nth == 0)")

dt = max_dt = 1.

# Use SBDF4 as the timestepper
ts = de.timesteppers.SBDF4
IVP = TC.build_solver(ts)
logger.info('IVP built')

# Setting stop criteria
IVP.stop_sim_time=1800 
IVP.stop_wall_time=np.inf
IVP.stop_iteration=np.inf

p = IVP.state['p']
u = IVP.state['u']
v = IVP.state['v']
w = IVP.state['w']
ur = IVP.state['ur']
vr = IVP.state['vr']
wr = IVP.state['wr']



#Unfortunately, Gaussian noise on the grid is generally a bad idea: zone-to-zone variations (that is, the highest frequency components) 
# will be amplified arbitrarily by any differentiation. So, let’s filter out those high frequency components using this handy little function:


# Define a filter filed to filter high-frequency noise
def filter_field(field,frac=0.5):
    field.require_coeff_space()
    dom = field.domain
    local_slice = dom.dist.coeff_layout.slices(scales=dom.dealias)
    coeff = []
    for n in dom.global_coeff_shape:
        coeff.append(np.linspace(0,1,n,endpoint=False))
    cc = np.meshgrid(*coeff, indexing = 'ij')

    field_filter = np.zeros(dom.local_coeff_shape,dtype='bool')
    for i in range(dom.dim):
        field_filter = field_filter | (cc[i][local_slice] > frac)
    field['c'][field_filter] = 0j


u.set_scales(domain.dealias,keep_data=False) # initialize u on dealiased domain
p.set_scales(domain.dealias,keep_data=False)
v.set_scales(domain.dealias,keep_data=False)
w.set_scales(domain.dealias,keep_data=False)

CFL = flow_tools.CFL(IVP, initial_dt=1e-5, cadence=1, safety=0.3,
                     max_change=1.6,threshold=0.15)
CFL.add_velocities(('u', 'v', 'w'))


# If there is no restarting checkpoint
# if not pathlib.Path('/users/czhang54/scratch/checkpoints85_nl/checkpoints85_nl.h5').exists():


#    # Domain shape and slices
#    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
#    slices=domain.dist.grid_layout.slices(scales=domain.dealias)

#    # Initialize random noise globally (Gaussian)
#    rand = np.random.RandomState(seed=70)
#    noise = rand.standard_normal(gshape)[slices]

#    u_array=np.zeros(gshape)[slices]
#    v_array=np.zeros(gshape)[slices]
#    w_array=np.zeros(gshape)[slices]
#    p_array=np.zeros(gshape)[slices]

#    r = domain.grid(2, scales=domain.dealias)
#    z = domain.grid(0, scales=domain.dealias)


#    # The analytical solution of the laminar flow in the z-direction
#    w_analytic=(1/(4*nu))*(-dpdz)*(r_out**2-r**2+((r_out**2-r_in**2)/np.log(r_in/r_out))*np.log(r_out/r))

#    # compute the first derivatives
#    w['g'] = w_analytic


#    # Create the r_component of the noise field
#    noise_r = domain.new_field(name='noise_r')
#    noise_r.set_scales(domain.dealias,keep_data=False)
#    noise_r['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_r)



#    # Create the theta_component of the noise field
#    noise_theta = domain.new_field(name='noise_theta')
#    noise_theta.set_scales(domain.dealias,keep_data=False)
#    noise_theta['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_theta)




#    # Create the z_component of the noise field
#    noise_z = domain.new_field(name='noise_z')
#    noise_z.set_scales(domain.dealias,keep_data=False)
#    noise_z['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_z)



# # In order for the incompressibility equation to account for the effect of the noise, use the curl of the noise as the initial random kick, so there is zero divergence.

#    # the r-component of curl of noise
#    cnoise_r = domain.new_field(name = 'cnoise_r')
#    cnoise_r.set_scales(domain.dealias,keep_data=False)
#    cnoise_r['g'] = noise_z.differentiate('th')['g']/r - noise_theta.differentiate('z')['g']

#    # the theta-component of curl of noise
#    cnoise_th = domain.new_field(name = 'cnoise_th')
#    cnoise_th.set_scales(domain.dealias,keep_data=False)
#    cnoise_th['g'] = noise_r.differentiate('z')['g'] - noise_z.differentiate('r')['g']


#    # the z-component of curl of noise
#    cnoise_z = domain.new_field(name = 'cnoise_z')
#    cnoise_z.set_scales(domain.dealias,keep_data=False)
#    cnoise_z['g'] = noise_theta['g']/r + noise_theta.differentiate('r')['g'] - noise_r.differentiate('th')['g']/r

#    # Add noise to velocity fields
#    w['g'] += 2*cnoise_z['g']
#    u['g'] += 2*cnoise_r['g']
#    v['g'] += 2*cnoise_th['g']

#    w.differentiate('r', out = wr)
#    u.differentiate('r', out = ur)
#    v.differentiate('r', out = vr)

#    dt = CFL.compute_dt()
#    #dt = 1E-8

#    fh_mode = 'overwrite'


# Restart
write, last_dt = IVP.load_state('/glade/scratch/chenyuz/checkpoints65_10_nl/checkpoints65_10_nl_s1.h5', -1)

#Timestepping and output
fh_mode = 'append'

dt = last_dt*0.5



# Take snapshots
   # set up "state_variables" folder for HDF5 file to write to
   # sim_dt sets frequency of writing tasks to the file
      # sim_dt=2 -> write out all snapshots1 tasks every 2 seconds of simulated time
      # max_writes limits total number of writes per HDF5 file to keep file size reasonable
snapshots1 = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/ret65_nl/ret65_11_nl_g',sim_dt = 1,max_writes=100)
   # save full state variable in grid space ('g') for u,v
snapshots1.add_task(IVP.state['u'],layout='g',name='u')
snapshots1.add_task(IVP.state['v'],layout='g',name='v')
snapshots1.add_task(IVP.state['w'],layout='g',name='w')
snapshots1.add_task(IVP.state['p'],layout='g',name='p')

snapshots2 = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/ret65_nl/ret65_11_nl_c',sim_dt = 1,max_writes=100)

snapshots2.add_task(IVP.state['u'],layout='c',name='uc')
snapshots2.add_task(IVP.state['v'],layout='c',name='vc')
snapshots2.add_task(IVP.state['w'],layout='c',name='wc')
snapshots1.add_task(IVP.state['p'],layout='c',name='pc')

#snapshots1.add_task('flux',layout='g',name='flux')


# analysis
check = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/checkpoints65_11_nl', sim_dt=1, max_writes=100, mode = fh_mode)
check.add_system(IVP.state)




# Flow properties
   # allows printing of min/max of flow properties to the logger during the main loop
flow=flow_tools.GlobalFlowProperty(IVP)
   # add state variables as flow properties
flow.add_property('p', name='presurre')
flow.add_property('u', name='u-velocity')
flow.add_property('v',name='v-velocity')
flow.add_property('w',name='w-velocity')

start_time = time.time()
#f = open('filesize.txt','w')

try:
    logger.info('Starting loop')
    while IVP.ok:
        IVP.step(dt) # this step right here is the money-maker; advances the simulation forward in time by dt seconds
        if (IVP.iteration-1)%100==0: # print status message to logger every 100 iterations
            logger.info('Iteration:{}, Time: {:e}, dt:{:e}'.format(IVP.iteration,IVP.sim_time,dt))
            logger.info('Max/Min U={:f},{:f}'.format(flow.max('u-velocity'),flow.min('u-velocity')))
            logger.info('Max/Min V={:f},{:f}'.format(flow.max('v-velocity'),flow.min('v-velocity')))
            logger.info('Max/Min W={:f},{:f}'.format(flow.max('w-velocity'),flow.min('w-velocity')))
            #f.write('Iteration:{}, Time: {:e}, dt:{:e}'.format(IVP.iteration,IVP.sim_time,dt))
            #f.write(os.popen('ls -lh /tmp').read()+'\n')
            dt=CFL.compute_dt() # calculates time step based on CFL condition
            #dt=1E-8
except:
   logger.error('Exception raised, triggering end of main loop.')
   raise
finally: # print summary statistics to logger on completion of simulation
   end_time=time.time()
   logger.info('Iterations:{}'.format(IVP.iteration))
   logger.info('Sim end time:{:f}'.format(IVP.sim_time))
   logger.info('Run time: {:.2f} sec'.format(end_time-start_time))
   logger.info('Run time: {:f} cpu-hr'.format((end_time-start_time)/(60.*60.)*domain.dist.comm_cart.size))
   

#f.close()



for task in analysis_tasks:
    logger.info(task.base_path)
    post.merge_analysis(task.base_path)
