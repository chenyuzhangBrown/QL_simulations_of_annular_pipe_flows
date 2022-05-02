# This file is created to test the functions of the code. The parameter setting is based on Motlagh's paper about concentric pipe flow

import numpy as np
import time
import h5py
from mpi4py import MPI 
#import matplotlib.pyplot as plt


import dedalus.public as de
from dedalus.extras import flow_tools
import pathlib

import logging
root = logging.root
for h in root.handlers:
    h.setLevel("INFO")

logger = logging.getLogger(__name__)
import time


start_time = time.time()
# Define a different state-loading function which takes in NL data, average it over theta, and feed the corresponding fields to QL states
def load_my_state(self, path, index=-1):
        """
        Load state from HDF5 file.
        Parameters
        ----------
        path : str or pathlib.Path
            Path to Dedalus HDF5 savefile
        index : int, optional
            Local write index (within file) to load (default: -1)
        Returns
        -------
        write : int
            Global write number of loaded write
        dt : float
            Timestep at loaded write
        
        #path = pathlib.Path(path)
        """
        path = path
        logger.info("Loading solver state from: {}".format(path))
        #with h5py.File(str(path), mode='r') as file:
        with h5py.File('/users/czhang54/scratch/checkpoints9/restart.h5',mode = 'r') as file:
            # Load solver attributes
            write = file['scales']['write_number'][index]
            try:
                dt = file['scales']['timestep'][index]
            except KeyError:
                dt = None
            IVP.iteration = IVP.initial_iteration = file['scales']['iteration'][index]
            IVP.sim_time = IVP.initial_sim_time = file['scales']['sim_time'][index]
            # Log restart info
            logger.info("Loading iteration: {}".format(IVP.iteration))
            logger.info("Loading write: {}".format(write))
            logger.info("Loading sim time: {}".format(IVP.sim_time))
            logger.info("Loading timestep: {}".format(dt))
            #th = file['scales']['th']['1.0']
            #th1 = th[:]
            # Load fields
            print('IVP state fields: ', IVP.state.fields)
            for field in IVP.state.fields:
                dset = file['tasks'][field.name[0]]
                #print(field.name[0])
                # Find matching layout
                for layout in IVP.domain.dist.layouts:
                    if np.allclose(layout.grid_space, dset.attrs['grid_space']):
                        break
                else:
                    raise ValueError("No matching layout")

                # Set scales to match saved data
                scales = dset.shape[1:] / layout.global_shape(scales=1)

                 # getting the theta average of the field
                v_array = np.zeros(dset[:,:,0,:].shape)
                dset_copy = np.zeros(dset.shape)
                for i in range(0,255):
                    v_array += file['tasks'][field.name[0]][:,:,i,:]
                v_array = v_array/256

                # Feed the average to the zero mode fields
                if field.name in ['wl','vl','ul','pl','wlr','vlr','ulr']:
                    print('field name: ',field.name)
                    for i in range(0,255):
                        #print(i)
                        dset_copy[:,:,i,:] = v_array
                        #print('avg fields at:', dset_copy[:,:,i,:])
                    #print('dset_copy',dset_copy)
                else:
                    #print('field name:', field.name)
                    # feed the rest to the high frequency fields
                    for i in range(0,255):
                        #print(i)
                        dset_copy[:,:,i,:] = dset[:,:,i,:]-v_array
                        #print('h fields at: ', dset_copy[:,:,i,:])


                scales[~layout.grid_space] = 1
                # Extract local data from global dset
                dset_slices = (index,) + layout.slices(tuple(scales))
                #print(index,)
                local_dset = dset_copy[dset_slices]
                # Copy to field
                field_slices = tuple(slice(n) for n in local_dset.shape)
                field.set_scales(scales, keep_data=False)
                field[layout][field_slices] = local_dset
                field.set_scales(IVP.domain.dealias, keep_data=True)
        return write, dt


#Re = 8900.       This is the bulk mean Re. The paper uses nu in the equation
nu=0.00020704166     #original setting
#nu = 1.
Lp=2*np.pi

#r_in = 1.
r_in = 0.5
#r_out = 2.
r_out = 1.
#length = 68
length = 34.

#dpdz=-33800
#dpdz=-0.0098763844  Ret = 60
#dpdz = -0.01344285567   # Ret = 70
#dpdz = -0.05377142271   # Ret = 140
#dpdz = -0.01543184963   # Ret = 75
#dpdz = -0.01755801558   # Ret = 80
#dpdz = -0.01799970941	 # Ret = 81
#dpdz = -0.01712180863	 # Ret = 79
#dpdz = -0.01982135352   # Ret = 85
#dpdz = -0.02222186347    # Ret = 90
dpdz = -0.03950553506     # Ret = 120
#dpdz = -0.01159103372    # Ret = 65

# bases
r_basis = de.Chebyshev('r', 100, interval=(r_in, r_out), dealias=3/2)
th_basis = de.Fourier('th',128,interval=(0., 2*np.pi),dealias=3/2)
z_basis = de.Fourier('z', 512, interval=(0., length), dealias=3/2)
domain = de.Domain([z_basis, th_basis, r_basis], grid_dtype=np.float64,mesh=[16,32])

#r_basis = de.Chebyshev('r', 46, interval=(r_in, r_out), dealias=3/2)
#th_basis = de.Fourier('th',64,interval=(0., 2*np.pi),dealias=3/2)
#z_basis = de.Fourier('z', 128, interval=(0., length), dealias=3/2)
#domain = de.Domain([z_basis, th_basis, r_basis], grid_dtype=np.float64,mesh=[2,2])


TC = de.IVP(domain, variables=['pl', 'ph', 'ul', 'vl', 'wl','uh','vh','wh', 'ulr', 'vlr', 'wlr','uhr','vhr','whr'], ncc_cutoff=1e-8)
TC.meta[:]['r']['dirichlet'] = True
TC.parameters['nu'] = nu
TC.parameters['dpdz']=dpdz
TC.parameters['Lp']=Lp
#TC.parameters['length']=length
TC.parameters['r_out']=r_out


#TC.substitutions['ul_bar']="integ(ul,'th','z')/(height*Lp)"




#TC.substitutions['wl_bar']="integ(wl,'th','z')/(height*Lp)"
#TC.substitutions['p_bar']="integ(p,'th','z')/(height*Lp)"

# Define vorticies
#TC.substitutions['xi_r']="(1/r)*dth(w)-dz(v)"
#TC.substitutions['xi_th']="dz(u)-wr"
#TC.substitutions['xi_z']="v/r+vr-(1/r)*dth(u)"


# only change nth, we're only averaging over theta direction
#TC.add_equation("ulr + (1/r)*ul + (1/r)*dth(vl) + dz(wl) = 0",condition='(nth==0)')
#TC.add_equation("dt(ul) - nu*dr(ulr) - (1/r)*nu*ulr - (1/r)*(1/r)*nu*dth(dth(ul)) - nu*dz(dz(ul)) + (1/r)*(1/r)*nu*ul + 2*(1/r)*(1/r)*nu*dth(vl) + dr(pl) = -ul*ulr - wl*dz(ul) + (1/r)*vl*vl - (1/r)*vl*dth(ul) - uh*uhr - wh*dz(uh) + (1/r)*vh*vh - (1/r)*vh*dth(uh)",condition='(nth==0)')
#TC.add_equation("dt(vl) - nu*dr(vlr) - (1/r)*nu*vlr - (1/r)*(1/r)*nu*dth(dth(vl)) - nu*dz(dz(vl)) + (1/r)*(1/r)*nu*vl - 2*(1/r)*(1/r)*nu*dth(ul) + (1/r)*dth(pl) = -ul*vlr - wl*dz(vl) - (1/r)*vl*dth(vl) - (1/r)*ul*vl - uh*vhr - wh*dz(vh) - (1/r)*vh*dth(vh) - (1/r)*uh*vh",condition='(nth==0)')
#TC.add_equation("dt(wl) - nu*dr(wlr) - (1/r)*nu*wlr - (1/r)*(1/r)*nu*dth(dth(wl)) - nu*dz(dz(wl)) + dz(pl) = - ul*wlr - (1/r)*vl*dth(wl) - wl*dz(wl) - uh*whr - (1/r)*vh*dth(wh) - wh*dz(wh) - dpdz",condition='(nth==0)')

TC.add_equation("r*ulr + ul + dth(vl) + r*dz(wl) = 0",condition='(nth==0)')
TC.add_equation("r*r*dt(ul) - r*r*nu*dr(ulr) - r*nu*ulr - nu*dth(dth(ul)) - r*r*nu*dz(dz(ul)) + nu*ul + 2*nu*dth(vl) + r*r*dr(pl) = -r*r*ul*ulr - r*r*wl*dz(ul) + r*vl*vl - r*vl*dth(ul) - r*r*uh*uhr - r*r*wh*dz(uh) + r*vh*vh - r*vh*dth(uh)",condition='(nth==0)')

TC.add_equation("r*r*dt(vl) - r*r*nu*dr(vlr) - r*nu*vlr - nu*dth(dth(vl)) - r*r*nu*dz(dz(vl)) + nu*vl - 2*nu*dth(ul) + r*dth(pl) = -r*r*ul*vlr - r*r*wl*dz(vl) - r*vl*dth(vl) - r*ul*vl - r*r*uh*vhr - r*r*wh*dz(vh) - r*vh*dth(vh) - r*uh*vh",condition='(nth==0)')
TC.add_equation("r*r*dt(wl) - r*r*nu*dr(wlr) - r*nu*wlr - nu*dth(dth(wl)) - r*r*nu*dz(dz(wl)) + r*r*dz(pl) = - r*r*ul*wlr - r*vl*dth(wl) - r*r*wl*dz(wl) - r*r*uh*whr - r*vh*dth(wh) - r*r*wh*dz(wh) - r*r*dpdz",condition='(nth==0)')


TC.add_equation("ulr - dr(ul) = 0",condition='(nth==0)')
TC.add_equation("vlr - dr(vl) = 0",condition='(nth==0)')
TC.add_equation("wlr - dr(wl) = 0",condition='(nth==0)')

#problem.add_equation('r**2*dt(u)+r**2*dz(p)-(1/Re)*(r*ur+r**2*dr(ur)+dphi(dphi(u))+r**2*dz(dz(u)))=-r*w*dphi(u)-r**2*u*dz(u)-r**2*v*ur-r**2*dpdz')
#problem.add_equation('r*r*dt(v)+r*r*dr(p)-(1/Re)*(r*vr+r*r*dr(vr)+dphi(dphi(v))+r*r*dz(dz(v))-v-2*dphi(w))=-r*w*dphi(v)-r*r*u*dz(v)-r*r*v*vr+r*w*w')
#problem.add_equation('r*r*dt(w)+r*dphi(p)-(1/Re)*(r*wr+r*r*dr(wr)+dphi(dphi(w))+r*r*dz(dz(w))-w+2*dphi(v))=-r*w*dphi(w)-r*r*u*dz(w)-r*r*v*wr-r*w*v')

# High-frequency components are zero for nr=0 and nz=0
TC.add_equation('ph=0',condition='(nth==0)')
TC.add_equation('uh=0',condition='(nth==0)')
TC.add_equation('vh=0',condition='(nth==0)')
TC.add_equation('wh=0',condition='(nth==0)')
TC.add_equation('uhr=0',condition='(nth==0)')
TC.add_equation('vhr=0',condition='(nth==0)')
TC.add_equation('whr=0',condition='(nth==0)')

# Add equations for the perturbation flow, uh, vh, wh, ph for nth != 0 or nz != 0
TC.add_equation("r*uhr + uh + dth(vh) + r*dz(wh) = 0",condition='(nth != 0)')
TC.add_equation("r*r*dt(uh) - r*r*nu*dr(uhr) - r*nu*uhr - nu*dth(dth(uh)) - r*r*nu*dz(dz(uh)) + nu*uh + 2*nu*dth(vh) + r*r*dr(ph) = -r*r*ul*uhr - r*r*wl*dz(uh) + r*vl*vh - r*vl*dth(uh) - r*r*uh*ulr - r*r*wh*dz(ul) + r*vl*vh - r*vh*dth(ul)",condition='(nth != 0)')
TC.add_equation("r*r*dt(vh) - r*r*nu*dr(vhr) - r*nu*vhr - nu*dth(dth(vh)) - r*r*nu*dz(dz(vh)) + nu*vh - 2*nu*dth(uh) + r*dth(ph) = -r*r*ul*vhr - r*r*wl*dz(vh) - r*vl*dth(vh) - r*uh*vl - r*r*uh*vlr - r*r*wh*dz(vl) - r*vh*dth(vl) - r*ul*vh",condition='(nth != 0)')
TC.add_equation("r*r*dt(wh) - r*r*nu*dr(whr) - r*nu*whr - nu*dth(dth(wh)) - r*r*nu*dz(dz(wh)) + r*r*dz(ph) = - r*r*ul*whr - r*vl*dth(wh) - r*r*wl*dz(wh) - r*r*uh*wlr - r*vh*dth(wl) - r*r*wh*dz(wl)",condition='(nth != 0)')

TC.add_equation("uhr - dr(uh) = 0",condition='(nth != 0)')
TC.add_equation("vhr - dr(vh) = 0",condition='(nth != 0)')
TC.add_equation("whr - dr(wh) = 0",condition='(nth != 0)')

# Low-frequency components are zero for nth !=0 or nz != 0
TC.add_equation('pl=0',condition='(nth != 0)')
TC.add_equation('ul=0',condition='(nth != 0)')
TC.add_equation('vl=0',condition='(nth != 0)')
TC.add_equation('wl=0',condition='(nth != 0)')
TC.add_equation('ulr=0',condition='(nth != 0)')
TC.add_equation('vlr=0',condition='(nth != 0)')
TC.add_equation('wlr=0',condition='(nth != 0)')




r = domain.grid(2, scales=domain.dealias)
th = domain.grid(1,scales=domain.dealias)
z = domain.grid(0, scales=domain.dealias)



# boundary conditions
# boundary conditions for nth == 0 mode
TC.add_bc("right(ul) = 0",condition='(nth == 0)')# replaced with pl gauge choice
TC.add_bc("right(vl) = 0",condition='(nth == 0)' )
TC.add_bc("right(wl) = 0",condition='(nth == 0)') 
TC.add_bc("left(ul) = 0", condition="(nth ==0 ) and (nz != 0)")
TC.add_bc("left(vl) = 0", condition="(nth ==0 )")
TC.add_bc("left(wl) = 0", condition="(nth ==0 )")
TC.add_bc("left(pl) = 0", condition="(nth == 0) and (nz == 0)")



# boundary conditions for nth != 0 mode
TC.add_bc("right(uh) = 0",condition='(nth != 0)')
TC.add_bc("right(vh) = 0",condition='(nth != 0)')
TC.add_bc("right(wh) = 0",condition='(nth != 0)')
TC.add_bc("left(uh) = 0", condition="(nth !=0 )")
TC.add_bc("left(vh) = 0", condition="(nth !=0 )")
TC.add_bc("left(wh) = 0", condition="(nth !=0 )")
#TC.add_bc("integ_r(ph)=0", condition="(nth != 0) and (nz != 0)")




#dt = max_dt = 1e-3
#omega1 = TC.parameters['v_l']/r_in<D-d>
#period = 2*np.pi/omega1

ts = de.timesteppers.SBDF4
IVP = TC.build_solver(ts)
logger.info('IVP built')
#IVP.stop_sim_time = 15.*period
#IVP.stop_wall_time = np.inf
#IVP.stop_iteration = 10000000

# Setting stop criteria
IVP.stop_sim_time=5000
IVP.stop_wall_time=np.inf
IVP.stop_iteration=np.inf

pl = IVP.state['pl']
ul = IVP.state['ul']
vl = IVP.state['vl']
wl = IVP.state['wl']
ulr = IVP.state['ulr']
vlr = IVP.state['vlr']
wlr = IVP.state['wlr']
ph = IVP.state['ph']
uh = IVP.state['uh']
vh = IVP.state['vh']
wh = IVP.state['wh']
uhr = IVP.state['uhr']
vhr = IVP.state['vhr']
whr = IVP.state['whr']



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


ul.set_scales(domain.dealias,keep_data=False) # initialize u on dealiased domain
pl.set_scales(domain.dealias,keep_data=False)
vl.set_scales(domain.dealias,keep_data=False)
wl.set_scales(domain.dealias,keep_data=False)
uh.set_scales(domain.dealias,keep_data=False) # initialize u on dealiasd domain
ph.set_scales(domain.dealias,keep_data=False)
vh.set_scales(domain.dealias,keep_data=False)
wh.set_scales(domain.dealias,keep_data=False)



CFL = flow_tools.CFL(IVP, initial_dt=1e-5, cadence=1, safety=0.3,
                     max_change=1.6,threshold=0.15)

CFL.add_velocities(('ul+uh', 'vl+vh', 'wl+wh'))


# If there is no restarting checkpoint
#if not pathlib.Path('/glade/scratch/chenyuz/checkpoints1_2/checkpoints1_2_s1.h5').exists():
#
#
#    # Domain shape and slices
#    gshape = domain.dist.grid_layout.global_shape(scales=domain.dealias)
#    slices=domain.dist.grid_layout.slices(scales=domain.dealias)
#
#    # Initialize random noise globally (Gaussian)
#    rand = np.random.RandomState(seed = 70)
#    noise = rand.standard_normal(gshape)[slices]
#
#    ul_array=np.zeros(gshape)[slices]
#    vl_array=np.zeros(gshape)[slices]
#    wl_array=np.zeros(gshape)[slices]
#    pl_array=np.zeros(gshape)[slices]
#    uh_array=np.zeros(gshape)[slices]
#    vh_array=np.zeros(gshape)[slices]
#    wh_array=np.zeros(gshape)[slices]
#    ph_array=np.zeros(gshape)[slices]
#
#   
#    r = domain.grid(2, scales=domain.dealias)
#    z = domain.grid(0, scales=domain.dealias)
#    
#    #r1 = r[0,0,:]
#
#    # The analytical solution of the laminar flow in the z-direction
#    w_analytic=(1/(4*nu))*(-dpdz)*(r_out**2-r**2+((r_out**2-r_in**2)/np.log(r_in/r_out))*np.log(r_out/r))
#
#    # compute the first derivatives
#    wl['g'] = w_analytic
#
#
#    # Create the r_component of the noise field
#    noise_r = domain.new_field(name='noise_r')
#    noise_r.set_scales(domain.dealias,keep_data=False)
#    noise_r['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_r)
#
#
#
#    # Create the theta_component of the noise field
#    noise_theta = domain.new_field(name='noise_theta')
#    noise_theta.set_scales(domain.dealias,keep_data=False)
#    noise_theta['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_theta)
#
#
#
#
#    # Create the z_component of the noise field
#    noise_z = domain.new_field(name='noise_z')
#    noise_z.set_scales(domain.dealias,keep_data=False)
#    noise_z['g'] = 0.01*noise*np.sin(np.pi*(r - r_in)/(r_out-r_in))
#    filter_field(noise_z)
#
#
#
## In order for the incompressibility equation to account for the effect of the noise, use the curl of the noise as the initial random kick, so there is zero divergence.
#
#    # the r-component of curl of noise
#    cnoise_r = domain.new_field(name = 'cnoise_r')
#    cnoise_r.set_scales(domain.dealias,keep_data=False)
#    cnoise_r['g'] = noise_z.differentiate('th')['g']/r - noise_theta.differentiate('z')['g']
#
#    # the theta-component of curl of noise
#    cnoise_th = domain.new_field(name = 'cnoise_th')
#    cnoise_th.set_scales(domain.dealias,keep_data=False)
#    cnoise_th['g'] = noise_r.differentiate('z')['g'] - noise_z.differentiate('r')['g']
#
#
#    # the z-component of curl of noise
#    cnoise_z = domain.new_field(name = 'cnoise_z')
#    cnoise_z.set_scales(domain.dealias,keep_data=False)
#    cnoise_z['g'] = noise_theta['g']/r + noise_theta.differentiate('r')['g'] - noise_r.differentiate('th')['g']/r
#
#    # Add noise to velocity fields
#    wl['g'] += cnoise_z['g']
#    wh['g'] += cnoise_z['g']
#    ul['g'] += cnoise_r['g']
#    uh['g'] += cnoise_r['g']
#    vl['g'] += cnoise_th['g']
#    vh['g'] += cnoise_th['g']
#   
#
#
#
#    wl.differentiate('r', out = wlr)
#    wh.differentiate('r', out = whr)
#
#    ul.differentiate('r', out = ulr)
#    uh.differentiate('r', out = uhr)
#
#    vl.differentiate('r', out = vlr)
#    vh.differentiate('r', out = vhr)
#
#
#    dt = CFL.compute_dt()
#    #dt = 1E-5
#    
#    fh_mode = 'overwrite'
#
#
#else:
    # Restart
write, last_dt = IVP.load_state('/glade/scratch/chenyuz/checkpoints1_2/checkpoints1_2_s1.h5', -1)
    #write, last_dt = load_my_state('/users/czhang54/scratch/checkpoints6/restart.h5', -1)


    # Timestepping and output
fh_mode = 'append'

dt = last_dt*0.5




# Take snapshots
   # set up "state_variables" folder for HDF5 file to write to
   # sim_dt sets frequency of writing tasks to the file
      # sim_dt=2 -> write out all snapshots1 tasks every 2 seconds of simulated time
      # max_writes limits total number of writes per HDF5 file to keep file size reasonable
snapshots1 = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/ret120/ret120_1_g',sim_dt = 0.5 ,max_writes=120)
   # save full state variable in grid space ('g') for u
snapshots1.add_task('ul',layout='g',name='ul')
snapshots1.add_task('vl',layout='g',name='vl')
snapshots1.add_task('wl',layout='g',name='wl')
snapshots1.add_task('pl',layout='g',name='pl')

snapshots1.add_task('uh',layout='g',name='uh')
snapshots1.add_task('vh',layout='g',name='vh')
snapshots1.add_task('wh',layout='g',name='wh')
snapshots1.add_task('ph',layout='g',name='ph')

snapshots2 = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/ret120/ret120_1_c',sim_dt = 0.5 ,max_writes=120)


snapshots2.add_task(IVP.state['ul'],layout='c',name='ul')
snapshots2.add_task(IVP.state['vl'],layout='c',name='vl')
snapshots2.add_task(IVP.state['wl'],layout='c',name='wl')
snapshots2.add_task(IVP.state['pl'],layout='c',name='pl')

snapshots2.add_task(IVP.state['uh'],layout='c',name='uh')
snapshots2.add_task(IVP.state['vh'],layout='c',name='vh')
snapshots2.add_task(IVP.state['wh'],layout='c',name='wh')
snapshots2.add_task(IVP.state['ph'],layout='c',name='ph')

check = IVP.evaluator.add_file_handler('/glade/scratch/chenyuz/ret120/checkpoints120', sim_dt = 0.5 , max_writes=120, mode = fh_mode)
check.add_system(IVP.state)



# Re=1200 saved in state_variables 20
# Re=1400 saved in state_variables 21
# Re=2300 saved in state_variables 22
# Re=1500 noise = 0.1 saved in state_variable 23
# Re=2300 noise = 0.1 saved in state_variable 24
# Re=~2225 concentric pipe, saved in state_variable 25
# Re=~1800 concentric pipe, saved in state_variable 27
# Re=1000 concentric pipe, saved in state_variable 29
# Re=1000 concentric pipe QL, r, saved in state_variable 30
# Re=1000 concentric pipe QL, r,z, saved in state_variable 31
# Re=800 concentric pipe QL, r,z, saved in state_variable 32




 #Flow properties
   # allows printing of min/max of flow properties to the logger during the main loop
flow = flow_tools.GlobalFlowProperty(IVP,cadence = 1)
   # add state variables as flow properties
flow.add_property('pl', name='l-presurre')
flow.add_property('ul+uh', name='u-velocity')
#flow.add_property('ul',name='ul')
#flow.add_property('uh',name='uh')

flow.add_property('vl+vh',name='v-velocity')
#flow.add_property('vl',name='vl')
#flow.add_property('vh',name='vh')

flow.add_property('wl+wh',name='w-velocity')
#flow.add_property('wl',name='wl')
#flow.add_property('wh',name='wh')

#flow.add_property('ph', name='h-presurre')
#flow.add_property('uh', name='uh-velocity')
#flow.add_property('vh',name='vh-velocity')
#flow.add_property('wh',name='wh-velocity')













start_time = time.time()

try:
    logger.info('Starting loop')
    while IVP.ok:
        IVP.step(dt) # this step right here is the money-maker; advances the simulation forward in time by dt seconds
        if (IVP.iteration-1)%100==0: # print status message to logger every 100 iterations
            logger.info('Iteration:{}, Time: {:e}, dt:{:e}'.format(IVP.iteration,IVP.sim_time,dt))
            logger.info('Max/Min U={:f},{:f}'.format(flow.max('u-velocity'),flow.min('u-velocity')))
            #logger.info('Max/Min Ul={:f},{:f}'.format(flow.max('ul'),flow.min('ul')))
            #logger.info('Max/Min Uh={:f},{:f}'.format(flow.max('uh'),flow.min('uh')))

            logger.info('Max/Min V={:f},{:f}'.format(flow.max('v-velocity'),flow.min('v-velocity')))
            #logger.info('Max/Min Vl={:f},{:f}'.format(flow.max('vl'),flow.min('vl')))
            #logger.info('Max/Min Vh={:f},{:f}'.format(flow.max('vh'),flow.min('vh')))

            logger.info('Max/Min W={:f},{:f}'.format(flow.max('w-velocity'),flow.min('w-velocity')))
            #logger.info('Max/Min Wl={:f},{:f}'.format(flow.max('wl'),flow.min('wl')))
            #logger.info('Max/Min Wh={:f},{:f}'.format(flow.max('wh'),flow.min('wh')))


            dt=CFL.compute_dt() # calculates time step based on CFL condition
            #dt = 1E-5
except:
   logger.error('Exception raised, triggering end of main loop.')
   raise
finally: # print summary statistics to logger on completion of simulation
   end_time=time.time()
   logger.info('Iterations:{}'.format(IVP.iteration))
   logger.info('Sim end time:{:f}'.format(IVP.sim_time))
   logger.info('Run time: {:.2f} sec'.format(end_time-start_time))
   logger.info('Run time: {:f} cpu-hr'.format((end_time-start_time)/(60.*60.)*domain.dist.comm_cart.size))

end_time = time.time()
print('Run time: {}'.format(end_time - start_time))
