import os,sys
import time
import SourcePy.evolve as evolve
import SourcePy.record as record
from matplotlib import pyplot as plt
import numpy as np

run_name = 'zhong_cal_06'

hdict = {   # DM halo profile:
            'profile': 'NFW', # 'NFW', 'Hernquist', or 'Isothermal' 
                              # Specify halo density profile.
            'r_s': 9.1, # DM Scale radius of halo in units of [kpc]
            'rho_s': 0.0069, # DM Scale density in units of [M_sun/pc^3]
            
            # baryon properties: (The profile is always Hernquist)
            'M_b': 1e9, # baryon mass in units of [M_sun]
            'r_b': 0.77, # baryon scale radius in units of [kpc]

            # Fluid properties:
            'a': 4./np.sqrt(np.pi), # Coefficient for hard-sphere scattering
            'b': 25.*np.sqrt(np.pi)/32., # Effective impact parameter
            'C': 0.6, # alibration parameter for LMFP thermal conductivity kappa. Must be calibrated to simulation.
            'gamma': 5./3., # Gas constant for monatomic gas.
            
            # particle properties:
            'model_elastic_scattering_lmfp': 'constant', # Particle physics model for the scattering cross section in the LMFP
                                                         # regime. See `initialize_elastic_scattering()` for details.
            'model_elastic_scattering_smfp': 'constant', # Particle physics model for the scattering cross section in the SMFP
                                                         # regime. See `initialize_elastic_scattering()` for details.
            'sigma_m_with_units': 100, # [cm^2/g] Cross section prefactor for elastic scattering.
            'w_units': 1., # [km/s] Velocity scale for elastic scattering.
            
            # inputs for numerical solving: 
            'n_shells': 400, # Number of shells to divide the halo into.
            'r_min': 0.01, # Minimum dimensionless radius of radial bins.
            'r_max': 100., # Maximum dimensionless radius of radial bins.
            'p_rmax_factor': 10., # Factor to multiply r_max to obtain upper limit of integral when
                                  # numerically integrating to find initial halo pressure.
            'n_adjustment_fixed': -1, # Number of forced hydrostatic adjustment steps per heat conduction
                                      # steps. A negative value indicates a fixed number should not be used
                                      # and this parameter is ignored. The default behavior is to determine
                                      # the number of steps dynamically to satisfy a convergence criterion
                                      # set at run time; see implementation of `hydrostatic_adjustment()`.
            'n_adjustment_max': -1, # Number of maximum hydrostatic adjustment steps per heat conduction
                                    # steps. A negative value indicates a maximum should not be enforced
                                    # and this parameter is ignored. The default behavior is to determine
                                    # the number of steps dynamically to satisfy a convergence criterion
                                    # set at run time; see implementation of `hydrostatic_adjustment()`.
            'flag_timestep_use_relaxation': True, # If True, incorporate the minimum local relaxation time of the halo
                                                  # to help set the heat conduction time step.
            'flag_timestep_use_energy': True, # If True, incorporate the relative specific energy change
                                              # to help set the heat conduction time step.
            'flag_hydrostatic_initial': True # If True, perform hydrostatic adjustment to initialized halo
                                              # at t=0 before beginning halo evolution process.
        }

rdict = {'t_end': 100,'save_frequency_rate': 10}

dir_output = 'data'

# set data directory:
dir_data = os.path.join(dir_output,run_name)

# create halo record
halorec = record.HaloRecord(dir_data)

# create halo
haloevo = evolve.Halo(halorec,**hdict)

# evolve halo
start=time.time()
haloevo.evolve_halo(**rdict)
end=time.time()

print('time elapsed for run {} = {}'.format(run_name,end-start))
