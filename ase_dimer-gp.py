import os
import numpy as np
from math import pi
from dump_parser import lammps_parser
from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.units import *
#from ase.optimize import BFGS, FIRE
from ase.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from flare import kernels, gp
from flare.modules import gp_calculator
from mc_kernels import mc_simple
import pickle

# lammps dump files
TS = 'TS.dmp'
FS = 'rlx.dmp'

# set species
dump = open(TS,'r')
dump_lines = dump.readlines()
dump_lines = [line.split() for line in dump_lines]
N_Ag = 0
N_Pd = 0
for line in dump_lines[9:]:
    if int(line[1]) == 1:
        N_Ag += 1
    else:
        N_Pd += 1
N_atom = N_Ag + N_Pd
species = 'Ag' + str(N_Ag) + 'Pd' + str(N_Pd)
dump.close()

# dimer parameters
prefix = 'dimer'
force_tol = 0.01    # eV/Ang
dimer_dist = 0.01

# attempt a restart
restart = False
for name in os.listdir('.'):
    if name==('%s.traj'%prefix): restart = True

if restart:
    print("Found %s.traj, attempting to restart..."%prefix)
    try:
        dimer = read('%s.traj@-1'%prefix)
        mode = np.loadtxt('%s.MODE'%prefix) 
    except Exception as err:
        print("ERROR: %s"%err)
        raise err
    print('Restarting from saved trajectory') 
else:
    positions_TS, cell_TS = lammps_parser(TS)
    positions_FS, cell_FS = lammps_parser(FS)
    dimer = Atoms(species, positions=positions_TS, cell=cell_TS, pbc=True)
    mode = positions_FS - positions_TS
    mode = mode/np.linalg.norm(mode.ravel())

# fix bottommost layer
mask = [atom.index for atom in dimer if atom.position[2] < 9]
N_fix = len(mask)
constraint = FixAtoms(mask=mask)
dimer.set_constraint(constraint)
d_mask = [False] * N_fix + [True] * (N_atom - N_fix)

# set gp calculator
gp_file = open('PdAg.gp', 'rb')
gp_model = pickle.load(gp_file)
gp_model.energy_force_kernel = mc_simple.two_plus_three_mc_force_en
dimer.set_calculator(gp_calculator.GPCalculator(gp_model))

# dimer setup
dimer_control = DimerControl(initial_eigenmode_method='displacement',
                             displacement_method='vector',
                             maximum_translation=0.2, 
                             trial_trans_step=5.0e-3,
                             dimer_separation=dimer_dist,
                             mask=d_mask,
                             f_rot_min=0.01,
                             f_rot_max=0.2,
                             trial_angle=pi/4,
                             max_num_rot=2,
                             logfile='%s_control.log'%prefix,
                             eigenmode_logfile='%s_eigenmode.log'%prefix)

dimer_atoms = MinModeAtoms(dimer, dimer_control, eigenmodes=[mode])
dimer_atoms.displace(displacement_vector = mode)

dimer_relax = MinModeTranslate(dimer_atoms, trajectory='%s.traj'%prefix, logfile='%s_relax.log'%prefix)
# also possible to use these:
#dimer_relax = FIRE(dimer_atoms, trajectory='dimer.traj',
#                   restart='restart_file',
#                   dt=0.1, # default 0.1
#                   maxmove=0.2, # default 0.2
#                   dtmax=1.0, # default 1.0
#                   Nmin=3, # default 5
#                   finc=1.1, # default 3
#                   fdec=0.71, # default 0.5
#                   astart=0.2, # default 0.1
#                   fa=0.99) # default 0.99
#dimer_relax = BFGS(dimer_atoms, trajectory='dimer.traj', restart='restart_file')

# run dimer
dimer_relax.run(fmax=force_tol)

# clean up
np.savetxt('%s.MODE'%prefix, dimer_atoms.get_eigenmode())
write('POSCAR.vasp', dimer_atoms.get_atoms(), format='vasp')
