import os
import numpy as np
from ase.io import read
from ase.constraints import FixAtoms
from ase.units import *
from ase.vibrations import Vibrations
from flare import kernels
from flare.modules import gp_calculator
from mc_kernels import mc_simple
import pickle

# load TS
TS = read('dimer.traj')

# fix bottommost layer
fix = [atom.index for atom in TS if atom.position[2] < 9]
move = [atom.index for atom in TS if atom.position[2] > 9]
constraint = FixAtoms(mask=fix)
TS.set_constraint(constraint)

# set gp calculator
gp_file = open('PdAg.gp', 'rb')
gp_model = pickle.load(gp_file)
gp_model.energy_force_kernel = mc_simple.two_plus_three_mc_force_en
TS.set_calculator(gp_calculator.GPCalculator(gp_model))

# calculate Hessian
vib = Vibrations(TS, indices=move)
vib.run()

# clean up
vib.summary(log='gp_freq.txt')
vib.combine()
