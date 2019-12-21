import numpy as np
from ase.build import fcc111, add_adsorbate
from ase.optimize import MDMin, BFGS
from ase.neb import NEB
from ase.io.trajectory import Trajectory
import sys
import os
import time
from flare import otf, kernels, gp
from flare.modules import qe_input, initialize_velocities, \
    gp_calculator
import pickle

# load gp model
gp_file = open('../slab.gp', 'rb')
gp_model = pickle.load(gp_file)
gp_model.par = True
gp_model.hyp_labels = ['sig2', 'ls2', 'sig3', 'ls3', 'noise']
gp_model.energy_force_kernel = kernels.two_plus_three_force_en

# get initial and final image
hcp_struc = Trajectory('../relax_111/gp_hcp_relax.traj')[-1]
fcc_struc = Trajectory('../relax_111/gp_fcc_relax.traj')[-1]

# make band
no_images = 7
images = [hcp_struc]
images += [hcp_struc.copy() for i in range(no_images-2)]
images += [fcc_struc]
neb = NEB(images)

# interpolate middle images
neb.interpolate()

# set calculators of images
pes = np.zeros(no_images)
for n, image in enumerate(images):
    image.set_calculator(gp_calculator.GPCalculator(gp_model))
    pes[n] = image.get_potential_energy()

np.save('unoptimized_pes', pes)

# optimize the NEB trajectory
optimizer = MDMin(neb, trajectory='neb.traj')
optimizer.run(fmax=0.01)

# calculate the potential energy of each image
pes = np.zeros(no_images)
pos = np.zeros((no_images, len(images[0]), 3))
for n, image in enumerate(images):
    pes[n] = image.get_potential_energy()
    pos[n] = image.positions

np.save('neb_pes', pes)
np.save('neb_pos', pos)
np.save('neb_cell', images[0].cell)
