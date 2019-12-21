import os
import numpy as np
from dump_parser import lammps_parser
from flare import struc, kernels, env
from mc_kernels import mc_simple
import pickle
from tqdm import tqdm

"""We begin by loading the GP model on which the Pd/Ag mapped force field was based. Note that to compute energies we need to assign a force/energy kernel to the GP. This was not assigned in the training process because we didn't need to compute energies to train the GP (it's trained on forces only). This kernel comes from the mc_kernels repository on the MIR github, which hasn't been open sourced yet. (Let me know if you run into access issues when downloading the code.)"""

gp_file = open('PdAg.gp', 'rb')
gp_model = pickle.load(gp_file)
gp_model.energy_force_kernel = mc_simple.two_plus_three_mc_force_en

# parse lammps output
ls_files = os.popen('ls -1 coords.final.*').readlines()
ls_files = [line.split() for line in ls_files]
lammps_files = []
for file in ls_files:
    lammps_files.append(file[0])

# set species
dump = open(lammps_files[0],'r')
dump_lines = dump.readlines()
dump_lines = [line.split() for line in dump_lines]
N_Ag = 0
N_Pd = 0
for line in dump_lines[9:]:
    if int(line[1]) == 1:
        N_Ag += 1
    else:
        N_Pd += 1
species = ['Ag'] * N_Ag + ['Pd'] * N_Pd
nat = len(species)
dump.close()

energies = np.zeros(len(lammps_files))

"""Print forces and GP standard deviations to make sure the values look reasonable."""
out_f = open('gp_f.txt','w')

# loop over NEB structures
for count, lammps_file in enumerate(tqdm(lammps_files)):

    """The lammps_parser function, defined in dump_parser.py in this directory, extracts the coordinates from a dump file and converts the BOX_BOUNDS output into an array of Bravais lattice vectors."""
    positions, cell = lammps_parser(lammps_file)
    structure = struc.Structure(cell, species, positions)

    # loop over atoms in the structure
    local_energies = np.zeros(nat)
    for n in range(structure.nat):

        """Construct an atomic environment object, which stores all the interatomic distances within 2- and 3-body cutoff spheres that are needed to compute the local energy assigned to the atom."""
        chemenv = env.AtomicEnvironment(structure, n, gp_model.cutoffs)
        for i in range(3):
            force, var = gp_model.predict(chemenv, i + 1)
            structure.forces[n][i] = float(force)
            structure.stds[n][i] = np.sqrt(np.abs(var))
            local_energies[n] = gp_model.predict_local_energy(chemenv)

    out_f.write('Image = ' + str(count) + '\n')
    out_f.write('forces:')
    out_f.write(str(structure.forces))
    out_f.write('\n')
    out_f.write('stds:')
    out_f.write(str(structure.stds))
    out_f.write('\n')

    """Sum up the local energies to get the total energy of the structure."""
    total_energy = np.sum(local_energies)
    energies[count] = total_energy

out_f.close()

# store energies as np array
np.save('gp_neb', energies)
