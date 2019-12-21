import numpy as np

import os
import sys

import ase
from ase.visualize import view

from surfator import StructureGroupAnalysis, STRUCTURE_GROUP_ATTRIBUTE
import surfator.grouping
from surfator.analysis import calculate_coord_numbers
from surfator.util.layers import get_layer_heights_kmeans

from sitator import SiteNetwork
from sitator.misc import GenerateClampedTrajectory
from sitator.dynamics import RemoveShortJumps

from tqdm.autonotebook import tqdm

import logging

# ---- FROM ASE https://gitlab.com/ase/ase/blob/master/ase/io/lammpsrun.py ----
# Based on ASE's code for loading lammpstrj (which we don't use because its
# performance is very bad)
def construct_cell(celldata):
    """
    ASE's original docs:

    Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    diagdisp = celldata[:, :2].reshape(6, 1).flatten()
    offdiag = celldata[:, 2]

    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def read_lammpstraj(path):
    """
    Assumes unchancing cell/box bounds and unchanging number of atoms.

    !!! ASSUMES SORTED OUTPUT !!!

    Returns:
        - traj
        - frame_times
        - atoms
        - cellstr
    """

    frames = []
    frame_times = []
    n_atoms = None
    cell = None

    with open(path) as f:
        # f will raise StopIteration
        try:
            first_frame = True
            while(True):
                assert next(f).startswith("ITEM: TIMESTEP")
                processed_full_frame = False
                frame_times.append(int(next(f)))
                assert next(f).startswith("ITEM: NUMBER OF ATOMS")
                n_atoms = int(next(f))
                assert next(f).startswith("ITEM: BOX BOUNDS xy xz yz pp pp ff")
                cellstr = [next(f) for _ in range(3)]
                celldata = np.loadtxt(cellstr)
                cellstr = "ITEM: BOX BOUNDS xy xz yz pp pp ff\n" + "".join(cellstr).strip()
                cell, _ = construct_cell(celldata)
                assert next(f).startswith("ITEM: ATOMS id type xu yu z")
                frame = np.empty(shape = (n_atoms, 3))
                if first_frame
                    types = np.empty(shape = n_atoms, dtype = np.int)
                    ids = np.empty(shape = n_atoms, dtype = np.int)
                for atom_i in range(n_atoms):
                    lsplit = next(f).split()
                    frame[atom_i] = lsplit[2:5]
                    types[atom_i] = lsplit[1]
                    ids[atom_i] = lsplit[0]
                frames.append(frame)
                processed_full_frame = True
                first_frame = False
        except StopIteration:
            pass

    assert processed_full_frame

    frames = np.asarray(frames)
    frame_times = np.asarray(frame_times)

    # Make atoms
    atoms = ase.Atoms(positions = frames[0], cell = cell, pbc = True)
    atoms.set_atomic_numbers(types)
    atoms.set_tags(ids)

    return frames, frame_times, atoms, cellstr


def write_lammpstraj(path, cellstr, traj, atoms, coords = None, timesteps = None):
    """WARNING: this function is NOT generic AT ALL

    JUST FOR SOME REALLY SPECIFIC TRAJECTORY FILES
    """
    if timesteps is None:
        timesteps = np.arange(len(traj))

    n_atoms = traj.shape[1]
    assert traj.shape[2] == 3

    if coords is None:
        atoms_header = "ITEM: ATOMS id type x y z"
        atom_format = "{} {} {:05f} {:05f} {:05f}"
    else:
        atoms_header = "ITEM: ATOMS id type xu yu z c_cn "
        atom_format = "{} {} {:05f} {:05f} {:05f} {:d}"

    ids = atoms.get_tags()
    types = atoms.get_atomic_numbers()

    with open(path, 'w') as f:
        for f_idex, frame in enumerate(traj):
            print("ITEM: TIMESTEP", file = f)
            print(timesteps[f_idex], file = f)
            print("ITEM: NUMBER OF ATOMS", file = f)
            print(n_atoms, file = f)
            print(cellstr, file = f)
            print(atoms_header, file = f)
            for atom_i in range(n_atoms):
                print(atom_format.format(ids[atom_i], types[atom_i], frame[atom_i, 0], frame[atom_i, 1], frame[atom_i, 2], (None if coords is None else coords[f_idex, atom_i])), file = f)

    return


def main(traj_path,
         ref_path,
         ref_structgrps_path,
         struct_compat_path,
         out_path,
         n = None,
         surface_layer_index = None,
         trajslice = None,
         cutoff = 3,
         min_layer_dist = 1.0,
         runoff_votes_weight = 0.6,
         winner_bias = 0.5,
         assign_cutoff = None,
         skin = 0,
         surface_normal = np.array([0, 0, 1]),
         min_winner_percentage = 0.50001):
    """
    Args:
        - traj (ndarray n_frames x n_atoms x 3)
        - ref_structure (ASE atoms len(.) = n_atoms)
        - cutoff (float, Angstrom): For computing coordination number
    """
    fh = logging.FileHandler(os.path.join(out_path, 'surfator.log')
    surfator_log = logging.getLogger("surfator")
    surfator_log.setLevel(logging.INFO)
    surfator_log.addHandler(fh)
    sitator_log = logging.getLogger("sitator")
    sitator_log.setLevel(logging.INFO)
    sitator_log.addHandler(fh)

    print("Loading trajectory and reference structure...")
    traj, timesteps, atoms, cellstr = read_lammpstraj(traj_path)
    ref_structure = ase.io.read(ref_path, parallel = False)
    ref_struct_groups = np.load(ref_structgrps_path)
    struct_compat = np.load(struct_compat_path)

    ref_sn = SiteNetwork(
        structure = atoms,
        static_mask = np.zeros(len(atoms), dtype = np.bool),
        mobile_mask = np.ones(len(atoms), dtype = np.bool)
    )
    ref_sn.centers = ref_structure.get_positions()
    ref_sn.add_site_attribute(STRUCTURE_GROUP_ATTRIBUTE, ref_struct_groups)

    if trajslice is not None:
        trajslice = slice(*(None if e == '' else int(e) for e in trajslice.split(":")))
        traj = traj[trajslice]
        timesteps = timesteps[trajslice]

    print("Cell:")
    print(atoms.cell)

    print("Determining layers...")
    assert n is not None and surface_layer_index is not None
    heights_kmeans_stride = max(len(traj) // 300, 1)  # Why not? 300 frames of heights sounds reasonable
    layers = get_layer_heights_kmeans(traj[::heights_kmeans_stride], atoms.cell, n, surface_normal = surface_normal)
    print("Layer heights: %s" % layers)

    print("Assigning to reference sites...")
    if assign_cutoff is None:
        assign_cutoff = cutoff

    # layerfunc = surfator.grouping.layers.agree_within_layers_kmeans(
    #     initial_layer_heights = layers,
    #     surface_normal = surface_normal,
    #     min_layer_dist = min_layer_dist
    # )
    layerfunc = surfator.grouping.layers.agree_within_layers(
        layer_heights = layers,
        surface_normal = surface_normal,
        #cutoff_above_top = assign_cutoff # Be a little more generous on top.
    )
    agreefunc = surfator.grouping.layers.agree_within_layers_and_deposits(
        layerfunc,
        surface_layer_index = surface_layer_index,
        cutoff = cutoff
    )

    sga = StructureGroupAnalysis(
        min_winner_percentage = min_winner_percentage,
        runoff_votes_weight = runoff_votes_weight,
        winner_bias = winner_bias,
        error_on_no_majority = False
    )
    st, agreegrp_assign, structgrp_assign = sga.run(
        ref_sn = ref_sn,
        traj = traj,
        cutoff = assign_cutoff,
        agreement_group_function = agreefunc,
        structure_group_compatability = struct_compat,
        return_assignments = True,
    )
    np.save(os.path.join(out_path, "agreegrp-assignments.npy"), agreegrp_assign)
    np.save(os.path.join(out_path, "structgrp-assignments.npy"), structgrp_assign)

    print("    Average majority: %i%%; minimum majority %i%%" % (100 * sga.average_majority, 100 * sga.minimum_majority))
    st.compute_site_occupancies()
    occs = st.site_network.occupancies
    print("    Min occupancy: %.2f; avg. occupancy: %.2f; max occupancy: %.2f" % (np.min(occs), np.mean(occs), np.max(occs)))
    n_multiple_assign = 0
    for frame in st.traj:
        _, counts = np.unique(frame, return_counts = True)
        n_multiple_assign += np.sum(counts > 1)
    print("    n multiple assignment: %i" % n_multiple_assign)

    print("Removing short jumps...")
    rsj = RemoveShortJumps()
    st = rsj.run(st, threshold = 1)

    print("Clamping trajectory...")
    gct = GenerateClampedTrajectory(wrap = False, pass_through_unassigned = True)
    clamped_traj = gct.run(st)

    print("Computing new coordination numbers...")
    # Now get coordination numbers
    coords = calculate_coord_numbers(traj = clamped_traj,
                                     atoms = atoms,
                                     cutoff = cutoff,
                                     skin = skin)
    nums, counts = np.unique(coords, return_counts = True)
    maxcount = np.max(counts)
    width = 50
    print("Coordination histogram:")
    for n, c in zip(nums, counts):
        print(("  {:3d}: {:%is}    (x{:8d})" % width).format(n, "#" * int(width * c / maxcount), c))

    print("Writing trajectory out...")
    write_lammpstraj(os.path.join(out_path, "clamped-vmd.out"), traj = clamped_traj, atoms = atoms, timesteps = timesteps, cellstr = cellstr)
    write_lammpstraj(os.path.join(out_path, "clamped.out"), traj = clamped_traj, atoms = atoms, coords = coords, timesteps = timesteps, cellstr = cellstr)
    print("Done.")


if __name__ == "__main__":
    argv = sys.argv

    if len(argv) == 7:
        import json
        kwargs = argv[5]
        if kwargs[0] == '{':
            kwargs = json.loads()
        else: # It's a path
            with open(kwargs) as f:
                kwargs = json.load(f)
    elif len(argv) == 6:
        kwargs = {}
    else:
        print("lmpclamp.py traj_path ref_path structgrps_path structcompat_path [\"json-kwargs-str\"|/path/to/kwargs.json] out_path")
        sys.exit(-1)

    main(*argv[1:5],
         out_path = argv[-1],
         **kwargs)
