import numpy as np


def lammps_parser(dump_file):
    positions = []
    box = []

    with open(dump_file, 'r') as outf:
        lines = outf.readlines()

    for count, line in enumerate(lines):
        if line.startswith('ITEM: ATOMS'):
            position_start = count

        if line.startswith('ITEM: BOX BOUNDS'):
            box_start = count

    for line in lines[position_start+1:]:
        fline = line.split()
        positions.append([float(fline[-3]),
                          float(fline[-2]),
                          float(fline[-1])])
    positions = np.array(positions)

    for line in lines[box_start+1:box_start+4]:
        fline = line.split()
        box.append([float(fline[-3]),
                    float(fline[-2]),
                    float(fline[-1])])

    # create cell from lammps box
    box = np.array(box)
    xlo_bound = box[0, 0]
    xhi_bound = box[0, 1]
    xy = box[0, 2]
    ylo_bound = box[1, 0]
    yhi_bound = box[1, 1]
    xz = box[1, 2]
    zlo_bound = box[2, 0]
    zhi_bound = box[2, 1]
    yz = box[2, 2]

    xlo = xlo_bound - np.min(np.array([0, xy, xz, xy + xz]))
    xhi = xhi_bound - np.max(np.array([0, xy, xz, xy + xz]))
    ylo = ylo_bound - np.min(np.array([0, yz]))
    yhi = yhi_bound - np.max(np.array([0, yz]))
    zlo = zlo_bound
    zhi = zhi_bound

    cell = np.array([[xhi-xlo, 0, 0],
                     [xy, yhi-ylo, 0],
                     [xz, yz, zhi-zlo]])

    return positions, cell

if __name__ == '__main__':
    dump_file = 'Pd/coords.final.0'
    pos_test = lammps_parser(dump_file)

    print(pos_test)
