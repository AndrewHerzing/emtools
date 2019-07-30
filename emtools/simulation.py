import numpy as np
import os
import re
pixstem_is_installed = False
hyperspy_is_installed = False
try:
    import pixstem.api as ps
    pixstem_is_installed = True
except ImportError:
    try:
        import hyperspy.api as hs
        hyperspy_is_installed = True
    except ImportError:
        print('Missing both Pixstem and Hyperspy.')
        print('Data will be returned as NumPy arrays')


def load_mustem_pacbed(pathname, scanXY=None, detXY=None):
    """
    Function to load the output of a muSTEM PACBED simulation.

    Args
    ----------
    pathname : string
        Path to data series
    scanXY : list
        Integer number of probe positions in the x and y direction
    detXY : list
        Size of the simulated patterns in the X and Y direction

    Returns
    ----------
    s : PixelatedSTEM signal, Signal2D, NumPy Array
        4D-STEM simulation is returned as either a Pixstem class,
        a Hyperspy class, or a NumPy array depending on whether or not the
        packages are installed.
    """
    files = os.listdir(pathname)
    scans = np.zeros([len(files), 2])
    for i in range(0, len(files)):
        scan_check = re.search('([0-9]+)_([0-9]+)', files[i])
        scans[i, :] = np.array([np.int32(scan_check.group(0).split('_')[0]),
                               np.int32(scan_check.group(0).split('_')[1])])
    if not scanXY:
        scanY, scanX = np.int32(scans.max(0))
    else:
        scanY, scanX = scanXY

    if not detXY:
        det_check = re.search('([0-9]+)x([0-9]+)', files[0])
        detY, detX = np.int32(det_check.group(0).split('x'))
    else:
        detY, detX = detXY

    idx = np.lexsort((scans[:, 1], scans[:, 0])).tolist()
    files_sorted = [None] * len(files)
    count = 0
    for i in idx:
        files_sorted[count] = files[i]
        count += 1
    data = np.zeros([scanX * scanY, detX * detY], np.float32)

    pos = 0
    for i in files_sorted:
        filename = pathname + i
        with open(filename, 'rb') as h:
            data[pos, :] = \
                np.fromfile(h, count=detX * detY, dtype='>f4')
        pos += 1
    data = data.reshape([scanY, scanX, detY, detX])
    if pixstem_is_installed:
        s = ps.PixelatedSTEM(data)
    elif hyperspy_is_installed:
        s = hs.Signal2D(data)
    else:
        s = data
    return s


def read_vesta(filename):
    """
    Function to read a VESTA model file and parse to extract relevant
    structural parameters for TEM/STEM simulation.

    Args
    ----------
    filename : string
        Name of the VESTA model file

    Returns
    ----------
    header : list
        Information stored in the VESTA file prior to the atomic coordinates.
    coords : list
        List of atomic coordinates.
    """
    data = np.loadtxt(filename, dtype='str', delimiter='\t')
    for i in range(0, len(data)):
        if data[i] == 'STRUC':
            start = i + 1
        elif data[i] == 'THERI 0':
            end = i - 2
    header = data[0:start]
    coords = data[np.arange(start, end, 2)]
    return header, coords


def write_xtl(input_filename, output_filename, title='Phase', n_elements=1,
              atom_id='Si', atomic_num=14.0, occupancy=1.0,
              debye_waller=0.00292):
    """
    Function to read convert a VESTA model file to a XTL file for muSTEM
    simulation.

    Args
    ----------
    input_filename : string
        Name of the VESTA model file.
    output_filename : string
        Name of the resulting XTL file.
    title : string
        Text description of the structure.
    n_elements : integer
        Number of distinct elements contained in the structure.
    atom_id : string
        Atomic symbol of the majority constituent.
    atomic_num : float or integer
        Z number of majority consituent
    occupancy : float or integer
        Fractionaly occupancy for the atom
    debye_waller : float
        Debye-Waller factor for the majority consituent

    Returns
    ----------

    """
    data = np.loadtxt(input_filename, dtype='str', delimiter='\t')
    for i in range(0, len(data)):
        if data[i] == 'STRUC':
            start = i + 1
        elif data[i] == 'THERI 0':
            end = i - 2
    coords = data[np.arange(start, end, 2)]
    n_atoms = len(coords)
    lattice_param = data[start - 3].split()[0:6]
    with open(output_filename, 'w') as h:
        h.write(title + '\n')
        h.write('%s  %s  %s  %s  %s  %s\n' % (lattice_param[0],
                                              lattice_param[1],
                                              lattice_param[2],
                                              lattice_param[3],
                                              lattice_param[4],
                                              lattice_param[5]))
        h.write(str(n_elements) + '\n')
        h.write(atom_id + '\n')
        h.write('%s %.1f %.1f %.5f\n' %
                (str(n_atoms), atomic_num, occupancy, debye_waller))
        for i in coords:
            line = i.split()[4:7]
            for k in line:
                h.write('\t' + k + '\t')
            h.write('\n')
    return


def open_muSTEM_binary(filename):
    """
    Function to read the binary file output from muSTEM simulation.

    Args
    ----------
    filename : string
        Name of the muSTEM output file to be read.

    Returns
    ----------
    data : NumPy array
        2-D array containing the simulation data
    """
    '''opens binary with name filename outputted from the muSTEM software'''
    assert os.path.exists(filename), filename + ' does not exist'
    # Parse filename for array dimensions
    m = re.search('([0-9]+)x([0-9]+)', filename)
    if m:
        y = int(m.group(2))
        x = int(m.group(1))
    # Get file size and intuit datatype
    size = os.path.getsize(filename)
    if (size / (y * x) == 4):
        dtype = '>f4'
    elif(size / (y * x) == 8):
        dtype = '>f8'
    # Read data and reshape as required.
    data = np.reshape(np.fromfile(filename, dtype=dtype), (y, x))
    return data
