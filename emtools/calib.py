def get_lattice_spacings(material):
    aupd = {'111': 2.31, '200': 2.00, '220': 1.41, '311': 1.21, '222': 1.15}
    au = {'111': 2.36, '200': 2.04, '220': 1.44, '311': 1.23, '222': 1.18}
    si = {'111': 3.14, '200': 2.72, '220': 1.92, '311': 1.64, '222': 1.57}

    if material.lower() == 'aupd':
        return aupd
    elif material.lower() == 'au':
        return au
    elif material.lower() == 'si':
        return si
    else:
        raise(ValueError, "Unknown material.  Must be 'Au-Pd', 'Au', or 'Si'")
