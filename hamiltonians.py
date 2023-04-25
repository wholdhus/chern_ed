import numpy as np
from quspin.operators import quantum_operator


def pm_mapping(phi):
    return {'x': {'+': 0.5*np.exp(1j*phi),
                  '-': 0.5*np.exp(-1j*phi)},
            'y': {'+': -0.5j*np.exp(1j*phi),
                  '-': 0.5j*np.exp(-1j*phi)},
            'z': {'z': 1.0}}

def twisted_bond(d, phi, J, b, full=True):
    if full:
        if d == 'x':
            bond = [['x+', [[0.5*J*np.exp(1j*phi), *b]]],
                    ['x-', [[0.5*J*np.exp(-1j*phi), *b]]]]
        elif d == 'y':
            bond = [['y+', [[-0.5j*J*np.exp(1j*phi), *b]]],
                    ['y-', [[0.5j*J*np.exp(-1j*phi), *b]]]]
        elif d == 'z':
            bond = [['zz', [[J, b[0], b[1]]]]]
    else:
        if d == 'x':
            bond = [['++', [[0.25*J, *b]]],
                    ['--', [[0.25*J, *b]]],
                    ['-+', [[0.25*J*np.exp(1j*phi), *b]]],
                    ['+-', [[0.25*J*np.exp(-1j*phi), *b]]]]
        elif d == 'y':
            bond = [['++', [[-0.25*J, *b]]],
                    ['--', [[-0.25*J, *b]]],
                    ['-+', [[0.25*J*np.exp(1j*phi), *b]]],
                    ['+-', [[0.25*J*np.exp(-1j*phi), *b]]]]
        elif d == 'z':
            bond = [['zz', [[J, b[0], b[1]]]]]
    return bond


def twisted_hamiltonian(plaq, interactions, phis, basis,
                        dtype=np.complex128, full=True):
    phi1, phi2 = phis
    L = plaq['L']
    bonds = []
    Kx, Ky, Kz = interactions['Kitaev']
    h = {}
    h['x'], h['y'], h['z'] = interactions['h']
    couplings = {'x': Kx, 'y': Ky, 'z': Kz}
    for d in ['x', 'y', 'z']:
        J = couplings[d]
        bonds += [[d, [[-h[d], i] for i in range(L)]]]
        bonds += [[d+d, [[J, *b] for b in plaq['inter_{}'.format(d)]]]]
        for b in plaq['outer_1'][d]:
            bonds += twisted_bond(d, phi1, J, b, full=full)
        for b in plaq['outer_2'][d]:
            bonds += twisted_bond(d, phi2, J, b, full=full)
        if 'outer_3' in plaq:
            for b in plaq['outer_3'][d]:
                bonds += twisted_bond(d, phi1+phi2, J, b, full=full)
    H = quantum_operator({'static': bonds}, basis=basis,
                         check_herm=False, check_symm=False, dtype=dtype)
    return H
