import numpy as np
from quspin.basis import spin_basis_1d
from tqdm import tqdm
from concurrent import futures
from hamiltonians import twisted_hamiltonian
from chern import thread_job
from traceback import print_exc

TOL = 10**-9
CTYPE = np.complex64
TYPE = np.float32


"""
Code for doing this in a multithready way!
"""

def compute_cut(cluster, K, hs, phi, basis, states, tol, workers):
    with futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_results = [executor.submit(thread_job, cluster,
                                          {'Kitaev': K, 
                                           'h': h*np.ones(3)/np.sqrt(3)},
                                          phi, basis, states, tol)
                          for h in hs]
        futures.wait(future_results)
        for res in future_results:
            try:
                yield res.result()
            except:
                print_exc()


def thread_job(cluster, interactions, phis, basis, states, tol):
    H = twisted_hamiltonian(cluster, interactions, phis, basis, dtype=CTYPE)
    return H.eigsh(which='SA', k=states, tol=tol, return_eigenvectors=False)


def get_cut(cluster, K, hs, phi, basis, states=1, tol=TOL, workers=None):
    energies = np.zeros((N, states), dtype=TYPE)
    for i, r in enumerate(compute_cut(cluster, K, hs, phi, basis, states, tol, workers)):
        e = r
        energies[i,:] = np.sort(e)
    return energies





if __name__ == '__main__':
    from clusters import hex6, hex24
    from pickle import dump, load
    import time
    L = int(input('System size: '))
    N = int(input('Steps: '))
    h0 = float(input('h0: '))
    hf = float(input('hf: '))
    states = int(input('states: '))
    workers = int(input('Number of workers: '))
    hs = np.linspace(h0, hf, N)
    K = np.ones(3)

    fname = 'L{}_{}steps_{}states_h{}-{}.p'.format(L, N, states, h0, hf)

    if L == 24:
        basis = spin_basis_1d(24, pauli=1)
        cluster = hex24
    else:
        basis = spin_basis_1d(6, pauli=1)
        cluster = hex6

    phis = [[0,0], [np.pi, 0], [np.pi, np.pi]]
    out = {'K': K, 'hs': hs, 'L': L, 'phis': phis,
           'energies': np.zeros((len(phis), N, states), dtype=TYPE)}
    for i, p in enumerate(phis):
        print('Phase: ')
        print(p)
        t0 = time.time()
        out['energies'][i,:,:] = get_cut(cluster, K, hs, p, basis, 
                                         states=states, workers=workers)
        tf = time.time()
        print('Took {} seconds'.format(tf-t0))
        dump(out, open(fname, 'wb'))

