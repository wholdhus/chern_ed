import numpy as np
from quspin.basis import spin_basis_1d
from tqdm import tqdm
from concurrent import futures
from hamiltonians import twisted_hamiltonian
from traceback import print_exc

TOL = 10**-9


def chern_3pt_lm(plaq, phis, interactions, basis, tol=10**-12,
                 verbose=False, v0=None, states=1, return_energy=False,
                 full=True):
    """"
    Calculates the Chern number following the 3-point formula above
    K: vector of floats, [Kx, Ky, Kz] representing couplings of the Kitaev model
    h: vector of floats [hx, hy, hz] representing components of an external magnetic field
    phis: array of shape (Lx, Ly, 2) holding the pair (phi_1, phi_2) at each point in our grid
    output:
    C: the Chern number following the 3-point formula
    """
    Lx, Ly, _ = np.shape(phis)
    v0s = np.zeros((Lx, basis.Ns), dtype=np.complex128)
    if verbose:
        is0 = tqdm(range(Lx))
        js0 = tqdm(range(Ly-1))
    else:
        is0 = range(Lx)
        js0 = range(Ly-1)
    energies = np.zeros((Lx, Ly, states))
    for i in is0:
        H = twisted_hamiltonian_2(plaq, interactions, phis[i,0,:], basis, full=full)
        if states == basis.Ns:
            e, v = H.eigh()
        else:
            e, v = H.eigsh(k=states, which='SA', tol=tol, v0=v0)
        e0i = np.argmin(e)
        v0s[i,:] = v[:,e0i]
        v0 = v[:,e0i]
        energies[i,0,:] = np.sort(e)
    vs = v0s.copy()
    vys = np.zeros((Lx, basis.Ns), dtype=np.complex128)
    integrand = np.zeros([Lx, Ly], dtype=np.complex128)
    for j in js0:
        for i in range(Lx):
            H = twisted_hamiltonian_2(plaq, interactions, phis[i,(j+1)%Ly,:], basis,
                                      full=full)
            if states == basis.Ns:
                E, V = H.eigh()
            else:
                E, V = H.eigsh(k=1, which='SA', tol=tol , v0=vs[i,:])
            e0i = np.argmin(E)
            energies[i,j+1,:] = np.sort(E)
            vys[i,:] = V[:,e0i]
            v = vs[i,:]
            vx = vs[(i+1)%Lx,:]
            vy = vys[i,:]
            integrand[i,j] = np.vdot(v, vx)*np.vdot(vx, vy)*np.vdot(vy, v)
        vs = vys.copy()
    for i in range(Lx):
        v = vs[i,:]
        vx = vs[(i+1)%Lx,:]
        vy = v0s[i,:]
        integrand[i,Ly-1] = np.vdot(v, vx)*np.vdot(vx, vy)*np.vdot(vy, v)

    if return_energy:
        return np.sum(np.log(integrand)).imag/np.pi, integrand, energies
    else:
        return np.sum(np.log(integrand)).imag/np.pi, integrand

"""
Code for doing this in a multithready way!
"""

def thread_job(cluster, interactions, phis, basis, states, tol):
    H = twisted_hamiltonian(cluster, interactions, phis, basis)
    return H.eigsh(k=states, tol=tol)


def compute_row_vs(cluster, interactions, row_phis, basis,
                   states, tol):
    Nphi = np.shape(row_phis)[0]
    with futures.ProcessPoolExecutor() as executor:
        future_results = [executor.submit(thread_job, cluster, interactions,
                                          row_phis[i], basis, states=states,
                                          tol=tol)
                          for i in range(Nphi)]
        futures.wait(future_results)
        for res in future_results:
            try:
                yield res.result()
            except:
                print_exc()


def chern_phi_multi(cluster, interactions, phis, basis,
                    states=2, tol=TOL):
    Lx, Ly, _ = np.shape(phis)
    energies = np.zeros((Lx, Ly, states))
    ig = np.zeros((Lx, Ly), dtype=np.complex128)
    U1 = np.zeros((Lx, Ly), dtype=np.complex128)
    U2 = np.zeros((Lx, Ly), dtype=np.complex128)
    F = np.zeros((Lx, Ly), dtype=np.complex128)
    v = None
    vs = np.zeros((Lx, Ly, basis.Ns), dtype=np.complex128)
    bs = np.zeros(Lx, dtype=np.complex128)
    for i in tqdm(range(Lx)):
        row_phis = [[phis[i,j,0], phis[i,j,1]] for j in range(Ly)]
        for j, r in enumerate(compute_row_vs(cluster, interactions,
                                             row_phis, basis,
                                             states, tol)):
            e, v = r
            energies[i,j,:] = np.sort(e)
            vs[i,j,:] = v[:,np.argmin(e)]

    for i in range(Lx):
        bs[i] = np.vdot(vs[i,0,:], vs[(i+1)%Lx,0,:])
        for j in range(Ly):
            v = vs[i,j,:]
            v1 = vs[(i+1)%Lx,j,:]
            v2 = vs[i,(j+1)%Ly,:]
            ig[i,j] = np.vdot(v,v1)*np.vdot(v1,v2)*np.vdot(v2,v)
            U1[i,j] = np.vdot(v,v1)/np.abs(np.vdot(v,v1))
            U2[i,j] = np.vdot(v,v2)/np.abs(np.vdot(v,v2))
    for i in range(Lx):
        for j in range(Ly):
            frac = U1[i,j]*U2[(i+1)%Lx,j]/(U1[i,(j+1)%Ly]*U2[i,j])
            F[i,j] = np.log(frac)
    c1 = (np.sum(F)/(2j*np.pi)).real
    c2 = np.imag(np.sum(np.log(ig)))/np.pi
    return c1, c2, energies



if __name__ == '__main__':
    import time
    from clusters import hex6
    N = int(input('Steps: '))
    h = float(input('h: '))
    interactions = {'Kitaev': np.ones(3),
                    'h': h*np.ones(3)/np.sqrt(3)}
    phis_row = np.arange(N)*2*np.pi/N
    phis = np.array([[[phis_row[i], phis_row[j]] for i in range(N)]
                     for j in range(N)])
    print('phis:')
    print(phis)
    basis = spin_basis_1d(6, pauli=1)
    c1, c2, energies = chern_phi_multi(hex6, interactions, phis, basis)
    print('Chern numbers:')
    print(c1)
    print(c2)
