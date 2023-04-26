import numpy as np
from quspin.basis import spin_basis_1d
from tqdm import tqdm
from concurrent import futures
from hamiltonians import twisted_hamiltonian
from traceback import print_exc

TOL = 10**-9


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


def get_row_states(cluster, interactions, row_phis, basis,
                   states, tol):
    N = len(row_phis)
    row_energies = np.zeros((N, states))
    row_states = np.zeros((N, basis.Ns), dtype=np.complex128)
    for i, r in enumerate(compute_row_vs(cluster, interactions,
                                         row_phis, basis,
                                         states, tol)):
        e, v = r
        row_energies[i,:] = np.sort(e)
        row_states[i,:] = v[:,np.argmin(e)]
    return row_energies, row_states


def chern_3pt(cluster, interactions, phis, basis):
    """"
    Calculates the Chern number following the 3-point formula above
    """
    Lx, Ly, _ = np.shape(phis)
    energies = np.zeros((Lx, Ly))
    vs = np.zeros([Lx, Ly, basis.Ns], dtype=np.complex128)
    for i in range(Lx):
        for j in range(Ly):
            H = twisted_hamiltonian(cluster, interactions, phis[i,j,:], basis)
            e, v = H.eigsh(k=1, which='SA')
            energies[i,j] = e
            vs[i,j,:] = v[:,0]
    integrand = np.zeros([Lx, Ly], dtype=np.complex128)
    for i in range(Lx):
        for j in range(Ly):
            v = vs[i, j,:]
            vx = vs[(i+1)%Lx, j,:] # (i+1)%Lx returns 0 when i +1 = Lx (so $\pi \rightarrow -\pi$)
            vy = vs[i, (j+1)%Ly,:] # same for Ly
            integrand[i,j] = np.log(np.vdot(v, vx)*np.vdot(vx, vy)*np.vdot(vy, v))/np.pi

    return np.sum(integrand).imag, energies


def chern_phi_multi(cluster, interactions, phis, basis,
                    states=1, tol=TOL):
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
        # energies[i,:,:], vs[i,:,:] = get_row_states(cluster, interactions,
        #                                             row_phis, basis, states, tol)
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


def chern_3pt_multi(cluster, interactions, phis, basis, tol=TOL,
                    states=1):
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
    energies = np.zeros((Lx, Ly, states))
    # Creating first row of states (will save until the end)
    phis_0 = [phis[i,0,:] for i in range(Lx)]
    energies[:,0,:], v0s = get_row_states(cluster, interactions,
                                          phis_0, basis, states, tol)
    vs = v0s.copy()
    vys = np.zeros((Lx, basis.Ns), dtype=np.complex128)
    integrand = np.zeros([Lx, Ly], dtype=np.complex128)
    # Getting first through second-to-last integrands
    # In the process: getting the next row of states up.
    for j in tqdm(range(Ly-1)):
        row_phis = [phis[i,j+1, :] for i in range(Lx)]
        energies[:,j+1,:], vys = get_row_states(cluster, interactions,
                                                phis_0, basis, states, tol)
        for i in range(Lx):
            v = vs[i,:]
            vx = vs[(i+1)%Lx,:]
            vy = vys[i,:]
            integrand[i,j] = np.vdot(v, vx)*np.vdot(vx, vy)*np.vdot(vy, v)
        vs = vys.copy()
    # Now, we have the final row. The next row of states up is the first row,
    # which we have been holding on to. Perhaps this is a waste of memory?
    for i in range(Lx): # final row
        v = vs[i,:]
        vx = vs[(i+1)%Lx,:]
        vy = v0s[i,:]
        integrand[i,Ly-1] = np.vdot(v, vx)*np.vdot(vx, vy)*np.vdot(vy, v)

    return np.sum(np.log(integrand)).imag/np.pi, energies, integrand




if __name__ == '__main__':
    import time
    from clusters import hex6
    N = int(input('Steps: '))
    h = float(input('h: '))

    interactions = {'Kitaev': np.ones(3)/3,
                    'h': h*np.ones(3)/np.sqrt(3)}
    phis_row = np.arange(N)*2*np.pi/N
    phis = np.array([[[phis_row[i], phis_row[j]] for i in range(N)]
                     for j in range(N)])
    basis = spin_basis_1d(6, pauli=0)
    c0, e = chern_3pt(hex6, interactions, phis, basis)
    print(c0)
    print(np.round(e, 3))
    c1, c2, e = chern_phi_multi(hex6, interactions, phis, basis)
    print(c1)
    print(c2)
    print(np.round(e[:,:,0], 3))
    c3, e, _ = chern_3pt_multi(hex6, interactions, phis, basis)
    print(c3)
    print(np.round(e[:,:,0], 3))
