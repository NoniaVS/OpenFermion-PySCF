#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Driver to initialize molecular object from pyscf program."""

from __future__ import absolute_import

from functools import reduce

import numpy
import scipy
from pyscf import gto, scf, ao2mo, ci, cc, fci, mp, mcscf

from openfermion import MolecularData
from openfermionpyscf import PyscfMolecularData


def prepare_pyscf_molecule(molecule):
    """
    This function creates and saves a pyscf input file.

    Args:
        molecule: An instance of the MolecularData class.

    Returns:
        pyscf_molecule: A pyscf molecule instance.
    """
    pyscf_molecule = gto.Mole()
    pyscf_molecule.atom = molecule.geometry
    pyscf_molecule.basis = molecule.basis
    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = False
    pyscf_molecule.build()

    return pyscf_molecule


def compute_scf(pyscf_molecule):
    """
    Perform a Hartree-Fock calculation.

    Args:
        pyscf_molecule: A pyscf molecule instance.

    Returns:
        pyscf_scf: A PySCF "SCF" calculation object.
    """
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule)
    else:
        pyscf_scf = scf.RHF(pyscf_molecule)
    return pyscf_scf


def mixed_orbitals_density_matrix(pyscf_molecule, mixing_parameter=numpy.pi/4):
    """
    Calculates a density matrix to initialize the UHF calculation

    Args:
        pyscf_molecule: A pyscf molecule instance.
        mixing_parameter: Mixing parameter.

    Returns:
        dm: Density matrix.
    """
    rhf = scf.RHF(pyscf_molecule)
    h_core = pyscf_molecule.intor_symmetric('int1e_kin') + pyscf_molecule.intor_symmetric('int1e_nuc')
    overlap_matrix = rhf.get_ovlp(pyscf_molecule)
    mo_energy, mo_coeff = rhf.eig(h_core, overlap_matrix)
    mo_occ = rhf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx = 0
    lumo_idx = 1

    for i in range(len(mo_occ) - 1):
        if mo_occ[i] > 0 and mo_occ[i + 1] < 0:
            homo_idx = i
            lumo_idx = i + 1

    psi_homo = mo_coeff[:, homo_idx]
    psi_lumo = mo_coeff[:, lumo_idx]

    Ca = numpy.zeros_like(mo_coeff)
    Cb = numpy.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:, k] = numpy.cos(mixing_parameter) * psi_homo + numpy.sin(mixing_parameter) * psi_lumo
            Cb[:, k] = numpy.cos(mixing_parameter) * psi_homo - numpy.sin(mixing_parameter) * psi_lumo
            continue
        if k == lumo_idx:
            Ca[:, k] = -numpy.sin(mixing_parameter) * psi_homo + numpy.cos(mixing_parameter) * psi_lumo
            Cb[:, k] = numpy.sin(mixing_parameter) * psi_homo + numpy.cos(mixing_parameter) * psi_lumo
            continue
        Ca[:, k] = mo_coeff[:, k]
        Cb[:, k] = mo_coeff[:, k]

    dm = scf.UHF(pyscf_molecule).make_rdm1((Ca, Cb), (mo_occ, mo_occ))
    return dm

def compute_integrals(pyscf_molecule, orb_coeff):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = orb_coeff.shape[1]
    h_core = pyscf_molecule.intor_symmetric('int1e_kin') + pyscf_molecule.intor_symmetric('int1e_nuc')
    one_electron_compressed = reduce(numpy.dot, (orb_coeff.T, h_core, orb_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule, orb_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


def compute_integrals_casci(casci, n_orbitals):
    """
    Compute the effective 1-electron, 2-electron integrals with frozen core approach

    Args:
        casci: A pyscf "CASCI" calculation object
        n_orbitals: number of CAS orbitals

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}
    """

    # get effective electronic integrals
    one_electron_integrals, nuclear_repulsion = casci.get_h1eff()
    two_electron_compressed = casci.get_h2eff()

    two_electron_integrals = ao2mo.restore(1, two_electron_compressed, n_orbitals)

    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    return one_electron_integrals, two_electron_integrals, nuclear_repulsion


def set_mo_coefficients(molecule, mo_coefficients):
    """
    overwrite molecular orbitals coefficients of a MolecularData object and recompute one/two boy integrals

    :param molecule: MolecularData object
    :param mo_coefficients: Molecular Orbitals matrix (MO in columns)
    :return:
    """
    one_body_integrals, two_body_integrals = compute_integrals(molecule._pyscf_data['mol'], mo_coefficients)
    molecule._one_body_integrals = one_body_integrals
    molecule._two_body_integrals = two_body_integrals
    molecule._canonical_orbitals = mo_coefficients


def run_pyscf(molecule,
              nat_orb=False,
              guess_mix=False,
              frozen_core=0,
              n_orbitals=None,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False,
              verbose=False):
    """
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.

    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.
    """
    n_orbitals_max = n_orbitals

    # Prepare pyscf molecule.
    pyscf_molecule = prepare_pyscf_molecule(molecule)
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    if nat_orb:
        print('----------------------')
        print('UHF calculation with stabilization, then finding NOs ')
        print('Using guessed mixed as initial point') if guess_mix else print('Not using guessed mixed')
        print('----------------------')
    else:
        print('----------------------')
        print('RHF calculation')
        print('----------------------')

    # Run SCF.
    if nat_orb:
        # UHF calculation
        pyscf_scf = scf.UHF(pyscf_molecule)

        # parameters
        pyscf_scf.conv_tol = 1e-6
        # pyscf_scf.conv_tol_grad = numpy.sqrt(1e-6)
        # pyscf_scf.direct_scf_tol = 1e-13
        pyscf_scf.verbose = 0

        # define initial guess
        dm = mixed_orbitals_density_matrix(pyscf_molecule) if guess_mix else None

        # stability analysis
        max_attempts = 5
        max_mo_diff = 1e-5

        if verbose:
            print('Starting stability analysis')
        for j in range(max_attempts):
            if verbose:
                print("Rotating orbitals to find stable solution: attempt %d." % (j + 1))

            pyscf_scf = pyscf_scf.run(dm)
            ref_mo = pyscf_scf.mo_coeff
            energies_uhf = pyscf_scf.mo_energy

            new_mo = pyscf_scf.stability()[0]
            mo_diff = numpy.linalg.norm(ref_mo[0] - new_mo[0]) + \
                      numpy.linalg.norm(ref_mo[1] - new_mo[1])

            if verbose:
                print('mo_diff: {:10.6e}'.format(mo_diff) + ' S^2: {:5.3f}  2S+1: {:5.3f}'.format(*pyscf_scf.spin_square()))
            spin = pyscf_scf.spin_square()
            dm = pyscf_scf.make_rdm1(new_mo, pyscf_scf.mo_occ)

            if mo_diff < max_mo_diff:
                print("SCF solution is internally stable.")
                break

            if j+1 >= max_attempts:
                print("Unable to find a stable SCF solution after {} attempts.".format(max_attempts))

        if verbose:
            print('The spin of the wavefuction is' ' S^2: {:5.3f}  2S+1: {:5.3f}'.format(*spin))
            print('----------------------')

        # Calculation of natural orbitals
        dm_tot = dm[0] + dm[1]
        overlap_matrix = pyscf_scf.get_ovlp(pyscf_molecule)
        nat_occ, nat_coeff = scipy.linalg.eigh(a=dm_tot, b=overlap_matrix, type=2)

        # Ordering by occupancies
        nat_occ = nat_occ[::-1]
        idx = nat_occ.argsort()
        nat_coeff = nat_coeff[:, idx]
        if verbose:
            print('ENERGY UHF CALCULATED', pyscf_scf.e_tot)

    pyscf_scf = compute_scf(pyscf_molecule)
    pyscf_scf.verbose = 0
    pyscf_scf.run()

    molecule.hf_energy = float(pyscf_scf.e_tot)
    if verbose:
        print('RESTRICTED Hartree-Fock energy for {} ({} electrons) is {}.'.format(
            molecule.name, molecule.n_electrons, molecule.hf_energy))

    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Populate fields.
    if nat_orb:
        molecule.canonical_orbitals = nat_coeff.astype(float)
        molecule.orbital_energies = nat_occ.astype(float)
        pyscf_scf.mo_coeff = molecule.canonical_orbitals
    else:
        molecule.canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
        molecule.orbital_energies = pyscf_scf.mo_energy.astype(float)

    if frozen_core > 0 or n_orbitals_max is not None:

        if n_orbitals_max is None:
            n_orbitals_max = molecule.n_orbitals
        # get CASCI MO effective integrals (frozen core)
        n_cas_elec = molecule.n_electrons - frozen_core*2

        if n_cas_elec < 0:
            raise Exception('cannot freeze more electrons than orbitals')

        if n_orbitals_max - frozen_core < 0:
            raise Exception('n_orbital should be larger than frozen_core')

        if n_orbitals_max > molecule.n_orbitals:
            n_orbitals_max = molecule.n_orbitals

        casci = mcscf.CASCI(pyscf_scf, n_orbitals_max - frozen_core, n_cas_elec)

        if verbose:
            molecule.casci_energy = casci.kernel()[0]

        one_body_integrals, two_body_integrals, nuclear = compute_integrals_casci(casci, n_orbitals_max - frozen_core)
        molecule.nuclear_repulsion = nuclear
    else:
        # Get MO integrals.
        one_body_integrals, two_body_integrals = compute_integrals(pyscf_molecule, molecule.canonical_orbitals)

    molecule.one_body_integrals = one_body_integrals
    molecule.two_body_integrals = two_body_integrals
    molecule.overlap_integrals = pyscf_scf.get_ovlp()

    # Run MP2.
    if run_mp2:
        if molecule.multiplicity != 1:
            print("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            pyscf_mp2.run()
            # molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
            pyscf_data['mp2'] = pyscf_mp2
            if verbose:
                print('MP2 energy for {} ({} electrons) is {}.'.format(
                    molecule.name, molecule.n_electrons, molecule.mp2_energy))

    # Run CISD.
    if run_cisd:
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        pyscf_cisd.run()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data['cisd'] = pyscf_cisd
        if verbose:
            print('CISD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.cisd_energy))

    # Run CCSD.
    if run_ccsd:
        pyscf_ccsd = cc.CCSD(pyscf_scf)
        pyscf_ccsd.verbose = 0
        pyscf_ccsd.run()
        molecule.ccsd_energy = pyscf_ccsd.e_tot
        pyscf_data['ccsd'] = pyscf_ccsd
        if verbose:
            print('CCSD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.ccsd_energy))

    # Run FCI.
    if run_fci:
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        spin = (molecule.multiplicity - 1)//2
        pyscf_fci.spin = spin # s2 s(s+1)
        s2 = spin*(spin+1)

        fci.addons.fix_spin_(pyscf_fci, shift=0.1, ss=s2)  # s2
        molecule.fci_energy, fcivec = pyscf_fci.kernel()
        pyscf_data['fci'] = pyscf_fci
        if verbose:
            print('FCI energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.fci_energy))
            print('E = %.12f  2S+1 = %.7f' %
                  (molecule.fci_energy, pyscf_fci.spin_square(fcivec,
                                                              molecule.n_orbitals,
                                                              molecule.n_electrons)[1]))
            print('----------------------')

    # Return updated molecule instance.
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    pyscf_molecular_data.save()
    return pyscf_molecular_data


def generate_molecular_hamiltonian(
        geometry,
        basis,
        multiplicity,
        charge=0,
        n_active_electrons=None,
        n_active_orbitals=None):
    """Generate a molecular Hamiltonian with the given properties.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the charge.
        n_active_electrons: An optional integer specifying the number of
            electrons desired in the active space.
        n_active_orbitals: An optional integer specifying the number of
            spatial orbitals desired in the active space.

    Returns:
        The Hamiltonian as an InteractionOperator.
    """

    # Run electronic structure calculations
    molecule = run_pyscf(
            MolecularData(geometry, basis, multiplicity, charge)
    )

    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)
