import itertools
import numpy as np
import scipy.constants
import qutip
import matplotlib.pyplot as plt

import nvmodels
"""

This module contains functions for a bespoke simple simulation of the NV- center CWODMR experiment.

The primary simplification is the number of states included in the simulation. Only the
ground and excited electronic states of the NV- center, and the inter-system crossing state are included. 
The nuclear spin states are not included.

This is still a work in progress. Does not yet include t2* dephasing or a dutuned MW field. 

See the accompanying Jupyter notebook, notebooks/simple_cwodmr.ipynb for more details.
"""


nvgs = nvmodels.NVNegativeGroundState(include_nuclear_states=False)
nves = nvmodels.NVNegativeExcitedState(include_nuclear_states=False)


def hamiltonian(static_magnetic_field):
    """
    Returns the Hamiltonian for an NV- center in the negative charge state,
    with no nuclear spin states included.

    The ground, excited and an inter-system crossing state are combined into a single Hamiltonian

    ```
    |  H_gs        |
    |       H_es   |
    |            S |
    ```

    H_gs = hamiltonian for the ground state
    H_es = hamiltonian for the excited state
    S = inter-system crossing hamiltonian (1 state)

    H_full = H_gs + H_es + S is represented as a 7 x 7 matrix

    NB - The excited electronic state hamiltonian does NOT include the ~470 THz energy difference
    above the ground state. However, 6 GHz is added to H_es in order to raise
    the excited states above the ground state. This will assist in visualizing the states later

    :param static_magnetic_field: The static magnetic field in Tesla.

    :return: (H_gs, H_es, H_full, state_id) where H_gs is the Hamiltonian for the ground state,
    H_es is the Hamiltonian for the excited state, H_full is the full Hamiltonian

    """

    artificial_energy_separation = 10e9

    h_gs = nvgs.zero_field_hamiltonian()
    h_gs += nvgs.nitrogen_hyperfine_hamiltonian()  # with include nuclear states = False, this line actually does nothing. But it is kept it here for future demonstrations
    h_gs += nvgs.nitrogen_electric_quadrupole_hamiltonian()  # with include nuclear states = False, this line actually does nothing. But it is kept it here for future demonstrations
    h_gs += nvgs.static_mag_field_hamiltonian(static_magnetic_field)

    h_es = nves.zero_field_hamiltonian()
    h_es += nves.nitrogen_hyperfine_hamiltonian()  # with include nuclear states = False, this line actually does nothing. But it is kept it here for future demonstrations
    h_es += nves.nitrogen_electric_quadrupole_hamiltonian()  # with include nuclear states = False, this line actually does nothing. But it is kept it here for future demonstrations
    h_es += nves.static_mag_field_hamiltonian(static_magnetic_field)

    h_es += artificial_energy_separation * qutip.qeye(3)  # add E to the excited state so can properly separate

    num_isc_states = 1   #  ignore the 1E ISC state, which has a lifetime of 1ns   + 2 * nvgs.include_nuclear_states

    full_h_shape = h_gs.shape[0] + h_es.shape[0] + num_isc_states

    hh = np.zeros((full_h_shape, full_h_shape), dtype=np.cdouble)

    gs_idx = [0, h_gs.shape[0], 0, h_gs.shape[1]]
    hh[gs_idx[0]:gs_idx[1], gs_idx[2]:gs_idx[3]] = h_gs

    es_idx = [gs_idx[1], gs_idx[1] + h_es.shape[0], gs_idx[3], gs_idx[3] + h_es.shape[1]]
    hh[es_idx[0]:es_idx[1], es_idx[2]:es_idx[3]] = h_es

    ss_idx = [es_idx[1], es_idx[1] + num_isc_states, es_idx[3], es_idx[3] + num_isc_states]
    hh[ss_idx[0]:ss_idx[1], ss_idx[2]:ss_idx[3]] = np.eye(num_isc_states) * artificial_energy_separation/2  #  artificially raise the E of the ISC so that we can visualize the states later

    return h_gs, h_es, qutip.Qobj(hh)

def state_ids():
    """
    State index

    * 0  -->  3A2, ground state, ms = 0
    * 1-2 --> 3A2, ground state, ms = +-1

    * 3 --> A1, ISC crossing state


    * 4  -->  3E, excited state, ms = 0
    * 5-6 --> 3E, excited state, ms = +-1


    Energy level diagram with state index


    ```
    ms = 0
                     ---- 6 (ms = +1)
                     ---- 5 (ms = -1)
    ---- 4
              ISC
            ---- 3
                    ----- 2 (ms = +1)
                    ----- 1 (ms = -1)
    ---- 0
    ```
    """
    return dict(gs_0=0, gs_m1=1, gs_p1=2, isc=3, es_0=4, es_m1=5, es_p1=6)


def print_and_examine_eigenstates(hamiltonian):
    """
    Prints the eigenvalues and eigenstates of a Hamiltonian, and returns a matplotlib figure and axis
    """
    print(hamiltonian)
    eigenvalues, eigenstates = hamiltonian.eigenstates()
    fig, ax = nvmodels.plotting.plot_eigenspectrum_mpl(eigenvalues, None)
    ax.set_ylabel('Energy (Hz)')
    ax.set_xlabel('State index')
    return fig, ax


def projection(s, from_l, to_m):
    return qutip.basis(s, to_m)*qutip.basis(s, from_l).dag()

def build_Lindblad_operators(gamma):
    """
    Build the Lindblad operators for a given transfer matrix, gamma
    gamma should be a square matrix

    returns a dictionary of the form {f'{l}->{m}': L_lm}

    Pass L.values() to qutip.mesolve
    """
    assert gamma.shape[0] == gamma.shape[1]

    return {f'{l}->{m}': np.sqrt(gamma[l, m]) * projection(gamma.shape[0], l, m)
         for l, m in itertools.product(range(gamma.shape[0]), range(gamma.shape[1])) if gamma[l, m] != 0}

def pl_rate_observable(gamma):
    """
    Build the observables for a given transfer matrix, gamma
    gamma should be a square matrix
    returns the following list of observables:
    - population of the ground state
    - population of the excited state
    - population of the inter-system crossing state
    """

    assert gamma.shape[0] == gamma.shape[1]

    state_id = state_ids()

    pl_rate = gamma[state_id['es_0'], state_id['gs_0']] * projection(gamma.shape[0], state_id['es_0'], state_id['es_0']) / gamma[state_id['es_0'], :].sum()
    pl_rate += gamma[state_id['es_m1'], state_id['gs_m1']] * projection(gamma.shape[0], state_id['es_m1'], state_id['es_m1']) / gamma[state_id['es_m1'], :].sum()
    pl_rate += gamma[state_id['es_p1'], state_id['gs_p1']] * projection(gamma.shape[0], state_id['es_p1'], state_id['es_p1']) / gamma[state_id['es_p1'], :].sum()

    return pl_rate

def state_population_observables(num_states):
    """
    Build the projection operators each of the states in the system

    p_gs_ms_0 = population of the ground state ms = 0
    p_gs_ms_p1 = population of the ground state ms = +1
    p_gs_ms_m1 = population of the ground state ms = -1

    p_es_ms_0 = population of the excited state ms = 0
    p_es_ms_p1 = population of the excited state ms = +1
    p_es_ms_m1 = population of the excited state ms = -1

    returns p_gs_ms_0, p_gs_ms_p1, p_gs_ms_m1, p_es_ms_0, p_es_ms_p1, p_es_ms_m1
    """
    return [projection(num_states, i, i) for i in range(num_states)]


def add_rabi_oscillation_to_hamiltonian(hamiltonian, rabi_frequency, rabi_plus_state='gs_m1', rabi_phase=0):
    """
    Add a Rabi oscillation to the Hamiltonian

    rabi_frequency: Rabi frequency in Hz
    rabi_phase: Rabi phase in radians
    """
    state_id = state_ids()
    hamiltonian += rabi_frequency/2 * (np.cos(rabi_phase) * projection(hamiltonian.shape[0], state_id['gs_0'], state_id[rabi_plus_state]) +
                                     -1j*np.sin(rabi_phase) * projection(hamiltonian.shape[0], state_id['gs_0'], state_id[rabi_plus_state]))

    hamiltonian += rabi_frequency / 2 * (np.cos(rabi_phase) * projection(hamiltonian.shape[0], state_id[rabi_plus_state], state_id['gs_0']) +
                +1j*np.sin(rabi_phase) * projection(hamiltonian.shape[0], state_id[rabi_plus_state], state_id['gs_0']))

    return hamiltonian

def build_transfer_rates(h_full, optical_pumping_watts=3, optical_pumping_rate_per_watt=1.1e6,
                         rabi_frequency=1e6, qubit_plus_one_state='gs_m1',
                         spontaneous_emission=1/15.8e-9, t_1_relaxation=1./3e-3,
                         t_2_dephasing=1./3e-6,
                         excited_ms1_to_isc=1/12.5e-9, excited_ms0_to_isc=1/76.9e-9,
                         isc_to_gs_ms0=1/289e-9, isc_to_gs_ms1=1/463e-9):
    """
    Build the transfer rates for the system.

    The qubit |0> state is the ground state ms = 0
    The qubit |1> state is specified by the qubit_plus_one_state parameter and should be one of 'gs_m1', 'gs_p1',
    (though in principle could use any other state: see state_ids() for the state names)

    """

    gamma = np.zeros(h_full.shape)

    state_id = state_ids()

    es_0 = state_id['es_0']
    gs_0 = state_id['gs_0']
    es_m1 = state_id['es_m1']
    gs_m1 = state_id['gs_m1']
    es_p1 = state_id['es_p1']
    gs_p1 = state_id['gs_p1']
    isc = state_id['isc']

    optical_pumping_rate = optical_pumping_rate_per_watt * optical_pumping_watts  # optical pumping rate (example: 1/285e-9 = 3.5 MHz
    gamma[gs_0, es_0] += optical_pumping_rate
    gamma[gs_m1, es_m1] += optical_pumping_rate
    gamma[gs_p1, es_p1] += optical_pumping_rate

    gamma[es_0, gs_0] += spontaneous_emission
    gamma[es_m1, gs_m1] += spontaneous_emission  # this may be slightly different, 1/15.9e-9
    gamma[es_p1, gs_p1] += spontaneous_emission

    # Boltzmann relaxation
    # decay from g, ms=+-1 -> g ms=0
    gamma[gs_m1, gs_0] += t_1_relaxation  # milliseconds
    gamma[gs_p1, gs_0] += gamma[gs_m1, gs_0]

    # rate the other way is relatively affected by energy difference
    room_temp = scipy.constants.physical_constants['Boltzmann constant in Hz/K'][0] * nvgs.temperature
    gamma[gs_0, gs_m1] += gamma[gs_m1, gs_0] * np.exp(-nvgs.zero_field_splitting() / room_temp)
    gamma[gs_0, gs_p1] += gamma[gs_p1, gs_0] * np.exp(-nvgs.zero_field_splitting() / room_temp)

    # ISC crossing from 3E to A1
    gamma[es_m1, isc] += excited_ms1_to_isc
    gamma[es_p1, isc] += excited_ms1_to_isc
    gamma[es_0, isc] += excited_ms0_to_isc

    # ISC from A1 to 3A2
    gamma[isc, gs_0] += isc_to_gs_ms0
    gamma[isc, gs_m1] += isc_to_gs_ms1
    gamma[isc, gs_p1] += isc_to_gs_ms1

    # # Rabi oscillations
    # gamma[gs_0, state_id[qubit_plus_one_state]] += rabi_frequency
    # gamma[state_id[qubit_plus_one_state], gs_0] += rabi_frequency

    # dephasing
    gamma[state_id[qubit_plus_one_state], state_id[qubit_plus_one_state]] += t_2_dephasing


    return gamma


def psi0_ground_state(h_full):
    state_id = state_ids()
    psi0 = 0.34 * projection(h_full.shape[0], state_id["gs_0"], state_id["gs_0"])
    psi0 += 0.33 * projection(h_full.shape[0], state_id["gs_p1"], state_id["gs_p1"])
    psi0 += 0.33 * projection(h_full.shape[0], state_id["gs_m1"], state_id["gs_m1"])
    return psi0


def psi0_from_results(observables):
    """
    :param observables: list of results.expect values. Should be the observables from `state_population_observables`
    :return: psi0 for the next simulation
    """
    num_states = len(observables)
    psi0 = [obs[-1] * projection(num_states, i, i) for i, obs in enumerate(observables)]
    psi0 = sum(psi0)
    return psi0


def plot_observables(times, observables, labels, **subplot_kwargs):
    """
    :param times: list of times
    :param observables: list of results.expect values. Could be the observables from `state_population_observables`
    :param labels: list of labels for the observables
    """
    fig, ax = plt.subplots(**subplot_kwargs)
    for label, obs in zip(labels, observables):
        ax.plot(times, obs, label=label)
    ax.set_xlabel('time')
    ax.legend()

    return fig, ax