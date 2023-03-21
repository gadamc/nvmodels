from typing import Optional, Tuple, Sequence
import itertools

import qutip
import numpy as np

def qutritdm_to_qubitket(state: qutip.Qobj, plus_zero = True) -> qutip.Qobj:
    """
    Trace out a 3-level density matrix to a 2-level ket.
    If plus_zero == True, then will trace out the |0> -> |+1> component
    If plus_zero == False, will trace out the |0> -> |-1> component

    In both cases (plus_zero == True or False),
    the |0> state will correspond to (1, 0)
    and the |+-1> state will project to (0, 1)

    This keeps with quantum information processing convention such that
    sigma_z |0> = |0>
    sigma_z |1> = -|1>

    One should note that this is now confusing due to the fact that

    sigma_plus |0> --> (0, 0)
    sigma_plus |1> --> |0>

    This is consistent with the convention where |0> correponsds, physically
    to |m_s = +1/2> and |1> = |m_s = -1/2> for a two level system.
    But in our case, |0> is the |m_s = 0>, so one would expect
    sigma_plus |m_s = 0> = |m_s = 1>

    Basically, don't operate on this 2-level representation and expect
    the results to be physically correct.

    See Qutip documentation on the subject:
    https://qutip.org/docs/latest/guide/guide-states.html#qubit-two-level-systems

    """
    if state.type != 'oper':
        raise TypeError(f'state must be a density matrix, found type {state.type}')

    if plus_zero:
        state = state.full()[:2, :2]
        b = np.sqrt(state[0, 0]) # choose that |0> is always (1,0)
        a = np.sqrt(state[1, 1])
    else:
        state = state.full()[1:3, 1:3]
        a = np.sqrt(state[0, 0])
        b = np.sqrt(state[1, 1])

    s01 = state[0, 1]
    if a*b != 0:
        phase = - np.arctan2(np.imag(s01 / a*b), np.real(s01 / a*b))
        traced_state = a * qutip.basis(2, 0) + b * np.exp(1j * phase) * qutip.basis(2, 1)
    else:
        if a == 0:
            traced_state = b * qutip.basis(2, 1)
        else:
            traced_state = a * qutip.basis(2, 0)

    return traced_state


def N_qutritdm_to_qubitket(state: qutip.Qobj, ith_qutrit: int, plus_zero=True) -> qutip.Qobj:
    """
    Trace out from an N-qutrit system to a 2-level ket for the ith qutrit of the system.

    Calls qutritdm_to_qubitket(qutip.ptrace(state, ith_qutrit), plus_zero)

    See docstring for qutritdm_to_qubitket.
    """

    return qutritdm_to_qubitket(qutip.ptrace(state, ith_qutrit), plus_zero)


# should be easy to write an NQutrit_state_labels function
def two_qutrit_eigenstate_labels():
    """
    returns the following list of strings,
    representing the spin states of a two qutrit system (labeled 1, 0, -1)

    ['|1,1>',
     '|1,0>',
     '|1,-1>',
     '|0,1>',
     '|0,0>',
     '|0,-1>',
     '|-1,1>',
     '|-1,0>',
     '|-1,-1>']
    """
    return [f'|{x[0]},{x[1]}>' for x in list(itertools.product([1,0,-1],[1,0,-1]))]

def two_qutrit_state_to_text(state: qutip.Qobj, decimals=3):
    """
    Returns text representation of the state as a linear
    combination of 2-qutrit eigenstates.

    First, the state is rounded to nearest decimals so that
    small contributions are excluded.
    """
    states = []
    labels = two_qutrit_eigenstate_labels()
    probs = np.abs(state.full().flatten().round(decimals=decimals)) ** 2
    for prob, label in zip(probs, labels):
        if prob > 0:
            states.append(f'{np.sqrt(prob):.3f}{label}')
    return " + ".join(states)

def spin1dm2text(rho: qutip.Qobj, decimals=5) -> Tuple[Optional[str], Optional[qutip.Qobj]]:

    """
    Attempts to determine if rho matches close enough to a particular
    pure state for spin 1 system within some tolerance defined by decimals.
    If a match is found, a string and the matched density matrix is returned.

    If no match is found, then ('NA', rho_rounded) is returned,
    where

    rho_rounded = qutip.Qobj(rho.full().round(decimals=decimals))

    We define the following pure states we try to match:

    1   -- the m_s = 1 state
    0   -- the m_s = 0 state
    -1   -- the m_s = -1 state

    +(up arrow)    -- the state   (|0> + |+1>)/sqrt(2)
    +(down arrow)  -- the state   (|0> + |-1>)/sqrt(2)

    -(up arrow)    -- the state   (|0> - |+1>)/sqrt(2)
    -(down arrow)  -- the state   (|0> - |-1>)/sqrt(2)

    +j(up arrow)   -- the state   (|0> + j|+1>)/sqrt(2)
    +j(down arrow) -- the state   (|0> + j|-1>)/sqrt(2)

    -j(up arrow)   -- the state   (|0> - j|+1>)/sqrt(2)
    -j(down arrow) -- the state   (|0> - j|-1>)/sqrt(2)

    p   -- the state   (|1> + |-1>)/sqrt(2)
    m   -- the state   (|1> - |-1>)/sqrt(2)

    pi   -- the state (|1> + j|-1>)/sqrt(2)
    mu   -- the state (|1> - j|-1>)/sqrt(2)

    p+m   -- the density matrix  |p><p| + |m><m|
    p-m   -- the density matrix  |p><p| - |m><m|

    """

    if rho.type != 'oper':
        raise TypeError(f'state must be a density matrix, found type {rho.type}')

    rho = qutip.Qobj(rho.full().round(decimals=decimals))

    up_arrow = u'\u2191'
    down_arrow = u'\u2193'
    pi = u'\u03C0'
    mu =  u'\u03BC'
    defined_states = [
        (qutip.basis(3,0), "1"),
        (qutip.basis(3,1), "0"),
        (qutip.basis(3,2), "-1"),
        ((qutip.basis(3,1) + qutip.basis(3,0)).unit(), "+" + up_arrow),
        ((qutip.basis(3,1) + qutip.basis(3,2)).unit(), "+" + down_arrow),
        ((qutip.basis(3,1) - qutip.basis(3,0)).unit(), "-" + up_arrow),
        ((qutip.basis(3,1) - qutip.basis(3,2)).unit(), "-" + down_arrow),
        ((qutip.basis(3,1) + 1j*qutip.basis(3,0)).unit(), "+j" + up_arrow),
        ((qutip.basis(3,1) + 1j*qutip.basis(3,2)).unit(), "+j" + down_arrow),
        ((qutip.basis(3,1) - 1j*qutip.basis(3,0)).unit(), "-j" + up_arrow),
        ((qutip.basis(3,1) - 1j*qutip.basis(3,2)).unit(), "-j" + down_arrow),
        ((qutip.basis(3,0) + qutip.basis(3,2)).unit(), "p"),
        ((qutip.basis(3,0) - qutip.basis(3,2)).unit(), "m"),
        ((qutip.basis(3,0) + 1j*qutip.basis(3,2)).unit(), pi),
        ((qutip.basis(3,0) - 1j*qutip.basis(3,2)).unit(), mu),
        ((qutip.ket2dm((qutip.basis(3,0) + qutip.basis(3,2)).unit()) + \
          qutip.ket2dm((qutip.basis(3,0) - qutip.basis(3,2)).unit())).unit(), "p+m"),
        ((qutip.ket2dm((qutip.basis(3,0) + qutip.basis(3,2)).unit()) - \
          qutip.ket2dm((qutip.basis(3,0) - qutip.basis(3,2)).unit())).unit(), "p-m"),

    ]

    for ket, text in defined_states:
        if ket.type != 'oper':
            rho_test = qutip.ket2dm(ket)
        else:
            rho_test = ket #already a density matrix

        if rho == rho_test:
            return text, rho_test

    return 'NA', rho
