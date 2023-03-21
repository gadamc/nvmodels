from typing import Sequence

import plotly.graph_objs as go

import qutip
import numpy as np

import nvmodels.utilities

def plot_eigenspectrum(energies: Sequence[float], labels: Sequence[str]=None,
                       ylabel=None, yscale=1., fig=None):
    """
    Plots energies as a function of eigenstate, with labels that give the
    probabilities in the uncoupled basis

    :param energies:
    :param labels:
    :param y_label:
    :return:
    """

    if fig is None:
        fig = go.Figure()

    scatter = go.Scatter(y=energies, hovertext=labels,
                         line={'width':0}, mode='markers',
                         marker=dict(size=12,line=dict(width=2,
                         color='DarkSlateGrey')
                         )
    )
    fig.update_yaxes(title_text=ylabel)
    fig.update_xaxes(title_text='state index')
    fig.add_trace(scatter)
    fig.update_yaxes(tickformat="s")
    return fig

def plot_states_on_Bloch(states, subsystem, plus_zero,
                         color = "b", start_color = "green", end_color = "red",
                         bloch = None, bloch_kwargs = {}):
    """
    Creates a qutip.Bloch and adds the subsystem from each of the states.

    Expects that states is a list of N-qutrit states (electron, nitrogen
    spin states of the NV center, plus any other subsystems such as
    nearby 13C). The `subsystem` is used to select
    either the electron (`subsystem = 0`) or nitrogen (`subsystem = 1`) from each
    state in list states.

    For each state, uses the nvmodels.utiliies.qutritdm_to_qubitket function
    to project either the m_s = 0 and m_s = +1 states to a two-level system
    (set `plus_zero = True`) or the m_s = 0  and m_s = -1 states to a two-level
    system (set `plus_zero = False`).

    The first state in the list is displayed with color `start_color`.
    The final state in the list is displayed with color `end_color`.
    All other states are displayed with `color` with varying degrees of 'alpha'
    transparency to show progression over time (more transparent being early
    in time).

    Will plot states on `bloch` if not None, otherwise will instantiate a
    new qutip.Bloch object.

    `bloch_kwargs` is passed to qutip.Bloch instatiation if `bloch == None`.
    """
    if bloch == None:
        bloch = qutip.Bloch(**bloch_kwargs)
        bloch.vector_color = [color] * len(states)
        bloch.vector_color[0] = start_color
        bloch.vector_color[-1] = end_color
    else:
        current_len = len(bloch.vector_color)
        bloch.vector_color += [color] * len(states)
        bloch.vector_color[current_len] = start_color
        bloch.vector_color[-1] = end_color

    for i, s in enumerate(states):
        s = nvmodels.utilities.N_qutritdm_to_qubitket(s, subsystem, plus_zero = plus_zero)
        if i == 0:
            alpha = 0.5
        else:
            alpha = 0.2 + 0.8 * (i / len(bloch.vector_color))
        bloch.add_states([s], alpha=alpha)

    return bloch
