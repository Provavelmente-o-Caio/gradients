from collections.abc import Callable

import numpy as np
from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import RY, Z, obs
from ket.operations import exp_value

from gradients.utils.utils import ObservableBuilder, ParameterizedGate


# |ψ(θ)> = RY(θ)|0>
def ansatz(
    theta: float,
    qubits: Quant,
    gate: ParameterizedGate = RY,
) -> None:
    gate(theta, qubits[0])


# |ψ(θ)>
def parametrized_state(
    theta: float,
    gate: ParameterizedGate = RY,
) -> Quant:
    process = Process()
    qubits = process.alloc(1)
    ansatz(theta, qubits, gate)
    return qubits


# <ψ|operator|ψ>
def observable(
    qubits: Quant,
    operator: ObservableBuilder,
) -> Hamiltonian | Pauli:
    with obs():
        return operator(qubits[0])


# f = Cost function
def expectation(
    theta: float,
    gate: ParameterizedGate = RY,
    operator: ObservableBuilder = Z,
) -> float:
    qubits = parametrized_state(theta, gate)
    return exp_value(observable(qubits, operator)).get()


# [f(θ+s) - f(θ-s)] / (2 sin(s))
# Útil quando sua função de custo recebe *um escalar*.
def parameter_shift_gradient(
    cost_function: Callable[[float], float],
    theta: float,
    s: float = np.pi / 2,
) -> float:
    forward = cost_function(theta + s)
    backward = cost_function(theta - s)
    return (forward - backward) / (2 * np.sin(s))
