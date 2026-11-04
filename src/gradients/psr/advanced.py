from collections.abc import Sequence

import numpy as np
from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import RY, Z, obs
from ket.operations import exp_value

from gradients.utils.utils import (
    GateInstruction,
    ObservableBuilder,
    ParameterizedGate,
    shift_param,
)


# |ψ(θ)> = RY(θ)|0>
def ansatz(theta: float, qubits: Quant, gate: ParameterizedGate = RY) -> None:
    _ = gate(theta, qubits)


# |ψ(θ)>
def parametrized_state(
    num_qubits: int,
    parameters: Sequence[float],
    gates: Sequence[GateInstruction],
) -> Quant:
    process = Process()
    qubits = process.alloc(num_qubits)
    for gate in gates:
        ansatz(gate.value, qubits[gate.target], gate.gate)
    return qubits


# <ψ|operator|ψ>
def observable(qubits: Quant, operator: ObservableBuilder) -> Hamiltonian | Pauli:
    with obs():
        return operator(qubits)


# f = Cost function
def expectation(
    num_qubits: int,
    parameters: Sequence[float],
    gates: Sequence[GateInstruction],
    operator: ObservableBuilder = Z,
) -> float:
    qubits = parametrized_state(num_qubits, parameters, gates)
    return exp_value(observable(qubits, operator)).get()


# [f(θ+s) - f(θ-s)] / (2 sin(s))
# Útil quando sua função de custo recebe *um escalar*.
def parameter_shift_gradient(
    num_qubits: int,
    gates: Sequence[GateInstruction],
    parameters: Sequence[float],
    shift: float = np.pi / 2,
    operator: ObservableBuilder = Z,
) -> list[float]:
    gradients: list[float] = []
    for param_idx, _ in enumerate(parameters):
        multipliers: list[tuple[int, float]] = []
        for inst_idx, gate in enumerate(gates):
            idx, mult = gate.param_index()
            if idx is not None and mult and idx == param_idx:
                multipliers.append((inst_idx, mult))
        if len(multipliers) == 0:
            continue
        total_grad = 0.0
        for inst_idx, mult in multipliers:
            shifted_forward = shift_param(gates, inst_idx, shift)
            shifted_backward = shift_param(gates, inst_idx, -shift)
            forward = expectation(
                num_qubits, parameters, shifted_forward, operator=operator
            )
            backward = expectation(
                num_qubits, parameters, shifted_backward, operator=operator
            )

            total_grad += mult * (forward - backward) / (2 * np.sin(shift))
        gradients.append(total_grad)
    return gradients
