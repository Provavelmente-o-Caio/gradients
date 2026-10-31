from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ket.base import Process
from ket.gates import RY, Z, obs
from ket.operations import ExpValue


@dataclass
class GateInstruction:
    gate: Callable
    target: int
    value: float
    index: int | None = None
    multiplier: float = 1.0

    def param_index(self):
        return (self.index, self.multiplier)



def shift_param(gates: list[GateInstruction], shift_inst: int, shift_amt: float):
    shifted_gates = []
    for inst_idx, gate in enumerate(gates):
        if inst_idx == shift_inst:
            shifted_gates.append(GateInstruction(gate.gate, gate.target, gate.value + shift_amt, gate.index, gate.multiplier))
        else:
            shifted_gates.append(gate)

    return shifted_gates


# |ψ(θ)> = RY(θ)|0>
def ansatz(theta: float, qubits, gate=RY):
    _ = gate(theta, qubits)


# |ψ(θ)>
def parametrized_state(num_qubits: int, parameters, gates: list[GateInstruction]):
    processo = Process()
    qubits = processo.alloc(num_qubits)
    for gate in gates:
        ansatz(gate.value, qubits[gate.target], gate.gate)
    return qubits


# <ψ|operator|ψ>
def observable(qubits, operator):
    with obs():
        return operator(qubits)


# f = Cost function
def expectation(num_qubits: int, parameters: list[float], gates:list[GateInstruction], operator=Z):
    qubits = parametrized_state(num_qubits, parameters, gates)
    return ExpValue(observable(qubits, operator)).get()


# [f(θ+s) - f(θ-s)] / (2 sin(s))
# Útil quando sua função de custo recebe *um escalar*.
def parameter_shift_gradient(
    num_qubits: int, gates: list[GateInstruction], parameters: list[float], shift: float = np.pi / 2,
) -> list[float]:
    # forward = cost_function(theta + shift)
    # backward = cost_function(theta - shift)
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
            forward = expectation(num_qubits, parameters, shifted_forward, operator=Z)
            backward = expectation(num_qubits, parameters, shifted_backward, operator=Z)

            total_grad += mult * (forward - backward) / (2 * np.sin(shift))
        gradients.append(total_grad)
    return gradients


if __name__ == "__main__":
    parameters = [0.3, 0.5]

    circuit = [GateInstruction(gate=RY, target=0, value=0.3, index=0), GateInstruction(gate=RY, target=0, value=0.5, index=1)]

    cost = expectation(num_qubits=1, parameters=parameters, gates=circuit, operator=Z)

    grad_psr = parameter_shift_gradient(num_qubits=1, gates=circuit, parameters=parameters)
    grad_analitico = -np.sin(sum(parameters))

    print(f"theta = {parameters}")
    print(f"cost Ket        = {cost:.12f}")
    print(f"cost analitico  = {np.cos(sum(parameters)):.12f}")
    print(f"grad PSR        = {grad_psr}")
    print(f"grad analitico  = {grad_analitico:.12f}")

    assert np.isclose(cost, np.cos(sum(parameters)), atol=1e-8)
    assert np.all(np.isclose(grad_psr, grad_analitico, atol=1e-8))

    print("\nTeste PSR básico passou.")
