import numpy as np
from ket.base import Process
from ket.gates import RY, Z, obs
from ket.operations import ExpValue


# |ψ(θ)> = RY(θ)|0>
def ansatz(theta: float, qubits, gate=RY):
    gate(theta, qubits[0])


# |ψ(θ)>
def parametrized_state(theta: float, gate=RY):
    processo = Process()
    qubits = processo.alloc(1)
    ansatz(theta, qubits, gate)
    return qubits


# <ψ|operator|ψ>
def observable(qubits, operator):
    with obs():
        return operator(qubits[0])


# f = Cost function
def expectation(theta: float, gate=RY, operator=Z):
    qubits = parametrized_state(theta, gate)
    return ExpValue(observable(qubits, operator)).get()


# [f(θ+s) - f(θ-s)] / (2 sin(s))
# Útil quando sua função de custo recebe *um escalar*.
def parameter_shift_gradient(
    cost_function, theta: float, s: float = np.pi / 2
) -> float:
    forward = cost_function(theta + s)
    backward = cost_function(theta - s)
    return (forward - backward) / (2 * np.sin(s))


if __name__ == "__main__":
    theta = 0.3

    cost = expectation(theta)
    grad_psr = parameter_shift_gradient(expectation, theta)
    grad_analitico = -np.sin(theta)

    print(f"theta = {theta}")
    print(f"cost Ket        = {cost:.12f}")
    print(f"cost analitico  = {np.cos(theta):.12f}")
    print(f"grad PSR        = {grad_psr:.12f}")
    print(f"grad analitico  = {grad_analitico:.12f}")

    assert np.isclose(cost, np.cos(theta), atol=1e-8)
    assert np.isclose(grad_psr, grad_analitico, atol=1e-8)

    print("\nTeste PSR básico passou.")
