import numpy as np
from ket.base import Process
from ket.gates import CNOT, RX, RY, Z, obs
from ket.operations import ExpValue

from gradients.psr.advanced import (
    GateInstruction,
    expectation,
    parameter_shift_gradient,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def finite_diff_gradient(func, params, h=1e-5):
    grads = []
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += h
        p_minus = params.copy()
        p_minus[i] -= h
        grads.append((func(p_plus) - func(p_minus)) / (2 * h))
    return grads


# ---------------------------------------------------------------------------
# Teste 1: Multi-qubit com CNOT
#
# Circuito:  RY(θ₀) → q0,  CNOT(0,1),  RY(θ₁) → q0
# Observable: Z qubit 1
#
# Analítico:  ⟨Z₁⟩ = cos(θ₀)   (CNOT + Z₁ só depende do qubit de controle)
# Gradientes: -sin(θ₀),  0
# ---------------------------------------------------------------------------

def expectation_cnot(params):
    processo = Process()
    qubits = processo.alloc(2)
    RY(params[0], qubits[0])
    CNOT(qubits[0], qubits[1])
    RY(params[1], qubits[0])
    with obs():
        ham = Z(qubits[1])
    return ExpValue(ham).get()


def psr_gradient(func, params, shift=np.pi / 2):
    grads = []
    for i in range(len(params)):
        p_plus = params.copy()
        p_plus[i] += shift
        p_minus = params.copy()
        p_minus[i] -= shift
        e_plus = func(p_plus)
        e_minus = func(p_minus)
        grads.append((e_plus - e_minus) / (2 * np.sin(shift)))
    return grads


def test_multi_qubit_cnot():
    params = [0.5, 0.8]

    cost = expectation_cnot(params)
    cost_analytical = np.cos(params[0])

    grad_psr = psr_gradient(expectation_cnot, params)
    grad_analytical = [
        -np.sin(params[0]),
        0.0,
    ]

    print("=== Teste 1: Multi-qubit CNOT ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")

    assert np.isclose(cost, cost_analytical, atol=1e-6)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-6)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 2: Parâmetros compartilhados
#
# Circuito:  1 qubit,  RY(0.5·θ₀) + RY(1.5·θ₀)
# Mesma param_index=0, multipliers diferentes.
#
# Analítico:  ⟨Z⟩ = cos(2θ₀),  grad = -2·sin(2θ₀)
# ---------------------------------------------------------------------------

def test_shared_parameters():
    theta0 = 0.7
    parameters = [theta0]

    circuit = [
        GateInstruction(gate=RY, target=0, value=0.5 * theta0, index=0, multiplier=0.5),
        GateInstruction(gate=RY, target=0, value=1.5 * theta0, index=0, multiplier=1.5),
    ]

    cost = expectation(num_qubits=1, parameters=parameters, gates=circuit, operator=Z)
    cost_analytical = np.cos(2 * theta0)

    grad_psr = parameter_shift_gradient(num_qubits=1, gates=circuit, parameters=parameters)
    grad_analytical = [-2 * np.sin(2 * theta0)]

    print("=== Teste 2: Parâmetros compartilhados ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 3: Finite differences vs PSR (multi-qubit CNOT)
#
# Usa o circuito do teste 1, mas compara PSR com gradiente numérico.
# ---------------------------------------------------------------------------

def test_finite_diff_multi_qubit():
    params = [0.5, 0.8]

    grad_psr = psr_gradient(expectation_cnot, params)
    grad_fd = finite_diff_gradient(expectation_cnot, params, h=0.1)

    print("=== Teste 3: Finite differences vs PSR (multi-qubit) ===")
    print(f"  grad PSR = {grad_psr}")
    print(f"  grad FD  = {grad_fd}")

    assert np.allclose(grad_psr, grad_fd, atol=0.05)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 4: Gate types variados (RX + RY)
#
# Circuito:  1 qubit,  RX(θ₀) + RY(θ₁)
# Observable: Z
#
# Analítico:  ⟨Z⟩ = cos(θ₀)·cos(θ₁)
# Gradientes: -sin(θ₀)·cos(θ₁),  -cos(θ₀)·sin(θ₁)
#
# Verificação extra: compara PSR com finite differences.
# ---------------------------------------------------------------------------

def expectation_rx_ry(params):
    processo = Process()
    qubits = processo.alloc(1)
    RX(params[0], qubits[0])
    RY(params[1], qubits[0])
    with obs():
        ham = Z(qubits[0])
    return ExpValue(ham).get()


def test_rx_ry_gates():
    params = [0.4, 1.1]

    cost = expectation_rx_ry(params)
    cost_analytical = np.cos(params[0]) * np.cos(params[1])

    grad_psr = psr_gradient(expectation_rx_ry, params)
    grad_analytical = [
        -np.sin(params[0]) * np.cos(params[1]),
        -np.cos(params[0]) * np.sin(params[1]),
    ]
    grad_fd = finite_diff_gradient(expectation_rx_ry, params, h=0.1)

    print("=== Teste 4: RX + RY gates ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")
    print(f"  grad FD        = {grad_fd}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-8)
    assert np.allclose(grad_psr, grad_fd, atol=0.05)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 5: 3 qubits com CNOTs
#
# Circuito:  RY(θ₀)→q0,  RY(θ₁)→q1,  RY(θ₂)→q2,
#            CNOT(q0,q1),  CNOT(q1,q2)
# Observable: Z qubit 0
#
# Analítico:  ⟨Z₀⟩ = cos(θ₀)
#              (CNOT propaga amplitude, mas Z em q0 só depende de θ₀)
# Gradientes: -sin(θ₀), 0, 0
#
# Verifica:  cost analítico, PSR vs analítico, PSR vs finite differences.
# ---------------------------------------------------------------------------

def expectation_3q(params):
    processo = Process()
    qubits = processo.alloc(3)
    RY(params[0], qubits[0])
    RY(params[1], qubits[1])
    RY(params[2], qubits[2])
    CNOT(qubits[0], qubits[1])
    CNOT(qubits[1], qubits[2])
    with obs():
        ham = Z(qubits[0])
    return ExpValue(ham).get()


def test_3_qubits_cnot():
    params = [0.6, 1.1, 0.3]

    cost = expectation_3q(params)
    cost_analytical = np.cos(params[0])

    grad_psr = psr_gradient(expectation_3q, params)
    grad_analytical = [-np.sin(params[0]), 0.0, 0.0]
    grad_fd = finite_diff_gradient(expectation_3q, params, h=0.1)

    print("=== Teste 5: 3 qubits CNOT ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")
    print(f"  grad FD        = {grad_fd}")

    assert np.isclose(cost, cost_analytical, atol=1e-6)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-6)
    assert np.allclose(grad_psr, grad_fd, atol=0.05)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_multi_qubit_cnot()
    test_shared_parameters()
    test_finite_diff_multi_qubit()
    test_rx_ry_gates()
    test_3_qubits_cnot()
    print("Todos os testes avançados passaram.")
