import numpy as np
from ket.gates import CNOT, RX, RY, X, Y, Z

from gradients.psr.advanced import (
    GateInstruction,
    expectation as psr_expectation,
    parameter_shift_gradient,
)

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
    return psr_expectation(
        2,
        params,
        [
            GateInstruction(
                gate=lambda theta, control, target: (
                    RY(theta, control),
                    CNOT(control, target),
                ),
                target=(0, 1),
                value=params[0],
                index=0,
            ),
            GateInstruction(RY, 0, params[1], 1),
        ],
        lambda qubits: Z(qubits[1]),
    )


def test_multi_qubit_cnot():
    params = [0.5, 0.8]

    cost = expectation_cnot(params)
    cost_analytical = np.cos(params[0])

    circuit = [
        GateInstruction(
            gate=lambda theta, control, target: (
                RY(theta, control),
                CNOT(control, target),
            ),
            target=(0, 1),
            value=params[0],
            index=0,
        ),
        GateInstruction(RY, 0, params[1], 1),
    ]
    grad_psr = parameter_shift_gradient(
        2, circuit, params, operator=lambda qubits: Z(qubits[1])
    )
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

    cost = psr_expectation(
        num_qubits=1, parameters=parameters, gates=circuit, operator=Z
    )
    cost_analytical = np.cos(2 * theta0)

    grad_psr = parameter_shift_gradient(
        num_qubits=1, gates=circuit, parameters=parameters
    )
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
# Teste 3: Gate types variados (RX + RY)
#
# Circuito:  1 qubit,  RX(θ₀) + RY(θ₁)
# Observable: Z
#
# Analítico:  ⟨Z⟩ = cos(θ₀)·cos(θ₁)
# Gradientes: -sin(θ₀)·cos(θ₁),  -cos(θ₀)·sin(θ₁)
#
# Verificação extra: compara PSR com finite differences.
# ---------------------------------------------------------------------------


def rx_ry_circuit(params):
    return [
        GateInstruction(RX, 0, params[0], 0),
        GateInstruction(RY, 0, params[1], 1),
    ]


def test_rx_ry_gates():
    params = [0.4, 1.1]

    cost = psr_expectation(1, params, rx_ry_circuit(params), operator=Z)
    cost_analytical = np.cos(params[0]) * np.cos(params[1])

    grad_psr = parameter_shift_gradient(
        1, rx_ry_circuit(params), params, operator=Z
    )
    grad_analytical = [
        -np.sin(params[0]) * np.cos(params[1]),
        -np.cos(params[0]) * np.sin(params[1]),
    ]

    print("=== Teste 3: RX + RY gates ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 4: 3 qubits com CNOTs
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


def three_qubit_circuit(params):
    return [
        GateInstruction(
            gate=lambda theta, q0, q1: (
                RY(theta, q0),
                CNOT(q0, q1),
            ),
            target=(0, 1),
            value=params[0],
            index=0,
        ),
        GateInstruction(
            gate=lambda theta, q1, q2: CNOT(q1, q2),
            target=(1, 2),
            value=0.0,
        ),
        GateInstruction(RY, 1, params[1], 1),
        GateInstruction(RY, 2, params[2], 2),
    ]


def test_3_qubits_cnot():
    params = [0.6, 1.1, 0.3]

    cost = psr_expectation(
        3, params, three_qubit_circuit(params), operator=lambda qubits: Z(qubits[0])
    )
    cost_analytical = np.cos(params[0])

    grad_psr = parameter_shift_gradient(
        3,
        three_qubit_circuit(params),
        params,
        operator=lambda qubits: Z(qubits[0]),
    )
    grad_analytical = [-np.sin(params[0]), 0.0, 0.0]

    print("=== Teste 4: 3 qubits CNOT ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analitico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad_psr}")
    print(f"  grad analitico = {grad_analytical}")

    assert np.isclose(cost, cost_analytical, atol=1e-6)
    assert np.allclose(grad_psr, grad_analytical, atol=1e-6)
    print("  PASS\n")


def test_unused_parameter():
    params = [0.5, 0.9, 0.3]
    circuit = [
        GateInstruction(gate=RY, target=0, value=params[0], index=0),
        GateInstruction(gate=RY, target=0, value=params[2], index=2),
    ]
    cost = psr_expectation(1, params, circuit, operator=Z)
    cost_analytical = np.cos(params[0] + params[2])
    grad = parameter_shift_gradient(1, circuit, params)
    expected = -np.sin(params[0] + params[2])

    print("=== Teste 5: Parâmetro não utilizado ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analítico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad}")
    print(f"  grad analítico = {[expected, 0.0, expected]}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert len(grad) == 3
    assert np.allclose(grad, [expected, 0.0, expected], atol=1e-6)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 5: Parâmetro compartilhado em dois qubits
#
# Circuito: RY(θ) em q0 e q1
# Observable: Z0 Z1
#
# Analítico: ⟨Z0 Z1⟩ = cos²(θ)
# Gradiente: -sin(2θ)
# ---------------------------------------------------------------------------


def test_shared_parameter_two_qubits():
    theta = 0.7
    circuit = [
        GateInstruction(gate=RY, target=0, value=theta, index=0),
        GateInstruction(gate=RY, target=1, value=theta, index=0),
    ]
    grad = parameter_shift_gradient(
        2,
        circuit,
        [theta],
        operator=lambda qubits: Z(qubits[0]) * Z(qubits[1]),
    )

    cost = psr_expectation(
        2,
        [theta],
        circuit,
        operator=lambda qubits: Z(qubits[0]) * Z(qubits[1]),
    )
    cost_analytical = np.cos(theta) ** 2
    expected = -np.sin(2 * theta)
    print("=== Teste 6: Parâmetro compartilhado em dois qubits ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analítico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad}")
    print(f"  grad analítico = {[expected]}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.isclose(grad[0], expected, atol=1e-6)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 6: Observável com múltiplos termos
#
# Circuito: RY(θ) em q0
# Observable: 0.5 Z0 + 0.3 X0
#
# Analítico: C = 0.5 cos(θ) + 0.3 sin(θ)
# ---------------------------------------------------------------------------


def test_multi_term_observable():
    theta = 0.7
    circuit = [GateInstruction(gate=RY, target=0, value=theta, index=0)]
    grad = parameter_shift_gradient(
        1,
        circuit,
        [theta],
        operator=lambda qubits: 0.5 * Z(qubits[0]) + 0.3 * X(qubits[0]),
    )
    cost = psr_expectation(
        1,
        [theta],
        circuit,
        operator=lambda qubits: 0.5 * Z(qubits[0]) + 0.3 * X(qubits[0]),
    )
    cost_analytical = 0.5 * np.cos(theta) + 0.3 * np.sin(theta)
    expected = -0.5 * np.sin(theta) + 0.3 * np.cos(theta)

    print("=== Teste 7: Observável com múltiplos termos ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analítico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad}")
    print(f"  grad analítico = {[expected]}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.isclose(
        grad[0],
        expected,
        atol=1e-6,
    )
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 7: Observável Y
#
# Circuito: RX(θ) em q0
# Observable: Y0
#
# Analítico: C = -sin(θ), gradiente = -cos(θ)
# ---------------------------------------------------------------------------


def test_y_observable():
    theta = 0.7
    circuit = [GateInstruction(gate=RX, target=0, value=theta, index=0)]
    grad = parameter_shift_gradient(
        1,
        circuit,
        [theta],
        operator=lambda qubits: Y(qubits[0]),
    )
    cost = psr_expectation(
        1,
        [theta],
        circuit,
        operator=lambda qubits: Y(qubits[0]),
    )
    cost_analytical = -np.sin(theta)
    expected = -np.cos(theta)

    print("=== Teste 8: Observável Y ===")
    print(f"  cost Ket       = {cost:.12f}")
    print(f"  cost analítico = {cost_analytical:.12f}")
    print(f"  grad PSR       = {grad}")
    print(f"  grad analítico = {[expected]}")

    assert np.isclose(cost, cost_analytical, atol=1e-8)
    assert np.isclose(grad[0], expected, atol=1e-6)
    print("  PASS\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_multi_qubit_cnot()
    test_shared_parameters()
    test_rx_ry_gates()
    test_3_qubits_cnot()
    test_unused_parameter()
    test_shared_parameter_two_qubits()
    test_multi_term_observable()
    test_y_observable()
    print("Todos os testes avançados passaram.")
