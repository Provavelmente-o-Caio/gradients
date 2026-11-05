import numpy as np
from ket.base.quant import Quant
from ket.gates import CNOT, RY, X, Y, Z, evolve, obs

from gradients.hadamard_test.hadamard import (
    hadamard_test_gradient,
    hadamard_test_gradient_grouped,
)
from gradients.hadamard_test.reversed_hadamard import (
    reversed_hadamard_test_gradient,
    reversed_hadamard_test_gradient_grouped,
)
from gradients.psr.advanced import (
    GateInstruction,
    expectation as psr_expectation,
    parameter_shift_gradient as psr_gradient,
)
from gradients.utils.utils import (
    ParameterGate,
    cost,
    direct_parameter_gate,
)


def ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    parameter_gate(thetas[0], 0, RY, qubits[0])
    parameter_gate(thetas[1], 1, RY, qubits[1])
    CNOT(qubits[0], qubits[1])
    parameter_gate(thetas[2], 2, RY, qubits[0])
    parameter_gate(thetas[3], 3, RY, qubits[1])


def ansatz_segments(thetas: list[float]):
    return [
        lambda qubits: RY(thetas[0], qubits[0]),
        lambda qubits: RY(thetas[1], qubits[1]),
        lambda qubits: (
            CNOT(qubits[0], qubits[1]),
            RY(thetas[2], qubits[0]),
        ),
        lambda qubits: RY(thetas[3], qubits[1]),
    ]


def mixed_generator(qubits: Quant):
    with obs():
        return 0.6 * X(qubits[0]) + 0.8 * Z(qubits[1])


def mixed_generator_ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    parameter_gate(
        thetas[0],
        0,
        lambda theta, _target: evolve(mixed_generator(qubits) * (theta / 2)),
        qubits[0],
    )


def mixed_generator_segments(thetas: list[float]):
    return [lambda qubits: evolve(mixed_generator(qubits) * (thetas[0] / 2))]


def observable(qubits):
    return Z(qubits[0]) * Z(qubits[1])


def grouped_observable(qubits):
    return Z(qubits[0]) + Z(qubits[1]) + Z(qubits[0]) * Z(qubits[1])


def generators():
    return [
        lambda qubits: Y(qubits[0]),
        lambda qubits: Y(qubits[1]),
        lambda qubits: Y(qubits[0]),
        lambda qubits: Y(qubits[1]),
    ]


def analytical_observable_cost(thetas):
    a, b, c, d = thetas
    return (
        np.sin(a) * np.sin(c) * np.sin(d)
        - np.sin(b) * np.sin(d) * np.cos(a) * np.cos(c)
        + np.cos(b) * np.cos(c) * np.cos(d)
    )


def analytical_gradient(thetas):
    a, b, c, d = thetas
    return [
        (np.sin(a) * np.sin(b) * np.cos(c) + np.sin(c) * np.cos(a))
        * np.sin(d),
        -(np.sin(b) * np.cos(d) + np.sin(d) * np.cos(a) * np.cos(b))
        * np.cos(c),
        np.sin(a) * np.sin(d) * np.cos(c)
        + np.sin(b) * np.sin(c) * np.sin(d) * np.cos(a)
        - np.sin(c) * np.cos(b) * np.cos(d),
        np.sin(a) * np.sin(c) * np.cos(d)
        - np.sin(b) * np.cos(a) * np.cos(c) * np.cos(d)
        - np.sin(d) * np.cos(b) * np.cos(c),
    ]


def analytical_grouped_cost(thetas):
    a, b, c, d = thetas
    z0 = -np.sin(a) * np.sin(b) * np.sin(c) + np.cos(a) * np.cos(c)
    z1 = -np.sin(b) * np.sin(d) + np.cos(a) * np.cos(b) * np.cos(d)
    return z0 + z1 + analytical_observable_cost(thetas)


def analytical_grouped_gradient(thetas):
    a, b, c, d = thetas
    observable_gradient = analytical_gradient(thetas)
    return [
        observable_gradient[0] - np.cos(a) * np.sin(b) * np.sin(c)
        - np.sin(a) * np.cos(c) - np.sin(a) * np.cos(b) * np.cos(d),
        observable_gradient[1] - np.sin(a) * np.cos(b) * np.sin(c)
        - np.cos(b) * np.sin(d) - np.cos(a) * np.sin(b) * np.cos(d),
        observable_gradient[2] - np.sin(a) * np.sin(b) * np.cos(c)
        - np.cos(a) * np.sin(c),
        observable_gradient[3] - np.sin(b) * np.cos(d)
        - np.cos(a) * np.cos(b) * np.sin(d),
    ]


# ---------------------------------------------------------------------------
# Teste 1: Hadamard direto
#
# Circuito: RY(θ₀) → q0, RY(θ₁) → q1, CNOT(q0,q1),
#           RY(θ₂) → q0, RY(θ₃) → q1
# Observável: Z₀ Z₁
#
# Referência: expressões analíticas do custo ⟨Z₀Z₁⟩ e do seu gradiente.
# ---------------------------------------------------------------------------


def test_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, observable, ansatz)
    cost_expected = analytical_observable_cost(thetas)
    expected = analytical_gradient(thetas)
    gradients = [
        hadamard_test_gradient(
            thetas,
            index,
            2,
            observable,
            generators(),
            ansatz,
        )
        for index in range(len(thetas))
    ]
    reverse_gradients = [
        reversed_hadamard_test_gradient(
            thetas,
            index,
            2,
            observable,
            generators(),
            ansatz_segments,
        )
        for index in range(len(thetas))
    ]

    print("=== Teste 1: Hadamard test ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {cost_expected:.12f}")
    print(f"  grad Hadamard  = {gradients}")
    print(f"  grad reverso   = {reverse_gradients}")
    print(f"  grad analítico = {expected}")

    assert np.allclose(gradients, expected, atol=2e-3)
    assert np.allclose(reverse_gradients, expected, atol=2e-3)
    assert np.isclose(cost_value, cost_expected, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 2: Hadamard reverso
#
# Usa o mesmo circuito e observável do teste anterior, aplicando os segmentos
# do circuito em ordem reversa para calcular o gradiente.
#
# Referência: expressões analíticas do custo ⟨Z₀Z₁⟩ e do seu gradiente.
# ---------------------------------------------------------------------------


def test_reversed_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, observable, ansatz)
    cost_expected = analytical_observable_cost(thetas)
    expected = analytical_gradient(thetas)
    gradients = [
        reversed_hadamard_test_gradient(
            thetas,
            index,
            2,
            observable,
            generators(),
            ansatz_segments,
        )
        for index in range(len(thetas))
    ]
    direct_gradients = [
        hadamard_test_gradient(
            thetas,
            index,
            2,
            observable,
            generators(),
            ansatz,
        )
        for index in range(len(thetas))
    ]

    print("=== Teste 2: Hadamard reverso ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {cost_expected:.12f}")
    print(f"  grad Hadamard  = {gradients}")
    print(f"  grad direto    = {direct_gradients}")
    print(f"  grad analítico = {expected}")

    assert np.allclose(gradients, expected, atol=2e-3)
    assert np.allclose(direct_gradients, expected, atol=2e-3)
    assert np.isclose(cost_value, cost_expected, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 3: Gerador misto
#
# Circuito: exp[-i θ₀ (0.6 X₀ + 0.8 Z₁) / 2]
# Observável: Z₀
#
# O gradiente direto e o reverso são comparados com a expressão analítica.
# ---------------------------------------------------------------------------


def test_mixed_generator_hadamard_gradients():
    theta = [0.37]

    def mixed_observable(qubits):
        return Z(qubits[0])

    expected_cost = np.cos(0.6 * theta[0])
    expected = -0.6 * np.sin(0.6 * theta[0])
    cost_value = cost(theta, 2, mixed_observable, mixed_generator_ansatz)
    hadamard = hadamard_test_gradient(
        theta,
        0,
        2,
        mixed_observable,
        [mixed_generator],
        mixed_generator_ansatz,
    )
    reversed_hadamard = reversed_hadamard_test_gradient(
        theta,
        0,
        2,
        mixed_observable,
        [mixed_generator],
        mixed_generator_segments,
    )

    print("=== Teste 3: Gerador misto ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {expected_cost:.12f}")
    print(f"  grad analítico = {expected:.12f}")
    print(f"  grad Hadamard  = {hadamard:.12f}")
    print(f"  grad reverso   = {reversed_hadamard:.12f}")

    assert np.isclose(hadamard, expected, atol=2e-3)
    assert np.isclose(reversed_hadamard, expected, atol=2e-3)
    assert np.isclose(cost_value, expected_cost, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 4: Gerador com mais de dois autovalores
#
# Gerador: G = 0.6 X₀ + 0.8 Z₁, com espectro
#          {-1.4, -0.2, 0.2, 1.4}.
# Observável: Z₀.
#
# Como |00⟩ é autestado de Z₁, o custo conhecido é cos(0.6θ₀) e o gradiente
# exato é -0.6 sin(0.6θ₀). O Hadamard calcula esse gradiente diretamente.
# A PSR padrão abaixo assume dois autovalores e, portanto, produz um valor
# incorreto para este gerador. Uma regra espectral generalizada poderia tratar
# o caso, mas não é usada neste teste.
# ---------------------------------------------------------------------------


def test_hadamard_non_two_eigenvalue_generator():
    theta = [0.37]
    cost_expected = np.cos(0.6 * theta[0])
    gradient_expected = -0.6 * np.sin(0.6 * theta[0])
    cost_value = cost(
        theta,
        2,
        lambda qubits: Z(qubits[0]),
        mixed_generator_ansatz,
    )

    hadamard = hadamard_test_gradient(
        theta,
        0,
        2,
        lambda qubits: Z(qubits[0]),
        [mixed_generator],
        mixed_generator_ansatz,
    )
    reverse_hadamard = reversed_hadamard_test_gradient(
        theta,
        0,
        2,
        lambda qubits: Z(qubits[0]),
        [mixed_generator],
        mixed_generator_segments,
    )

    psr_gates = [
        GateInstruction(
            gate=lambda value, qubits: evolve(
                mixed_generator(qubits) * (value / 2)
            ),
            target=None,
            value=theta[0],
            index=0,
        )
    ]
    psr_cost = psr_expectation(
        2,
        theta,
        psr_gates,
        operator=lambda qubits: Z(qubits[0]),
    )
    psr = psr_gradient(
        2,
        psr_gates,
        theta,
        operator=lambda qubits: Z(qubits[0]),
    )[0]

    print("=== Teste 4: Gerador com mais de dois autovalores ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {cost_expected:.12f}")
    print(f"  cost PSR       = {psr_cost:.12f}")
    print(f"  grad analítico = {gradient_expected:.12f}")
    print(f"  grad Hadamard  = {hadamard:.12f}")
    print(f"  grad reverso   = {reverse_hadamard:.12f}")
    print(f"  grad PSR       = {psr:.12f}")

    assert np.isclose(cost_value, cost_expected, atol=1e-8)
    assert np.isclose(psr_cost, cost_expected, atol=1e-8)
    assert np.isclose(hadamard, gradient_expected, atol=2e-3)
    assert np.isclose(reverse_hadamard, gradient_expected, atol=2e-3)
    assert not np.isclose(psr, gradient_expected, atol=2e-3)
    print("  PASS: Hadamard funciona; PSR de dois autovalores falha\n")


# ---------------------------------------------------------------------------
# Teste 5: Hadamard agrupado
#
# Observável: Z₀ + Z₁ + Z₀ Z₁
#
# Verifica que a decomposição em grupos de medição preserva o resultado do
# Hadamard direto e reduz os termos a um único circuito de medição.
# ---------------------------------------------------------------------------


def test_grouped_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, grouped_observable, ansatz)
    cost_expected = analytical_grouped_cost(thetas)
    expected = analytical_grouped_gradient(thetas)
    reverse_grouped_gradients = []
    grouped_gradients = []

    for parameter_index in range(len(thetas)):
        grouped, circuit_count = hadamard_test_gradient_grouped(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz,
        )
        reverse_grouped, reverse_circuit_count = (
            reversed_hadamard_test_gradient_grouped(
                thetas,
                parameter_index,
                2,
                grouped_observable,
                generators(),
                ansatz_segments,
            )
        )

        grouped_gradients.append(grouped)
        reverse_grouped_gradients.append(reverse_grouped)
        assert circuit_count == 1
        assert reverse_circuit_count == 3

    print("=== Teste 5: Hadamard agrupado ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {cost_expected:.12f}")
    print(f"  grad analítico = {expected}")
    print(f"  grad agrupado  = {grouped_gradients}")
    print(f"  grad reverso   = {reverse_grouped_gradients}")
    print(f"  circuitos      = {circuit_count}/{reverse_circuit_count}")

    assert np.allclose(grouped_gradients, expected, atol=2e-3)
    assert np.allclose(reverse_grouped_gradients, expected, atol=2e-3)
    assert np.isclose(cost_value, cost_expected, atol=1e-8)
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Teste 6: Hadamard reverso agrupado
#
# Usa o mesmo observável agrupado do teste anterior e verifica diretamente
# a implementação reversa agrupada.
# ---------------------------------------------------------------------------


def test_grouped_reversed_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, grouped_observable, ansatz)
    cost_expected = analytical_grouped_cost(thetas)
    expected = analytical_grouped_gradient(thetas)
    grouped_gradients = []
    direct_grouped_gradients = []

    for parameter_index in range(len(thetas)):
        grouped, circuit_count = reversed_hadamard_test_gradient_grouped(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz_segments,
        )
        direct_grouped, direct_circuit_count = hadamard_test_gradient_grouped(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz,
        )

        grouped_gradients.append(grouped)
        direct_grouped_gradients.append(direct_grouped)
        assert circuit_count == 3
        assert direct_circuit_count == 1

    print("=== Teste 6: Hadamard reverso agrupado ===")
    print(f"  cost Ket       = {cost_value:.12f}")
    print(f"  cost analítico = {cost_expected:.12f}")
    print(f"  grad analítico = {expected}")
    print(f"  grad agrupado  = {grouped_gradients}")
    print(f"  grad direto    = {direct_grouped_gradients}")
    print(f"  circuitos      = {circuit_count}/{direct_circuit_count}")

    assert np.allclose(grouped_gradients, expected, atol=2e-3)
    assert np.allclose(direct_grouped_gradients, expected, atol=2e-3)
    assert np.isclose(cost_value, cost_expected, atol=1e-8)
    print("  PASS\n")


if __name__ == "__main__":
    test_hadamard_test_gradient()
    test_reversed_hadamard_test_gradient()
    test_mixed_generator_hadamard_gradients()
    test_hadamard_non_two_eigenvalue_generator()
    test_grouped_hadamard_gradients()
    test_grouped_reversed_hadamard_gradients()
