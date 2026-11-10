import functools
import sys

import numpy as np
from ket.base.quant import Quant
from ket.gates import CNOT, RX, RY, RZ, H, X, Y, Z, evolve, obs

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
    ansatz as default_ansatz,
    benchmark_gradient,
    cost,
    group_qubit_commuting,
    pauli_expansion,
)

# ---------------------------------------------------------------------------
# Auxiliares
#
# O simulador do Ket tem erro absoluto da ordem de 1e-7 por valor esperado. O
# Hadamard Test soma varios valores esperados (um por termo do gerador, ou por
# grupo de medicao), entao a tolerancia dos gradientes e 2e-6 e a do custo, 1e-6.
# Quando nao ha solucao analitica, a referencia e um simulador em numpy com
# diferencas centrais de passo 1e-6 em float64 (erro ~1e-10), independente do Ket.
# ---------------------------------------------------------------------------

KET_TOL = 1e-6
METHOD_TOL = 2e-6


def close(a, b, tol=KET_TOL):
    return abs(a - b) <= tol


def all_close(a, b, tol=METHOD_TOL):
    return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))


def known_failure(reason):
    """Marca um teste que descreve um comportamento ainda nao implementado.

    Se o teste falhar, imprime XFAIL. Se passar, avisa que a limitacao foi
    corrigida e que o decorador deve ser removido.
    """

    def wrap(fn):
        @functools.wraps(fn)
        def inner():
            try:
                fn()
            except Exception:
                print(f"  XFAIL ({reason})")
                return
            print("  XPASS: limitacao corrigida; remova @known_failure")

        return inner

    return wrap


def show(title, *lines):
    print(f"=== {title} ===")
    for line in lines:
        print(f"  {line}")


def floats(values):
    return [float(v) for v in values]


# ---------------------------------------------------------------------------
# Circuitos, observaveis e geradores
# ---------------------------------------------------------------------------


def ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = default_ansatz,
) -> None:
    parameter_gate(thetas[0], RY, qubits[0])
    parameter_gate(thetas[1], RY, qubits[1])
    CNOT(qubits[0], qubits[1])
    parameter_gate(thetas[2], RY, qubits[0])
    parameter_gate(thetas[3], RY, qubits[1])


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


def complex_ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = default_ansatz,
) -> None:
    """Variante com RX e RY: os estados tem amplitudes complexas."""
    parameter_gate(thetas[0], RX, qubits[0])
    parameter_gate(thetas[1], RX, qubits[1])
    CNOT(qubits[0], qubits[1])
    parameter_gate(thetas[2], RY, qubits[0])
    parameter_gate(thetas[3], RX, qubits[1])


def complex_segments(thetas: list[float]):
    return [
        lambda qubits: RX(thetas[0], qubits[0]),
        lambda qubits: RX(thetas[1], qubits[1]),
        lambda qubits: (
            CNOT(qubits[0], qubits[1]),
            RY(thetas[2], qubits[0]),
        ),
        lambda qubits: RX(thetas[3], qubits[1]),
    ]


def mixed_generator(qubits: Quant):
    with obs():
        return 0.6 * X(qubits[0]) + 0.8 * Z(qubits[1])


def mixed_generator_ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = default_ansatz,
) -> None:
    parameter_gate(
        thetas[0],
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


def complex_generators():
    return [
        lambda qubits: X(qubits[0]),
        lambda qubits: X(qubits[1]),
        lambda qubits: Y(qubits[0]),
        lambda qubits: X(qubits[1]),
    ]


def single_gate_case(gate, prefix=None):
    """Uma unica porta parametrizada em um qubit, com porta fixa opcional antes."""

    def circuit(thetas, qubits, parameter_gate=default_ansatz):
        if prefix is not None:
            prefix(qubits[0])
        parameter_gate(thetas[0], gate, qubits[0])

    def segments(thetas):
        return [
            lambda qubits: (
                prefix(qubits[0]) if prefix is not None else None,
                gate(thetas[0], qubits[0]),
            )
        ]

    return circuit, segments


def evolution_case(generator, qubit_count=1, prefix=None):
    """Uma porta exp(-i theta G / 2) com G dado por um construtor de Hamiltoniano."""

    def circuit(thetas, qubits, parameter_gate=default_ansatz):
        if prefix is not None:
            prefix(qubits[0])
        parameter_gate(
            thetas[0],
            lambda theta, _target: evolve(generator(qubits) * (theta / 2)),
            qubits[0],
        )

    def segments(thetas):
        return [
            lambda qubits: (
                prefix(qubits[0]) if prefix is not None else None,
                evolve(generator(qubits) * (thetas[0] / 2)),
            )
        ]

    return circuit, segments


def noncommuting_generator(qubits):
    """G = 0,6 X + 0,8 Z no mesmo qubit: os termos nao comutam."""
    with obs():
        return 0.6 * X(qubits[0]) + 0.8 * Z(qubits[0])


NONCOMMUTING_ANGLE = float(np.arctan2(0.6, 0.8))


def noncommuting_gate(theta, qubit):
    """exp(-i theta (0,6 X + 0,8 Z) / 2), exata, com portas nativas.

    RY(a) RZ(theta) RY(-a) e a rotacao em torno do eixo (sin a, 0, cos a) =
    (0,6; 0; 0,8), isto e, tem exatamente o gerador 0,6 X + 0,8 Z. A ordem das
    chamadas e a ordem de aplicacao: RY(-a) primeiro.

    NAO se usa evolve(0,6 X + 0,8 Z) porque, quando os termos nao comutam, o
    evolve do Ket nao exponencia a soma: o custo e o produto das exponenciais
    de cada termo (ver test_evolve_of_non_commuting_sum_is_not_exact).
    """
    RY(-NONCOMMUTING_ANGLE, qubit)
    RZ(theta, qubit)
    RY(NONCOMMUTING_ANGLE, qubit)


def noncommuting_case():
    return single_gate_case(noncommuting_gate)


def xx_generator(qubits):
    with obs():
        return X(qubits[0]) * X(qubits[1])


def zz_generator(qubits):
    with obs():
        return Z(qubits[0]) * Z(qubits[1])


# Formas analiticas do ansatz de dois qubits com RY e observavel Z0 Z1 (e de
# Z0 + Z1 + Z0 Z1), obtidas propagando o observavel de tras para frente.


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
# Simulador de referencia em numpy (independente do Ket)
# ---------------------------------------------------------------------------

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}
_CNOT = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
_PAULI_GATES = {"X": X, "Y": Y, "Z": Z}


def _kron(*matrices):
    result = np.eye(1)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def _rotation(letter, theta):
    return np.cos(theta / 2) * _PAULI["I"] - 1j * np.sin(theta / 2) * _PAULI[letter]


def reference_state_he(t):
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1
    layer_1 = _kron(_rotation("Y", t[0]), _rotation("Y", t[1]))
    layer_2 = _kron(_rotation("Y", t[2]), _rotation("Y", t[3]))
    return layer_2 @ _CNOT @ layer_1 @ psi


def reference_state_complex(t):
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1
    layer_1 = _kron(_rotation("X", t[0]), _rotation("X", t[1]))
    layer_2 = _kron(_rotation("Y", t[2]), _rotation("X", t[3]))
    return layer_2 @ _CNOT @ layer_1 @ psi


def terms_matrix(terms):
    return sum(c * _kron(*[_PAULI[letter] for letter in word]) for c, word in terms)


def reference_value(final_state, matrix, thetas):
    psi = final_state(thetas)
    return float((psi.conj() @ matrix @ psi).real)


def reference_gradient(final_state, matrix, thetas, index, step=1e-6):
    plus, minus = list(thetas), list(thetas)
    plus[index] += step
    minus[index] -= step
    return (
        reference_value(final_state, matrix, plus)
        - reference_value(final_state, matrix, minus)
    ) / (2 * step)


def build_observable(terms):
    """Constroi, a partir de (coeficiente, palavra de Pauli), o observavel do Ket."""

    def builder(qubits):
        total = None
        for coefficient, word in terms:
            operator = None
            for position, letter in enumerate(word):
                if letter == "I":
                    continue
                pauli = _PAULI_GATES[letter](qubits[position])
                operator = pauli if operator is None else operator * pauli
            term = coefficient * operator
            total = term if total is None else total + term
        return total

    return builder


CIRCUITS = {
    "RY (hardware-efficient)": (
        ansatz,
        ansatz_segments,
        generators,
        reference_state_he,
    ),
    "RX/RY (amplitudes complexas)": (
        complex_ansatz,
        complex_segments,
        complex_generators,
        reference_state_complex,
    ),
}

# nome, termos (coeficiente, palavra), numero esperado de grupos QWC
OBSERVABLE_ZOO = [
    ("Z0 Z1", [(1.0, "ZZ")], 1),
    ("Z0 + Z1 + Z0 Z1", [(1.0, "ZI"), (1.0, "IZ"), (1.0, "ZZ")], 1),
    ("X0 X1 + Z0 Z1", [(1.0, "XX"), (1.0, "ZZ")], 2),
    ("0,5 Z0 + 0,3 X1 + 0,2 Y0 Y1", [(0.5, "ZI"), (0.3, "IX"), (0.2, "YY")], 2),
    ("Y0 + Y1 + Y0 Y1", [(1.0, "YI"), (1.0, "IY"), (1.0, "YY")], 1),
    ("X0 + Z0", [(1.0, "XI"), (1.0, "ZI")], 2),
    ("0,7 X0 X1 - 0,4 Y0 Y1 + 0,2 Z0", [(0.7, "XX"), (-0.4, "YY"), (0.2, "ZI")], 3),
]


def run_methods(
    thetas, index, num_qubits, observable_builder, generator_builders, circuit, segments
):
    """Calcula o gradiente do parametro `index` pelos quatro metodos."""
    plain = hadamard_test_gradient(
        thetas, index, num_qubits, observable_builder, generator_builders, circuit
    )
    reverse = reversed_hadamard_test_gradient(
        thetas, index, num_qubits, observable_builder, generator_builders, segments
    )
    grouped, grouped_count = hadamard_test_gradient_grouped(
        thetas, index, num_qubits, observable_builder, generator_builders, circuit
    )
    reverse_grouped, reverse_grouped_count = reversed_hadamard_test_gradient_grouped(
        thetas, index, num_qubits, observable_builder, generator_builders, segments
    )
    values = {
        "HT": plain,
        "RHT": reverse,
        "HT agrupado": grouped,
        "RHT agrupado": reverse_grouped,
    }
    counts = {"HT agrupado": grouped_count, "RHT agrupado": reverse_grouped_count}
    return values, counts


def check_methods(values, expected, label, tol=METHOD_TOL):
    for name, value in values.items():
        assert close(value, expected, tol), (label, name, value, expected)


# ---------------------------------------------------------------------------
# A. Contrato: os quatro metodos (HT, RHT, HT agrupado, RHT agrupado) devem
#    reproduzir o gradiente exato
# ---------------------------------------------------------------------------


# Circuito: RY(t0) -> q0, RY(t1) -> q1, CNOT(q0,q1), RY(t2) -> q0, RY(t3) -> q1
# Observavel: Z0 Z1
# Referencia: expressoes analiticas do custo <Z0 Z1> e do seu gradiente.
def test_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, observable, ansatz)
    cost_exact = analytical_observable_cost(thetas)
    exact = analytical_gradient(thetas)
    ht = [
        hadamard_test_gradient(thetas, i, 2, observable, generators(), ansatz)
        for i in range(len(thetas))
    ]
    rht = [
        reversed_hadamard_test_gradient(
            thetas, i, 2, observable, generators(), ansatz_segments
        )
        for i in range(len(thetas))
    ]

    show(
        "Teste 1: Hadamard test",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"grad Hadamard  = {floats(ht)}",
        f"grad reverso   = {floats(rht)}",
        f"grad analitico = {floats(exact)}",
    )
    assert close(cost_value, cost_exact)
    assert all_close(ht, exact)
    assert all_close(rht, exact)


# Mesmo circuito e observavel, com o RHT como metodo principal e o HT como
# verificacao cruzada.
def test_reversed_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, observable, ansatz)
    cost_exact = analytical_observable_cost(thetas)
    exact = analytical_gradient(thetas)
    rht = [
        reversed_hadamard_test_gradient(
            thetas, i, 2, observable, generators(), ansatz_segments
        )
        for i in range(len(thetas))
    ]
    ht = [
        hadamard_test_gradient(thetas, i, 2, observable, generators(), ansatz)
        for i in range(len(thetas))
    ]

    show(
        "Teste 2: Hadamard reverso",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"grad Hadamard  = {floats(rht)}",
        f"grad direto    = {floats(ht)}",
        f"grad analitico = {floats(exact)}",
    )
    assert close(cost_value, cost_exact)
    assert all_close(rht, exact)
    assert all_close(ht, exact)
    assert all_close(rht, ht)


# Circuito: exp[-i t0 (0,6 X0 + 0,8 Z1) / 2];  Observavel: Z0
# Como |00> e autoestado de Z1 e X0 comuta com Z1, o custo e cos(0,6 t0) e o
# gradiente e -0,6 sin(0,6 t0).
def test_mixed_generator_hadamard_gradients():
    theta = [0.37]
    mixed_observable = lambda qubits: Z(qubits[0])
    cost_exact = np.cos(0.6 * theta[0])
    exact = -0.6 * np.sin(0.6 * theta[0])
    cost_value = cost(theta, 2, mixed_observable, mixed_generator_ansatz)
    ht = hadamard_test_gradient(
        theta, 0, 2, mixed_observable, [mixed_generator], mixed_generator_ansatz
    )
    rht = reversed_hadamard_test_gradient(
        theta, 0, 2, mixed_observable, [mixed_generator], mixed_generator_segments
    )

    show(
        "Teste 3: Gerador misto",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"grad analitico = {exact:.12f}",
        f"grad Hadamard  = {ht:.12f}",
        f"grad reverso   = {rht:.12f}",
    )
    assert close(cost_value, cost_exact)
    assert close(ht, exact, METHOD_TOL)
    assert close(rht, exact, METHOD_TOL)


# Gerador G = 0,6 X0 + 0,8 Z1, com espectro {-1,4; -0,2; 0,2; 1,4}. O Hadamard
# calcula o gradiente exato. A PSR de dois termos assume dois autovalores e
# produz um valor incorreto para este gerador.
def test_hadamard_non_two_eigenvalue_generator():
    theta = [0.37]
    operator = lambda qubits: Z(qubits[0])
    cost_exact = np.cos(0.6 * theta[0])
    exact = -0.6 * np.sin(0.6 * theta[0])
    cost_value = cost(theta, 2, operator, mixed_generator_ansatz)

    ht = hadamard_test_gradient(
        theta, 0, 2, operator, [mixed_generator], mixed_generator_ansatz
    )
    rht = reversed_hadamard_test_gradient(
        theta, 0, 2, operator, [mixed_generator], mixed_generator_segments
    )
    psr_gates = [
        GateInstruction(
            gate=lambda value, qubits: evolve(mixed_generator(qubits) * (value / 2)),
            target=None,
            value=theta[0],
            index=0,
        )
    ]
    psr_cost = psr_expectation(2, theta, psr_gates, operator=operator)
    psr = psr_gradient(2, psr_gates, theta, operator=operator)[0]

    show(
        "Teste 4: Gerador com mais de dois autovalores",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"cost PSR       = {psr_cost:.12f}",
        f"grad analitico = {exact:.12f}",
        f"grad Hadamard  = {ht:.12f}",
        f"grad reverso   = {rht:.12f}",
        f"grad PSR       = {psr:.12f}",
    )
    assert close(cost_value, cost_exact)
    assert close(psr_cost, cost_exact)
    assert close(ht, exact, METHOD_TOL)
    assert close(rht, exact, METHOD_TOL)
    assert abs(psr - exact) > 2e-3


# Observavel: Z0 + Z1 + Z0 Z1 (um unico grupo de medicao). O agrupamento preserva
# o resultado e reduz o HT a 1 circuito por termo do gerador. O RHT agrupado
# executa 3 circuitos (um por termo do observavel, com o gerador em 1 grupo).
def test_grouped_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, grouped_observable, ansatz)
    cost_exact = analytical_grouped_cost(thetas)
    exact = analytical_grouped_gradient(thetas)
    grouped, reverse_grouped = [], []
    count, reverse_count = None, None

    for index in range(len(thetas)):
        value, count = hadamard_test_gradient_grouped(
            thetas, index, 2, grouped_observable, generators(), ansatz
        )
        reverse_value, reverse_count = reversed_hadamard_test_gradient_grouped(
            thetas, index, 2, grouped_observable, generators(), ansatz_segments
        )
        grouped.append(value)
        reverse_grouped.append(reverse_value)
        assert count == 1
        assert reverse_count == 3

    show(
        "Teste 5: Hadamard agrupado",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"grad analitico = {floats(exact)}",
        f"grad agrupado  = {floats(grouped)}",
        f"grad reverso   = {floats(reverse_grouped)}",
        f"circuitos      = {count}/{reverse_count}",
    )
    assert close(cost_value, cost_exact)
    assert all_close(grouped, exact)
    assert all_close(reverse_grouped, exact)


# Mesmo observavel agrupado, com o RHT agrupado como metodo principal e o HT
# agrupado como verificacao cruzada.
def test_grouped_reversed_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]
    cost_value = cost(thetas, 2, grouped_observable, ansatz)
    cost_exact = analytical_grouped_cost(thetas)
    exact = analytical_grouped_gradient(thetas)
    reverse_grouped, direct_grouped = [], []
    count, direct_count = None, None

    for index in range(len(thetas)):
        value, count = reversed_hadamard_test_gradient_grouped(
            thetas, index, 2, grouped_observable, generators(), ansatz_segments
        )
        direct_value, direct_count = hadamard_test_gradient_grouped(
            thetas, index, 2, grouped_observable, generators(), ansatz
        )
        reverse_grouped.append(value)
        direct_grouped.append(direct_value)
        assert count == 3
        assert direct_count == 1

    show(
        "Teste 6: Hadamard reverso agrupado",
        f"cost Ket       = {cost_value:.12f}",
        f"cost analitico = {cost_exact:.12f}",
        f"grad analitico = {floats(exact)}",
        f"grad agrupado  = {floats(reverse_grouped)}",
        f"grad direto    = {floats(direct_grouped)}",
        f"circuitos      = {count}/{direct_count}",
    )
    assert close(cost_value, cost_exact)
    assert all_close(reverse_grouped, exact)
    assert all_close(direct_grouped, exact)


# Observaveis X e Y com agrupamento. A rotacao para a base computacional usa H
# (para X) e S^dagger seguida de H (para Y), e o sinal de cada termo depende dela.
#   RX(a) x RX(b), Y0 + Y1 + Y0 Y1:  C = -sin a - sin b + sin a sin b
#   RY(a) x RY(b), X0 + X1 + X0 X1:  C =  sin a + sin b + sin a sin b
# Cada observavel forma um unico grupo QWC (1 circuito no HT agrupado, 3 no RHT).
def test_x_and_y_observables_with_grouping():
    a, b = 0.5, 0.8
    thetas = [a, b]
    y_terms = [(1.0, "YI"), (1.0, "IY"), (1.0, "YY")]
    x_terms = [(1.0, "XI"), (1.0, "IX"), (1.0, "XX")]
    cases = [
        (
            "RX x RX, Y0 + Y1 + Y0 Y1",
            RX,
            "X",
            y_terms,
            -np.sin(a) - np.sin(b) + np.sin(a) * np.sin(b),
            [-np.cos(a) + np.cos(a) * np.sin(b), -np.cos(b) + np.sin(a) * np.cos(b)],
        ),
        (
            "RY x RY, X0 + X1 + X0 X1",
            RY,
            "Y",
            x_terms,
            np.sin(a) + np.sin(b) + np.sin(a) * np.sin(b),
            [np.cos(a) + np.cos(a) * np.sin(b), np.cos(b) + np.sin(a) * np.cos(b)],
        ),
    ]
    print("=== Teste 7: Observaveis X e Y com agrupamento ===")
    for name, gate, letter, terms, cost_exact, exact in cases:

        def circuit(t, qubits, parameter_gate=default_ansatz, gate=gate):
            parameter_gate(t[0], gate, qubits[0])
            parameter_gate(t[1], gate, qubits[1])

        def segments(t, gate=gate):
            return [
                lambda qubits: gate(t[0], qubits[0]),
                lambda qubits: gate(t[1], qubits[1]),
            ]

        gens = [
            lambda qubits, letter=letter: _PAULI_GATES[letter](qubits[0]),
            lambda qubits, letter=letter: _PAULI_GATES[letter](qubits[1]),
        ]
        builder = build_observable(terms)
        cost_value = cost(thetas, 2, builder, circuit)
        assert close(cost_value, cost_exact), name
        for index in range(2):
            values, counts = run_methods(
                thetas, index, 2, builder, gens, circuit, segments
            )
            print(f"  {name}: parametro {index}: {floats(values.values())}  analitico={exact[index]:.9f}")
            check_methods(values, exact[index], (name, index))
            assert counts == {"HT agrupado": 1, "RHT agrupado": 3}, (name, counts)


# Termos do gerador que NAO comutam: G = 0,6 X + 0,8 Z no mesmo qubit. O Pauli
# controlado e inserido antes da porta, e a soma dos termos so e correta porque
# sum_k beta_k Q_k U = G U = U G. U = exp(-i theta G / 2) e a rotacao de angulo
# theta em torno do eixo (0,6; 0; 0,8) da esfera de Bloch, implementada com
# portas nativas (noncommuting_gate), aplicada a |0>:
#   <Z> = 0,64 + 0,36 cos(t),  <X> = 0,48 (1 - cos(t)),  <Y> = -0,6 sin(t)
# O HT agrupado executa N_j = 2 circuitos (X e Z nao comutam qubit a qubit) e o
# RHT agrupado executa 2 (1 termo do observavel x 2 grupos do gerador).
def test_non_commuting_generator_terms():
    theta = 0.9
    circuit, segments = noncommuting_case()
    cases = [
        ("Z", Z, 0.64 + 0.36 * np.cos(theta), -0.36 * np.sin(theta)),
        ("X", X, 0.48 * (1 - np.cos(theta)), 0.48 * np.sin(theta)),
        ("Y", Y, -0.6 * np.sin(theta), -0.6 * np.cos(theta)),
    ]
    print("=== Teste 8: Gerador com termos que nao comutam ===")
    for name, gate, cost_exact, exact in cases:
        builder = lambda qubits, gate=gate: gate(qubits[0])
        cost_value = cost([theta], 1, builder, circuit)
        values, counts = run_methods(
            [theta], 0, 1, builder, [noncommuting_generator], circuit, segments
        )
        print(f"  observavel {name}: {floats(values.values())}  analitico={exact:.9f}")
        assert close(cost_value, cost_exact), name
        check_methods(values, exact, name)
        assert counts == {"HT agrupado": 2, "RHT agrupado": 2}, (name, counts)


# Geradores com palavras de Pauli de dois qubits (o Pauli controlado atua em dois
# qubits ao mesmo tempo).
#   exp(-i t XX / 2) em |00>:  <Z0> = cos t,  <Z0 Z1> = 1 (Z0 Z1 comuta com XX)
#   exp(-i t ZZ / 2) com H em q0:  <X0> = cos t,  <Y0> = sin t
def test_two_qubit_pauli_generators():
    theta = 0.9
    cases = [
        ("XX / Z0", xx_generator, None, [(1.0, "ZI")], np.cos(theta), -np.sin(theta)),
        ("XX / Z0 Z1", xx_generator, None, [(1.0, "ZZ")], 1.0, 0.0),
        ("ZZ com H / X0", zz_generator, H, [(1.0, "XI")], np.cos(theta), -np.sin(theta)),
        ("ZZ com H / Y0", zz_generator, H, [(1.0, "YI")], np.sin(theta), np.cos(theta)),
    ]
    print("=== Teste 9: Geradores de Pauli de dois qubits ===")
    for name, generator, prefix, terms, cost_exact, exact in cases:
        circuit, segments = evolution_case(generator, prefix=prefix)
        builder = build_observable(terms)
        cost_value = cost([theta], 2, builder, circuit)
        values, _ = run_methods(
            [theta], 0, 2, builder, [generator], circuit, segments
        )
        print(f"  {name}: {floats(values.values())}  analitico={exact:.9f}")
        assert close(cost_value, cost_exact), name
        check_methods(values, exact, name)


# Portas RX e RZ e sinais. RX(t)|0>: <Z> = cos t, <Y> = -sin t. Para RZ e
# necessario H antes: RZ(t) H|0>: <X> = cos t, <Y> = sin t.
def test_rotation_gates_and_signs():
    theta = 0.9
    cases = [
        ("RX / Z", RX, None, "X", Z, np.cos(theta), -np.sin(theta)),
        ("RX / Y", RX, None, "X", Y, -np.sin(theta), -np.cos(theta)),
        ("H.RZ / X", RZ, H, "Z", X, np.cos(theta), -np.sin(theta)),
        ("H.RZ / Y", RZ, H, "Z", Y, np.sin(theta), np.cos(theta)),
    ]
    print("=== Teste 10: Portas RX e RZ e sinais ===")
    for name, gate, prefix, letter, observable_gate, cost_exact, exact in cases:
        circuit, segments = single_gate_case(gate, prefix)
        generator = lambda qubits, letter=letter: _PAULI_GATES[letter](qubits[0])
        builder = lambda qubits, g=observable_gate: g(qubits[0])
        cost_value = cost([theta], 1, builder, circuit)
        values, _ = run_methods(
            [theta], 0, 1, builder, [generator], circuit, segments
        )
        print(f"  {name}: {floats(values.values())}  analitico={exact:.9f}")
        assert close(cost_value, cost_exact), name
        check_methods(values, exact, name)


# Gerador com coeficiente diferente de 1 (e negativo): a porta RY(c t) tem
# gerador c Y, custo cos(c t) e gradiente -c sin(c t). Verifica o peso beta_k do
# HT e o gerador medido por inteiro no RHT.
def test_generator_coefficient():
    theta = 0.9
    print("=== Teste 11: Gerador com coeficiente c != 1 ===")
    for c in (0.5, -1.0, 2.0):
        gate = lambda value, qubit, c=c: RY(c * value, qubit)
        circuit, segments = single_gate_case(gate)
        generator = lambda qubits, c=c: c * Y(qubits[0])
        builder = lambda qubits: Z(qubits[0])
        cost_exact, exact = np.cos(c * theta), -c * np.sin(c * theta)
        cost_value = cost([theta], 1, builder, circuit)
        values, _ = run_methods(
            [theta], 0, 1, builder, [generator], circuit, segments
        )
        print(f"  c={c:+.1f}: {floats(values.values())}  analitico={exact:.9f}")
        assert close(cost_value, cost_exact), c
        check_methods(values, exact, c)


# RZ aplicada a |0> so altera a fase global: <Z> = 1 e o gradiente e 0 nos
# quatro metodos (o observavel comuta com o gerador).
def test_commuting_observable_gives_zero_gradient():
    theta = 0.9
    circuit, segments = single_gate_case(RZ)
    generator = lambda qubits: Z(qubits[0])
    builder = lambda qubits: Z(qubits[0])
    values, _ = run_methods([theta], 0, 1, builder, [generator], circuit, segments)
    print("=== Teste 12: Observavel que comuta com o gerador ===")
    print(f"  {floats(values.values())}")
    assert close(cost([theta], 1, builder, circuit), 1.0)
    check_methods(values, 0.0, "RZ|0>")


# Linearidade no observavel: g[a O1 + b O2] = a g[O1] + b g[O2], para os quatro
# metodos, com O1 = Z0 Z1, O2 = Z0 e a = 2, b = -0,5.
def test_linearity_in_observable():
    thetas = [0.3, 1.1, -0.4, 0.8]
    a, b = 2.0, -0.5
    o1 = build_observable([(1.0, "ZZ")])
    o2 = build_observable([(1.0, "ZI")])
    combined = build_observable([(a, "ZZ"), (b, "ZI")])
    print("=== Teste 13: Linearidade no observavel ===")
    for index in range(len(thetas)):
        g1, _ = run_methods(thetas, index, 2, o1, generators(), ansatz, ansatz_segments)
        g2, _ = run_methods(thetas, index, 2, o2, generators(), ansatz, ansatz_segments)
        gc, _ = run_methods(
            thetas, index, 2, combined, generators(), ansatz, ansatz_segments
        )
        for name in gc:
            assert close(gc[name], a * g1[name] + b * g2[name], 3 * METHOD_TOL), (
                index,
                name,
            )
    print("  OK para os quatro parametros e os quatro metodos")


# Parametro compartilhado: o HT e o RHT tratam cada porta separadamente (o indice
# e a posicao da porta), e o gradiente total e a soma dos gradientes por porta.
# RY(t) em q0 e q1, observavel Z0 Z1: C = cos^2 t, cada porta contribui com
# -sin t cos t e a soma e -sin 2t.
def test_shared_parameter_is_sum_of_gate_gradients():
    theta = 0.7

    def circuit(thetas, qubits, parameter_gate=default_ansatz):
        parameter_gate(thetas[0], RY, qubits[0])
        parameter_gate(thetas[0], RY, qubits[1])

    def segments(thetas):
        return [
            lambda qubits: RY(thetas[0], qubits[0]),
            lambda qubits: RY(thetas[0], qubits[1]),
        ]

    gens = [lambda qubits: Y(qubits[0]), lambda qubits: Y(qubits[1])]
    per_gate = [
        run_methods([theta], index, 2, observable, gens, circuit, segments)[0]
        for index in range(2)
    ]
    print("=== Teste 14: Parametro compartilhado ===")
    for name in per_gate[0]:
        total = per_gate[0][name] + per_gate[1][name]
        print(f"  {name}: por porta = {per_gate[0][name]:.9f}, {per_gate[1][name]:.9f}  soma = {total:.9f}")
        assert close(per_gate[0][name], -np.sin(theta) * np.cos(theta), METHOD_TOL)
        assert close(per_gate[1][name], -np.sin(theta) * np.cos(theta), METHOD_TOL)
        assert close(total, -np.sin(2 * theta), 2 * METHOD_TOL)


# O indice do parametro e a posicao da porta na ordem de aplicacao, e nao o
# indice em thetas. Aqui o ansatz aplica RY(thetas[1]) em q1 ANTES de
# RY(thetas[0]) em q0, e os geradores estao na ordem de aplicacao. Com
# observavel Z0 + Z1, o indice 0 da o gradiente da primeira porta (em thetas[1]).
def test_parameter_index_is_gate_order():
    thetas = [0.4, 1.3]

    def circuit(t, qubits, parameter_gate=default_ansatz):
        parameter_gate(t[1], RY, qubits[1])
        parameter_gate(t[0], RY, qubits[0])

    def segments(t):
        return [
            lambda qubits: RY(t[1], qubits[1]),
            lambda qubits: RY(t[0], qubits[0]),
        ]

    gens = [lambda qubits: Y(qubits[1]), lambda qubits: Y(qubits[0])]
    builder = build_observable([(1.0, "ZI"), (1.0, "IZ")])
    exact = [-np.sin(thetas[1]), -np.sin(thetas[0])]
    print("=== Teste 15: Indice do parametro = ordem das portas ===")
    for index in range(2):
        values, _ = run_methods(thetas, index, 2, builder, gens, circuit, segments)
        print(f"  indice {index}: {floats(values.values())}  analitico={exact[index]:.9f}")
        check_methods(values, exact[index], index)


# Os quatro metodos devem rejeitar indices fora de [0, len(geradores)).
def test_out_of_range_parameter_index_raises():
    thetas = [0.3, 1.1, -0.4, 0.8]
    calls = {
        "HT": lambda i: hadamard_test_gradient(
            thetas, i, 2, observable, generators(), ansatz
        ),
        "RHT": lambda i: reversed_hadamard_test_gradient(
            thetas, i, 2, observable, generators(), ansatz_segments
        ),
        "HT agrupado": lambda i: hadamard_test_gradient_grouped(
            thetas, i, 2, observable, generators(), ansatz
        ),
        "RHT agrupado": lambda i: reversed_hadamard_test_gradient_grouped(
            thetas, i, 2, observable, generators(), ansatz_segments
        ),
    }
    for name, call in calls.items():
        for index in (4, 10, -1):
            try:
                call(index)
            except IndexError:
                continue
            raise AssertionError(f"{name}: indice {index} deveria levantar IndexError")


# Propriedades do agrupamento qubit a qubit (QWC), sem usar o Ket.
def test_group_qubit_commuting_partition():
    def compatible(basis, word):
        return all(b == "I" or w == "I" or b == w for b, w in zip(basis, word))

    assert group_qubit_commuting([]) == []

    single = group_qubit_commuting([(0.5, ("Z", "I"))])
    assert len(single) == 1 and single[0].basis == ["Z", "I"]

    repeated = group_qubit_commuting([(0.5, ("Z", "I")), (0.2, ("Z", "I"))])
    assert len(repeated) == 1 and len(repeated[0].terms) == 2

    # mesma posicao com letras diferentes nao e compativel
    assert len(group_qubit_commuting([(1.0, ("X", "I")), (1.0, ("Z", "I"))])) == 2

    # o termo de maior |coeficiente| define o primeiro grupo
    ordered = group_qubit_commuting(
        [(0.2, ("Y", "Y")), (0.5, ("Z", "I")), (0.3, ("I", "X"))]
    )
    assert len(ordered) == 2
    assert ordered[0].basis == ["Z", "X"] and ordered[1].basis == ["Y", "Y"]

    lists = [
        [(c, tuple(w)) for c, w in terms] for _, terms, _ in OBSERVABLE_ZOO
    ] + [
        [
            (0.9, ("X", "Y", "Z")),
            (0.8, ("X", "I", "I")),
            (0.7, ("I", "Y", "I")),
            (0.6, ("I", "I", "Z")),
            (0.5, ("Z", "Z", "Z")),
            (0.4, ("X", "X", "I")),
        ]
    ]
    for terms in lists:
        groups = group_qubit_commuting(terms)
        flat = [term for group in groups for term in group.terms]
        assert sorted(flat) == sorted(terms)  # cada termo em exatamente um grupo
        for group in groups:
            assert all(compatible(group.basis, word) for _, word in group.terms)
            for position in range(len(group.basis)):
                letters = {w[position] for _, w in group.terms} - {"I"}
                assert len(letters) <= 1
        assert abs(groups[0].terms[0][0]) == max(abs(c) for c, _ in terms)


# Os quatro metodos contra o simulador de referencia em numpy, para dois circuitos
# (RY e RX/RY, este com amplitudes complexas) e sete observaveis com estruturas
# de agrupamento diferentes. Tambem verifica o numero de grupos QWC e a contagem
# de circuitos: HT agrupado = 1 x grupos do observavel (geradores de um termo),
# RHT agrupado = (termos do observavel) x 1 (gerador de um termo, um grupo).
def test_observable_zoo():
    thetas = [0.3, 1.1, -0.4, 0.8]
    print("=== Teste 16: Observaveis variados contra a referencia numpy ===")
    for circuit_name, (circuit, segments, make_generators, state) in CIRCUITS.items():
        print(f"  circuito {circuit_name}")
        for name, terms, expected_groups in OBSERVABLE_ZOO:
            builder = build_observable(terms)
            matrix = terms_matrix(terms)
            groups = group_qubit_commuting(pauli_expansion(builder, 2))
            assert len(groups) == expected_groups, (name, len(groups))
            cost_value = cost(thetas, 2, builder, circuit)
            assert close(cost_value, reference_value(state, matrix, thetas)), name
            worst = 0.0
            for index in range(len(thetas)):
                exact = reference_gradient(state, matrix, thetas, index)
                values, counts = run_methods(
                    thetas, index, 2, builder, make_generators(), circuit, segments
                )
                check_methods(values, exact, (circuit_name, name, index))
                worst = max(worst, *[abs(v - exact) for v in values.values()])
                assert counts["HT agrupado"] == expected_groups, (name, counts)
                assert counts["RHT agrupado"] == len(terms), (name, counts)
            print(
                f"    {name:<34} termos={len(terms)} grupos={expected_groups} "
                f"circuitos HT/RHT={expected_groups}/{len(terms)}  erro max={worst:.1e}"
            )


# ---------------------------------------------------------------------------
# Velocidade dos metodos
#
# Mede o tempo de parede de um gradiente por metodo. Tempos absolutos dependem
# da maquina, entao os testes so comparam metodos entre si: as medidas sao
# intercaladas (HT, RHT, HT agrupado, RHT agrupado, HT, ...) para que a deriva
# do sistema afete todos igualmente, e cada metodo usa o MENOR tempo entre as
# rodadas, que e o estimador menos sensivel a ruido. Uma comparacao "A e mais
# rapido que B" so e exigida quando a diferenca esperada e grande, e ela vem do
# numero de circuitos executados: HT = N_j (termos do gerador), RHT = M (termos
# do observavel), HT agrupado = N_j x grupos do observavel e RHT agrupado =
# M x grupos do gerador. SPEED_SLACK e a folga contra ruido nessas comparacoes.
# ---------------------------------------------------------------------------

METHOD_NAMES = ("HT", "RHT", "HT agrupado", "RHT agrupado")
SPEED_REPETITIONS = 3
SPEED_ROUNDS = 3
SPEED_SLACK = 1.1


def gradient_methods(
    thetas,
    num_qubits,
    observable_builder,
    generator_builders,
    circuit,
    segments,
    indices=None,
):
    """Um callable sem argumentos por metodo, que calcula o gradiente dos `indices`."""
    indices = tuple(range(len(generator_builders)) if indices is None else indices)
    return {
        "HT": lambda: [
            hadamard_test_gradient(
                thetas, i, num_qubits, observable_builder, generator_builders, circuit
            )
            for i in indices
        ],
        "RHT": lambda: [
            reversed_hadamard_test_gradient(
                thetas, i, num_qubits, observable_builder, generator_builders, segments
            )
            for i in indices
        ],
        "HT agrupado": lambda: [
            hadamard_test_gradient_grouped(
                thetas, i, num_qubits, observable_builder, generator_builders, circuit
            )
            for i in indices
        ],
        "RHT agrupado": lambda: [
            reversed_hadamard_test_gradient_grouped(
                thetas, i, num_qubits, observable_builder, generator_builders, segments
            )
            for i in indices
        ],
    }


def time_methods(methods):
    """Menor tempo medio (ms) de cada metodo, com as medidas intercaladas."""
    best = {name: float("inf") for name in methods}
    for _ in range(SPEED_ROUNDS):
        for name, gradient in methods.items():
            best[name] = min(
                best[name], benchmark_gradient(gradient, SPEED_REPETITIONS)
            )
    return best


def assert_faster(times, fast, slow, label):
    assert times[fast] < times[slow] * SPEED_SLACK, (label, fast, slow, times)


# Relatorio de tempo dos quatro metodos para cada observavel do OBSERVABLE_ZOO
# (gradiente do parametro 1 no ansatz hardware-efficient). Nao compara metodos:
# so exige que as medidas sejam validas e mostra qual foi o mais rapido.
def test_method_speed_report():
    thetas = [0.3, 1.1, -0.4, 0.8]
    print("=== Teste 17: Velocidade dos metodos (ms por gradiente de 1 parametro) ===")
    print(f"  {'observavel':<34}" + "".join(f"{n:>14}" for n in METHOD_NAMES) + "  mais rapido")
    for name, terms, _ in OBSERVABLE_ZOO:
        methods = gradient_methods(
            thetas, 2, build_observable(terms), generators(), ansatz,
            ansatz_segments, indices=(1,),
        )
        times = time_methods(methods)
        assert all(0 < t < float("inf") for t in times.values()), (name, times)
        fastest = min(times, key=times.get)
        print(
            f"  {name:<34}"
            + "".join(f"{times[n]:>14.2f}" for n in METHOD_NAMES)
            + f"  {fastest}"
        )


# A ordem de velocidade segue o numero de circuitos de cada metodo:
#   Z0 + Z1 + Z0 Z1 com gerador Y (N_j = 1, M = 3, 1 grupo): HT e HT agrupado
#     executam 1 circuito; RHT e RHT agrupado executam 3.
#   Geradores com 2 termos e observavel Z0 (N_j = 2, M = 1): o HT executa 2
#     circuitos e o RHT executa 1.
def test_speed_follows_circuit_structure():
    z_observable = lambda qubits: Z(qubits[0])
    nc_circuit, nc_segments = noncommuting_case()
    cases = [
        (
            "HE / Z0 + Z1 + Z0 Z1 (N_j=1, M=3)",
            [0.3, 1.1, -0.4, 0.8], 2, grouped_observable, generators(),
            ansatz, ansatz_segments,
            [("HT", "RHT"), ("HT agrupado", "RHT agrupado")],
        ),
        (
            "gerador misto 0,6 X0 + 0,8 Z1 (N_j=2, M=1)",
            [0.37], 2, z_observable, [mixed_generator],
            mixed_generator_ansatz, mixed_generator_segments,
            [("RHT", "HT")],
        ),
        (
            "gerador nao comutante 0,6 X + 0,8 Z (N_j=2, M=1)",
            [0.9], 1, z_observable, [noncommuting_generator],
            nc_circuit, nc_segments,
            [("RHT", "HT")],
        ),
    ]
    print("=== Teste 17b: Velocidade segue o numero de circuitos (ms por gradiente) ===")
    for name, ts, n, builder, gens, circuit, segments, orderings in cases:
        times = time_methods(
            gradient_methods(ts, n, builder, gens, circuit, segments)
        )
        print(f"  {name}: " + ", ".join(f"{m}={times[m]:.2f}" for m in METHOD_NAMES))
        for fast, slow in orderings:
            assert_faster(times, fast, slow, name)


# O tempo cresce com o numero de circuitos. Observavel Z0 Z1 (M = 1, 1 grupo)
# contra 0,7 X0 X1 - 0,4 Y0 Y1 + 0,2 Z0 (M = 3, 3 grupos), com geradores de um
# termo: o RHT e o RHT agrupado passam de 1 para 3 circuitos (M) e o HT agrupado
# passa de 1 para 3 (grupos do observavel). O HT sem agrupamento executa sempre
# 1 circuito (o observavel inteiro vai em um unico exp_value) e fica de fora.
def test_speed_scales_with_terms_and_groups():
    thetas = [0.3, 1.1, -0.4, 0.8]
    small = build_observable([(1.0, "ZZ")])
    large = build_observable([(0.7, "XX"), (-0.4, "YY"), (0.2, "ZI")])
    t_small = time_methods(
        gradient_methods(thetas, 2, small, generators(), ansatz, ansatz_segments)
    )
    t_large = time_methods(
        gradient_methods(thetas, 2, large, generators(), ansatz, ansatz_segments)
    )
    print("=== Teste 17c: Escala do tempo com termos e grupos (ms por gradiente) ===")
    for name in METHOD_NAMES:
        print(
            f"  {name:<14} 1 termo = {t_small[name]:.2f}  3 termos = {t_large[name]:.2f}"
            f"  razao = {t_large[name] / t_small[name]:.2f}"
        )
    for name in ("RHT", "RHT agrupado", "HT agrupado"):
        assert t_large[name] > t_small[name], (name, t_small, t_large)


# As versoes agrupadas do RHT aceitam termos do observavel e grupos do gerador
# pre-calculados (usados nos testes de desempenho) e o resultado deve ser igual.
def test_reversed_grouped_accepts_precomputed_terms():
    thetas = [0.3, 1.1, -0.4, 0.8]
    terms = pauli_expansion(grouped_observable, 2)
    for index in range(len(thetas)):
        groups = group_qubit_commuting(pauli_expansion(generators()[index], 2))
        default = reversed_hadamard_test_gradient_grouped(
            thetas, index, 2, grouped_observable, generators(), ansatz_segments
        )
        precomputed = reversed_hadamard_test_gradient_grouped(
            thetas,
            index,
            2,
            grouped_observable,
            generators(),
            ansatz_segments,
            observable_terms=terms,
            generator_groups=groups,
        )
        assert close(default[0], precomputed[0], 1e-12)
        assert default[1] == precomputed[1]


# Chamadas intercaladas nao podem contaminar umas as outras.
def test_repeated_calls_are_independent():
    thetas = [0.3, 1.1, -0.4, 0.8]
    first, _ = run_methods(thetas, 1, 2, observable, generators(), ansatz, ansatz_segments)
    run_methods([1.3, -0.4, 0.2, 0.5], 2, 2, observable, generators(), ansatz, ansatz_segments)
    second, _ = run_methods(thetas, 1, 2, observable, generators(), ansatz, ansatz_segments)
    for name in first:
        assert close(first[name], second[name], 1e-12), name


# ---------------------------------------------------------------------------
# B. Limitacoes
#
# Estes testes descrevem COMO as funcoes se comportam fora do contrato. Passam
# hoje e documentam a limitacao. Os marcados com @known_failure descrevem um
# comportamento desejado que ainda nao existe.
# ---------------------------------------------------------------------------


# evolve(H) com termos que NAO comutam. O Ket nao exponencia a soma: o custo de
# evolve((0,6 X + 0,8 Z) theta / 2) em |0> e o de um produto de exponenciais, uma
# por termo (cos(0,6 theta) = 0,8577 para theta = 0,9), e nao o da rotacao exata
# em torno de (0,6; 0; 0,8) (0,64 + 0,36 cos theta = 0,8638). O HT e o RHT
# supoem U = exp(-i theta G / 2); com esse evolve, o resultado deixa de ser o
# gradiente de U e os dois metodos discordam entre si. Para termos que comutam
# (QAOA, geradores em qubits diferentes) o evolve e exato.
def test_evolve_of_non_commuting_sum_is_not_exact():
    theta = 0.9

    def circuit(thetas, qubits, parameter_gate=default_ansatz):
        parameter_gate(
            thetas[0],
            lambda t, _q: evolve(noncommuting_generator(qubits) * (t / 2)),
            qubits[0],
        )

    builder = lambda qubits: Z(qubits[0])
    cost_value = cost([theta], 1, builder, circuit)
    exact = 0.64 + 0.36 * np.cos(theta)
    print("=== Teste 18: evolve com termos que nao comutam ===")
    print(f"  cost evolve    = {cost_value:.9f}")
    print(f"  cost exato     = {exact:.9f}")
    print(f"  cos(0,6 theta) = {np.cos(0.6 * theta):.9f}")
    assert abs(cost_value - exact) > 1e-3


def two_gate_ansatz(thetas, qubits, parameter_gate=default_ansatz):
    parameter_gate(thetas[0], RY, qubits[0])
    parameter_gate(thetas[1], RY, qubits[1])


def two_gate_segments(thetas):
    return [
        lambda qubits: RY(thetas[0], qubits[0]),
        lambda qubits: RY(thetas[1], qubits[1]),
    ]


# O HT identifica a porta por um contador de chamadas a parameter_gate. Com
# geradores de sobra (4 geradores para um ansatz de 2 portas), o indice 3 passa
# pela verificacao de intervalo e o Pauli controlado nunca e inserido: o ancilla
# fica nao correlacionado, <Z_ancilla> = 0 e o gradiente devolvido e 0 em silencio.
def test_hadamard_index_beyond_gate_calls_returns_zero():
    thetas = [0.3, 1.1]
    value = hadamard_test_gradient(
        thetas, 3, 2, observable, generators(), two_gate_ansatz
    )
    grouped, _ = hadamard_test_gradient_grouped(
        thetas, 3, 2, observable, generators(), two_gate_ansatz
    )
    print("=== Teste 19: Indice alem do numero de portas (HT) ===")
    print(f"  HT = {value:.3e}  HT agrupado = {grouped:.3e}")
    assert close(value, 0.0, METHOD_TOL)
    assert close(grouped, 0.0, METHOD_TOL)


@known_failure("o HT nao valida o indice contra o numero de portas do ansatz")
def test_hadamard_index_beyond_gate_calls_is_rejected():
    try:
        hadamard_test_gradient(
            [0.3, 1.1], 3, 2, observable, generators(), two_gate_ansatz
        )
    except (IndexError, ValueError):
        return
    raise AssertionError("indice 3 com 2 portas deveria levantar erro")


# No RHT, o fatiamento segments[index + 1:] nao falha para indices alem da lista
# de segmentos: o resultado e um numero sem significado.
@known_failure("o RHT nao valida o indice contra o numero de segmentos")
def test_reversed_index_beyond_segments_is_rejected():
    try:
        reversed_hadamard_test_gradient(
            [0.3, 1.1], 3, 2, observable, generators(), two_gate_segments
        )
    except (IndexError, ValueError):
        return
    raise AssertionError("indice 3 com 2 segmentos deveria levantar erro")


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram.")
    sys.exit(1 if failures else 0)
