from collections.abc import Sequence

import numpy as np
from ket import Process
from ket.base.quant import Quant
from ket.gates import H, X, Z, evolve, obs
from ket.operations import exp_value

from gradients.hadamard_test.hadamard import (
    hadamard_test_gradient,
    hadamard_test_gradient_grouped,
)
from gradients.hadamard_test.reversed_hadamard import (
    reversed_hadamard_test_gradient,
    reversed_hadamard_test_gradient_grouped,
)
from gradients.psr.advanced import parameter_shift_gradient
from gradients.utils.utils import (
    GateInstruction,
    GeneratorBuilder,
    ParameterGate,
    Segment,
    benchmark_gradient,
    direct_parameter_gate,
    spectral_shift_rule,
)

EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
NUM_NODES = 5
DEPTH = 2
PARAMETERS = [0.5] * (2 * DEPTH)
REPETITIONS = 100


def cost_h(edges: Sequence[tuple[int, int]], qubits: Quant):
    with obs():
        return -0.5 * sum(
            1 - Z(qubits[left]) * Z(qubits[right])
            for left, right in edges
        )


def mixer_h(qubits: Quant):
    with obs():
        return sum(X(qubit) for qubit in qubits)


def qaoa(
    parameters: Sequence[float],
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
) -> Quant:
    process = Process()
    qubits = process.alloc(num_nodes)
    H(qubits)
    for layer in range(len(parameters) // 2):
        evolve(cost_h(edges, qubits) * parameters[2 * layer])
        evolve(mixer_h(qubits) * parameters[2 * layer + 1])
    return qubits


def objective(
    parameters: Sequence[float],
    edges: Sequence[tuple[int, int]] = EDGES,
    num_nodes: int = NUM_NODES,
) -> float:
    return _expectation(parameters, edges, num_nodes)


def _expectation(
    parameters: Sequence[float],
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
) -> float:
    qubits = qaoa(parameters, edges, num_nodes)
    with obs():
        observable = cost_h(edges, qubits)
    return exp_value(observable).get()


def qaoa_parameter_gate(
    edges: Sequence[tuple[int, int]],
    parameter_index: int,
    parameter_gate: ParameterGate,
    theta: float,
    qubits: Quant,
) -> None:
    hamiltonian = (
        cost_h(edges, qubits)
        if parameter_index % 2 == 0
        else mixer_h(qubits)
    )
    parameter_gate(
        theta,
        parameter_index,
        lambda value, _target: evolve(hamiltonian * value),
        qubits,
    )


def qaoa_ansatz(
    parameters: Sequence[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    H(qubits)
    for parameter_index, theta in enumerate(parameters):
        qaoa_parameter_gate(
            EDGES,
            parameter_index,
            parameter_gate,
            theta,
            qubits,
        )


def qaoa_segments(parameters: Sequence[float]) -> list[Segment]:
    segments: list[Segment] = []
    for parameter_index, theta in enumerate(parameters):
        segments.append(
            lambda qubits, index=parameter_index, value=theta: (
                H(qubits) if index == 0 else None,
                qaoa_parameter_gate(
                    EDGES,
                    index,
                    direct_parameter_gate,
                    value,
                    qubits,
                ),
            )
        )
    return segments


def qaoa_generators(
) -> list[GeneratorBuilder]:
    return [
        (lambda q, edges=EDGES: 2 * cost_h(edges, q))
        if index % 2 == 0
        else (lambda q: 2 * mixer_h(q))
        for index in range(2 * DEPTH)
    ]


def qaoa_hadamard_gradient(parameters: Sequence[float]) -> list[float]:
    return [
        hadamard_test_gradient(
            parameters,
            index,
            NUM_NODES,
            lambda qubits: cost_h(EDGES, qubits),
            qaoa_generators(),
            qaoa_ansatz,
        )
        for index in range(len(parameters))
    ]


def qaoa_reversed_gradient(parameters: Sequence[float]) -> list[float]:
    return [
        reversed_hadamard_test_gradient(
            parameters,
            index,
            NUM_NODES,
            lambda qubits: cost_h(EDGES, qubits),
            qaoa_generators(),
            qaoa_segments,
        )
        for index in range(len(parameters))
    ]


def qaoa_grouped_hadamard_gradient(parameters: Sequence[float]) -> list[float]:
    return [
        hadamard_test_gradient_grouped(
            parameters,
            index,
            NUM_NODES,
            lambda qubits: cost_h(EDGES, qubits),
            qaoa_generators(),
            qaoa_ansatz,
        )[0]
        for index in range(len(parameters))
    ]


def qaoa_grouped_reversed_gradient(
    parameters: Sequence[float],
) -> list[float]:
    return [
        reversed_hadamard_test_gradient_grouped(
            parameters,
            index,
            NUM_NODES,
            lambda qubits: cost_h(EDGES, qubits),
            qaoa_generators(),
            qaoa_segments,
        )[0]
        for index in range(len(parameters))
    ]


def qaoa_psr_gradient(parameters: Sequence[float]) -> list[float]:
    cost_eigenvalues = [
        -2 * sum(
            ((bits >> left) & 1) != ((bits >> right) & 1)
            for left, right in EDGES
        )
        for bits in range(1 << NUM_NODES)
    ]
    mixer_eigenvalues = [
        2 * sum(
            1 if (bits >> index) & 1 else -1
            for index in range(NUM_NODES)
        )
        for bits in range(1 << NUM_NODES)
    ]
    gates = [
        GateInstruction(
            gate=lambda theta, qubits, index=index: (
                H(qubits) if index == 0 else None,
                evolve(
                    (
                        cost_h(EDGES, qubits)
                        if index % 2 == 0
                        else mixer_h(qubits)
                    )
                    * theta
                ),
            ),
            target=None,
            value=theta,
            index=index,
            shift_rule=spectral_shift_rule(
                cost_eigenvalues if index % 2 == 0 else mixer_eigenvalues
            ),
        )
        for index, theta in enumerate(parameters)
    ]
    return parameter_shift_gradient(NUM_NODES, gates, parameters, operator=cost_h_for_psr)


def cost_h_for_psr(qubits: Quant):
    return cost_h(EDGES, qubits)


def qaoa_native_gradient(parameters: Sequence[float]) -> list[float]:
    process = Process(gradient=True)
    qubits = process.alloc(NUM_NODES)
    registered = [process.param(theta) for theta in parameters]
    H(qubits)
    for layer in range(DEPTH):
        evolve(cost_h(EDGES, qubits) * registered[2 * layer])
        evolve(mixer_h(qubits) * registered[2 * layer + 1])
    with obs():
        observable = cost_h(EDGES, qubits)
    exp_value(observable).get()
    return [parameter.grad for parameter in registered]


def test_qaoa_hadamard_gradient_methods_agree() -> None:
    hadamard = qaoa_hadamard_gradient(PARAMETERS)
    reversed_hadamard = qaoa_reversed_gradient(PARAMETERS)
    grouped_hadamard = qaoa_grouped_hadamard_gradient(PARAMETERS)
    grouped_reversed = qaoa_grouped_reversed_gradient(PARAMETERS)

    assert np.allclose(hadamard, reversed_hadamard, atol=2e-5)
    assert np.allclose(hadamard, grouped_hadamard, atol=2e-5)
    assert np.allclose(hadamard, grouped_reversed, atol=2e-5)
    assert np.allclose(qaoa_psr_gradient(PARAMETERS), hadamard, atol=2e-5)
    assert np.allclose(qaoa_native_gradient(PARAMETERS), hadamard, atol=2e-5)


def test_qaoa_gradient_performance() -> None:
    methods = {
        "PSR": lambda: qaoa_psr_gradient(PARAMETERS),
        "KET native": lambda: qaoa_native_gradient(PARAMETERS),
        "Hadamard": lambda: qaoa_hadamard_gradient(PARAMETERS),
        "Reversed Hadamard": lambda: qaoa_reversed_gradient(PARAMETERS),
        "Grouped Hadamard": lambda: qaoa_grouped_hadamard_gradient(
            PARAMETERS
        ),
        "Grouped Reversed Hadamard": lambda: qaoa_grouped_reversed_gradient(
            PARAMETERS
        ),
    }
    results = {name: method() for name, method in methods.items()}
    assert np.allclose(
        results["Hadamard"],
        results["Reversed Hadamard"],
        atol=2e-5,
    )
    assert np.allclose(
        results["Hadamard"],
        results["Grouped Hadamard"],
        atol=2e-5,
    )
    assert np.allclose(
        results["Hadamard"],
        results["Grouped Reversed Hadamard"],
        atol=2e-5,
    )

    print("\nQAOA Max-Cut gradient performance (ms/evaluation):")
    for name, method in methods.items():
        print(f"{name:>18}: {benchmark_gradient(method, REPETITIONS):8.3f} ms")


if __name__ == "__main__":
    test_qaoa_hadamard_gradient_methods_agree()
    test_qaoa_gradient_performance()
