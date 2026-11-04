import numpy as np
from ket.base.quant import Quant
from ket.gates import RY, Y, Z

from gradients.hadamard_test.hadamard import (
    hadamard_test_gradient,
    hadamard_test_gradient_grouped,
)
from gradients.hadamard_test.reversed_hadamard import (
    reversed_hadamard_test_gradient,
    reversed_hadamard_test_gradient_grouped,
)
from gradients.psr.advanced import parameter_shift_gradient
from gradients.tests.test_hadamard import (
    generators,
    grouped_observable,
)
from gradients.utils.utils import (
    GateInstruction,
    ParameterGate,
    benchmark_gradient,
    direct_parameter_gate,
    group_qubit_commuting,
    pauli_expansion,
)


PARAMETERS = [0.3, 1.1, -0.4, 0.8]
NUM_QUBITS = 2
REPETITIONS = 1000

BENCHMARK_OBSERVABLE_GROUPS = group_qubit_commuting(
    pauli_expansion(grouped_observable, NUM_QUBITS)
)
BENCHMARK_OBSERVABLE_TERMS = pauli_expansion(grouped_observable, NUM_QUBITS)
BENCHMARK_GENERATOR_TERMS = tuple(
    pauli_expansion(generator, NUM_QUBITS) for generator in generators()
)
BENCHMARK_GENERATOR_GROUPS = tuple(
    group_qubit_commuting(terms) for terms in BENCHMARK_GENERATOR_TERMS
)


def many_z_observable(qubits: Quant):
    observable = Z(qubits[0])
    for index in range(1, len(qubits)):
        observable = observable + Z(qubits[index])
    for left in range(len(qubits)):
        for right in range(left + 1, len(qubits)):
            observable = observable + Z(qubits[left]) * Z(qubits[right])
    return observable


def many_qubit_ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    for parameter_index, theta in enumerate(thetas):
        parameter_gate(theta, parameter_index, RY, qubits[parameter_index])


def many_qubit_segments(thetas: list[float]):
    return [
        lambda qubits, theta=theta, index=index: RY(theta, qubits[index])
        for index, theta in enumerate(thetas)
    ]


def many_qubit_generators():
    return [lambda qubits, index=index: Y(qubits[index]) for index in range(4)]


def ungrouped_many_term_gradient() -> list[float]:
    return [
        hadamard_test_gradient(
            [0.2, 0.4, 0.6, 0.8],
            parameter_index,
            4,
            many_z_observable,
            many_qubit_generators(),
            many_qubit_ansatz,
        )
        for parameter_index in range(4)
    ]


def grouped_many_term_gradient() -> list[float]:
    return [
        hadamard_test_gradient_grouped(
            [0.2, 0.4, 0.6, 0.8],
            parameter_index,
            4,
            many_z_observable,
            many_qubit_generators(),
            many_qubit_ansatz,
        )[0]
        for parameter_index in range(4)
    ]


def test_grouping_reduces_measurements() -> None:
    parameters = [0.2, 0.4, 0.6, 0.8]
    generator_terms = many_qubit_generators()
    _, grouped_circuits = hadamard_test_gradient_grouped(
        parameters,
        0,
        4,
        many_z_observable,
        generator_terms,
        many_qubit_ansatz,
    )

    observable_terms = 4 + 6
    assert grouped_circuits == 1
    assert grouped_circuits < observable_terms
    assert np.allclose(
        grouped_many_term_gradient(),
        ungrouped_many_term_gradient(),
        atol=2e-3,
    )


def test_grouping_performance_opportunity() -> None:
    ungrouped_time = benchmark_gradient(
        ungrouped_many_term_gradient,
        REPETITIONS,
    )
    grouped_time = benchmark_gradient(
        grouped_many_term_gradient,
        REPETITIONS,
    )
    speedup = ungrouped_time / grouped_time

    print("\nGrouping opportunity (10 observable terms, 1 compatible group):")
    print(f"Ungrouped Hadamard: {ungrouped_time:8.3f} ms")
    print(f"Grouped Hadamard:   {grouped_time:8.3f} ms")
    print(f"Measured speedup:   {speedup:8.2f}x")


def benchmark_ansatz(
    thetas: list[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    targets = [0, 1, 0, 1]
    for parameter_index, theta in enumerate(thetas):
        parameter_gate(theta, parameter_index, RY, qubits[targets[parameter_index]])


def benchmark_segments(thetas: list[float]):
    targets = [0, 1, 0, 1]
    return [
        lambda qubits, theta=theta, target=targets[index]: RY(theta, qubits[target])
        for index, theta in enumerate(thetas)
    ]


def psr_gradient() -> list[float]:
    circuit = [
        GateInstruction(gate=RY, target=0, value=PARAMETERS[0], index=0),
        GateInstruction(gate=RY, target=1, value=PARAMETERS[1], index=1),
        GateInstruction(gate=RY, target=0, value=PARAMETERS[2], index=2),
        GateInstruction(gate=RY, target=1, value=PARAMETERS[3], index=3),
    ]
    return parameter_shift_gradient(
        NUM_QUBITS,
        circuit,
        PARAMETERS,
        operator=grouped_observable,
    )


def hadamard_gradient() -> list[float]:
    return [
        hadamard_test_gradient(
            PARAMETERS,
            parameter_index,
            NUM_QUBITS,
            grouped_observable,
            generators(),
            benchmark_ansatz,
        )
        for parameter_index in range(len(PARAMETERS))
    ]


def reversed_hadamard_gradient() -> list[float]:
    return [
        reversed_hadamard_test_gradient(
            PARAMETERS,
            parameter_index,
            NUM_QUBITS,
            grouped_observable,
            generators(),
            benchmark_segments,
        )
        for parameter_index in range(len(PARAMETERS))
    ]


def grouped_hadamard_gradient() -> list[float]:
    return [
        hadamard_test_gradient_grouped(
            PARAMETERS,
            parameter_index,
            NUM_QUBITS,
            grouped_observable,
            generators(),
            benchmark_ansatz,
            generator_terms=BENCHMARK_GENERATOR_TERMS[parameter_index],
            observable_groups=BENCHMARK_OBSERVABLE_GROUPS,
        )[0]
        for parameter_index in range(len(PARAMETERS))
    ]


def grouped_reversed_hadamard_gradient() -> list[float]:
    return [
        reversed_hadamard_test_gradient_grouped(
            PARAMETERS,
            parameter_index,
            NUM_QUBITS,
            grouped_observable,
            generators(),
            benchmark_segments,
            observable_terms=BENCHMARK_OBSERVABLE_TERMS,
            generator_groups=BENCHMARK_GENERATOR_GROUPS[parameter_index],
        )[0]
        for parameter_index in range(len(PARAMETERS))
    ]


def test_gradient_performance() -> None:
    gradients = {
        "PSR": psr_gradient,
        "Hadamard": hadamard_gradient,
        "Reversed Hadamard": reversed_hadamard_gradient,
        "Grouped Hadamard": grouped_hadamard_gradient,
        "Grouped Reversed Hadamard": grouped_reversed_hadamard_gradient,
    }
    results = {name: gradient() for name, gradient in gradients.items()}
    reference = np.asarray(results["PSR"])

    for result in results.values():
        assert np.allclose(result, reference, atol=2e-3)

    timings = {
        name: benchmark_gradient(gradient, REPETITIONS)
        for name, gradient in gradients.items()
    }
    print("\nGradient performance (ms/evaluation):")
    for name, timing in timings.items():
        print(f"{name:>18}: {timing:8.3f} ms")


if __name__ == "__main__":
    test_gradient_performance()
