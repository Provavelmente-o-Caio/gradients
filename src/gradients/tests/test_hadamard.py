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


def finite_difference_gradient(thetas, parameter_index, step=1e-4):
    forward = thetas.copy()
    backward = thetas.copy()
    forward[parameter_index] += step
    backward[parameter_index] -= step
    return (
        cost(forward, 2, observable, ansatz)
        - cost(backward, 2, observable, ansatz)
    ) / (2 * step)


def test_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]

    for parameter_index in range(len(thetas)):
        gradient = hadamard_test_gradient(
            thetas,
            parameter_index,
            2,
            observable,
            generators(),
            ansatz,
        )
        finite_difference = finite_difference_gradient(thetas, parameter_index)
        assert np.isclose(gradient, finite_difference, atol=2e-3)


def test_reversed_hadamard_test_gradient():
    thetas = [0.3, 1.1, -0.4, 0.8]

    for parameter_index in range(len(thetas)):
        gradient = reversed_hadamard_test_gradient(
            thetas,
            parameter_index,
            2,
            observable,
            generators(),
            ansatz_segments,
        )
        finite_difference = finite_difference_gradient(thetas, parameter_index)
        assert np.isclose(gradient, finite_difference, atol=2e-3)


def test_mixed_generator_hadamard_gradients():
    theta = [0.37]

    def mixed_observable(qubits):
        return Z(qubits[0])

    def finite_difference(step=1e-4):
        forward = [theta[0] + step]
        backward = [theta[0] - step]
        return (
            cost(forward, 2, mixed_observable, mixed_generator_ansatz)
            - cost(backward, 2, mixed_observable, mixed_generator_ansatz)
        ) / (2 * step)

    expected = finite_difference()
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

    assert np.isclose(hadamard, expected, atol=2e-3)
    assert np.isclose(reversed_hadamard, expected, atol=2e-3)


def test_grouped_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]

    for parameter_index in range(len(thetas)):
        direct = hadamard_test_gradient(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz,
        )
        grouped, circuit_count = hadamard_test_gradient_grouped(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz,
        )

        assert np.isclose(grouped, direct, atol=2e-3)
        assert circuit_count == 1


def test_grouped_reversed_hadamard_gradients():
    thetas = [0.3, 1.1, -0.4, 0.8]

    for parameter_index in range(len(thetas)):
        direct = reversed_hadamard_test_gradient(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz_segments,
        )
        grouped, circuit_count = reversed_hadamard_test_gradient_grouped(
            thetas,
            parameter_index,
            2,
            grouped_observable,
            generators(),
            ansatz_segments,
        )

        assert np.isclose(grouped, direct, atol=2e-3)
        assert circuit_count == 3
