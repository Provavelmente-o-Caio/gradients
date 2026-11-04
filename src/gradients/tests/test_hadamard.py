import numpy as np
from ket.gates import Y, Z

from gradients.hadamard_test.hadamard import (
    ansatz,
    hadamard_test_gradient,
)
from gradients.hadamard_test.reversed_hadamard import (
    ansatz_segments,
    reversed_hadamard_test_gradient,
)
from gradients.utils.utils import cost


def observable(qubits):
    return Z(qubits[0]) * Z(qubits[1])


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
