import numpy as np
from ket.gates import RY, Z

from gradients.psr.advanced import (
    GateInstruction,
    expectation as advanced_expectation,
    parameter_shift_gradient as advanced_parameter_shift_gradient,
)
from gradients.psr.main import expectation, parameter_shift_gradient


def test_scalar_parameter_shift():
    theta = 0.3

    cost = expectation(theta)
    gradient = parameter_shift_gradient(expectation, theta)

    assert np.isclose(cost, np.cos(theta), atol=1e-8)
    assert np.isclose(gradient, -np.sin(theta), atol=1e-8)


def test_instruction_parameter_shift():
    parameters = [0.3, 0.5]
    circuit = [
        GateInstruction(gate=RY, target=0, value=0.3, index=0),
        GateInstruction(gate=RY, target=0, value=0.5, index=1),
    ]

    cost = advanced_expectation(
        num_qubits=1,
        parameters=parameters,
        gates=circuit,
        operator=Z,
    )
    gradients = advanced_parameter_shift_gradient(
        num_qubits=1,
        gates=circuit,
        parameters=parameters,
    )

    assert np.isclose(cost, np.cos(sum(parameters)), atol=1e-8)
    assert np.allclose(
        gradients,
        [-np.sin(sum(parameters)), -np.sin(sum(parameters))],
        atol=1e-8,
    )
