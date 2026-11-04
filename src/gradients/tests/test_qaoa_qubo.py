from collections.abc import Sequence

import numpy as np
from ket import B, Process
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

COVARIANCE = [
    [1.08774352e-03, 2.59532811e-04, 1.80247155e-04, 3.21724369e-04],
    [2.59532811e-04, 4.43192629e-04, 7.43211072e-05, 2.27911525e-04],
    [1.80247155e-04, 7.43211072e-05, 3.89444953e-04, 1.37915422e-04],
    [3.21724369e-04, 2.27911525e-04, 1.37915422e-04, 8.75437564e-04],
]
EXPECTED_RETURN = [0.31542042, 0.0571331, 0.11430001, 0.30109367]
BUDGET = 2
RISK = 0.5
PENALTY = 1.0
NUM_NODES = len(EXPECTED_RETURN)
DEPTH = 2
PARAMETERS = [0.5] * (2 * DEPTH)
REPETITIONS = 10


def portfolio(
    exp_return: Sequence[float],
    cov_matrix: Sequence[Sequence[float]],
    budget: int,
    risk: float = RISK,
    penalty: float = PENALTY,
):
    def model(qubits: Quant):
        if not (0 <= risk <= 1):
            raise ValueError("risk must be between 0 and 1")
        if len(exp_return) != len(cov_matrix):
            raise ValueError("cov_matrix must be square")
        if not 0 <= budget <= len(exp_return):
            raise ValueError("budget is outside the valid range")

        assets = [B(qubit) for qubit in qubits]
        return -(
            sum(asset * value for asset, value in zip(assets, exp_return))
            - risk
            * sum(
                cov_matrix[i][j] * assets[i] * assets[j]
                for i in range(len(assets))
                for j in range(i + 1, len(assets))
            )
            - penalty * (budget - sum(assets)) ** 2
        )

    return model


MODEL = portfolio(EXPECTED_RETURN, COVARIANCE, BUDGET)


def mixer_h(qubits: Quant):
    with obs():
        return sum(X(qubit) for qubit in qubits)


def qaoa_parameter_gate(
    parameter_index: int,
    parameter_gate: ParameterGate,
    theta: float,
    qubits: Quant,
) -> None:
    hamiltonian = MODEL(qubits) if parameter_index % 2 == 0 else mixer_h(qubits)
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
        qaoa_parameter_gate(parameter_index, parameter_gate, theta, qubits)


def qaoa_segments(parameters: Sequence[float]) -> list[Segment]:
    return [
        lambda qubits, index=index, theta=theta: (
            H(qubits) if index == 0 else None,
            qaoa_parameter_gate(index, direct_parameter_gate, theta, qubits),
        )
        for index, theta in enumerate(parameters)
    ]


def qaoa_generators() -> list[GeneratorBuilder]:
    return [
        (lambda qubits: 2 * MODEL(qubits))
        if index % 2 == 0
        else (lambda qubits: 2 * mixer_h(qubits))
        for index in range(2 * DEPTH)
    ]


def qaoa_expectation(parameters: Sequence[float]) -> float:
    process = Process()
    qubits = process.alloc(NUM_NODES)
    qaoa_ansatz(parameters, qubits)
    with obs():
        observable = MODEL(qubits)
    return exp_value(observable).get()


def qaoa_native_gradient(parameters: Sequence[float]) -> list[float]:
    process = Process(gradient=True)
    qubits = process.alloc(NUM_NODES)
    registered = [process.param(theta) for theta in parameters]
    H(qubits)
    for parameter_index, theta in enumerate(registered):
        qaoa_parameter_gate(parameter_index, direct_parameter_gate, theta, qubits)
    with obs():
        observable = MODEL(qubits)
    exp_value(observable).get()
    return [parameter.grad for parameter in registered]


def qubo_energy(bits: int) -> float:
    assets = [(bits >> index) & 1 for index in range(NUM_NODES)]
    return -(
        sum(asset * value for asset, value in zip(assets, EXPECTED_RETURN))
        - RISK
        * sum(
            COVARIANCE[i][j] * assets[i] * assets[j]
            for i in range(NUM_NODES)
            for j in range(i + 1, NUM_NODES)
        )
        - PENALTY * (BUDGET - sum(assets)) ** 2
    )


def qaoa_psr_gradient(parameters: Sequence[float]) -> list[float]:
    cost_eigenvalues = [
        2 * qubo_energy(bits) for bits in range(1 << NUM_NODES)
    ]
    mixer_eigenvalues = [
        2
        * sum(
            1 if (bits >> index) & 1 else -1
            for index in range(NUM_NODES)
        )
        for bits in range(1 << NUM_NODES)
    ]
    gates = [
        GateInstruction(
            gate=lambda theta, qubits, index=index: evolve(
                (MODEL(qubits) if index % 2 == 0 else mixer_h(qubits)) * theta
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
    return parameter_shift_gradient(
        NUM_NODES,
        gates,
        parameters,
        operator=MODEL,
    )


def qaoa_hadamard_gradient(parameters: Sequence[float]) -> list[float]:
    return [
        hadamard_test_gradient(
            parameters,
            index,
            NUM_NODES,
            MODEL,
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
            MODEL,
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
            MODEL,
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
            MODEL,
            qaoa_generators(),
            qaoa_segments,
        )[0]
        for index in range(len(parameters))
    ]


def test_qaoa_qubo_gradient_methods_agree() -> None:
    native = qaoa_native_gradient(PARAMETERS)
    hadamard = qaoa_hadamard_gradient(PARAMETERS)
    reversed_hadamard = qaoa_reversed_gradient(PARAMETERS)
    grouped_hadamard = qaoa_grouped_hadamard_gradient(PARAMETERS)
    grouped_reversed = qaoa_grouped_reversed_gradient(PARAMETERS)

    assert np.allclose(hadamard, native, atol=2e-5)
    assert np.allclose(reversed_hadamard, native, atol=2e-5)
    assert np.allclose(grouped_hadamard, native, atol=2e-5)
    assert np.allclose(grouped_reversed, native, atol=2e-5)


def test_qaoa_qubo_gradient_performance() -> None:
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
    for name in (
        "KET native",
        "Hadamard",
        "Reversed Hadamard",
        "Grouped Hadamard",
        "Grouped Reversed Hadamard",
    ):
        assert np.allclose(results[name], results["KET native"], atol=2e-5)

    print("\nQAOA QUBO gradient performance (ms/evaluation):")
    for name, method in methods.items():
        timing = benchmark_gradient(method, REPETITIONS)
        print(f"{name:>18}: {timing:8.3f} ms")


if __name__ == "__main__":
    test_qaoa_qubo_gradient_methods_agree()
    test_qaoa_qubo_gradient_performance()
