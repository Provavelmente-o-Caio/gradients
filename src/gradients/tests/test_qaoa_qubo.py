from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    ansatz as default_ansatz,
    benchmark_gradient,
    pauli_expansion,
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
REPETITIONS = 100
TRAINING_STEPS = 4
LEARNING_RATE = 0.05
CIRCUIT_BUDGET = 200


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
        lambda value, _target: evolve(hamiltonian * value),
        qubits,
    )


def qaoa_ansatz(
    parameters: Sequence[float],
    qubits: Quant,
    parameter_gate: ParameterGate = default_ansatz,
) -> None:
    H(qubits)
    for parameter_index, theta in enumerate(parameters):
        qaoa_parameter_gate(parameter_index, parameter_gate, theta, qubits)


def qaoa_segments(parameters: Sequence[float]) -> list[Segment]:
    return [
        lambda qubits, index=index, theta=theta: (
            H(qubits) if index == 0 else None,
            qaoa_parameter_gate(index, default_ansatz, theta, qubits),
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


def qaoa_ket_gradient(
    parameters: Sequence[float],
    strategy: str,
) -> list[float]:
    process = Process(gradient=strategy)
    qubits = process.alloc(NUM_NODES)
    registered = [process.param(theta) for theta in parameters]
    H(qubits)
    for parameter_index, theta in enumerate(registered):
        qaoa_parameter_gate(parameter_index, default_ansatz, theta, qubits)
    with obs():
        observable = MODEL(qubits)
    exp_value(observable).get()
    return [parameter.grad for parameter in registered]


def qaoa_native_psr_gradient(parameters: Sequence[float]) -> list[float]:
    return qaoa_ket_gradient(parameters, "ParameterShiftRule")


def qaoa_native_hadamard_gradient(parameters: Sequence[float]) -> list[float]:
    return qaoa_ket_gradient(parameters, "HadamardTest")


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


def _gradient_with_circuit_count(
    name: str,
    parameters: Sequence[float],
) -> tuple[list[float], int]:
    if name in {"KET PSR", "KET Hadamard"}:
        gradient = (
            qaoa_native_psr_gradient(parameters)
            if name == "KET PSR"
            else qaoa_native_hadamard_gradient(parameters)
        )
        return gradient, 1
    if name == "PSR":
        count = 0
        cost_eigenvalues = [2 * qubo_energy(bits) for bits in range(1 << NUM_NODES)]
        mixer_eigenvalues = [
            2
            * sum(
                1 if (bits >> index) & 1 else -1
                for index in range(NUM_NODES)
            )
            for bits in range(1 << NUM_NODES)
        ]
        for index in range(len(parameters)):
            eigenvalues = cost_eigenvalues if index % 2 == 0 else mixer_eigenvalues
            count += 2 * len(spectral_shift_rule(eigenvalues))
        return qaoa_psr_gradient(parameters), count
    if name in {"Hadamard", "Reversed Hadamard"}:
        observable_terms = len(pauli_expansion(MODEL, NUM_NODES))
        generator_terms = sum(
            len(pauli_expansion(generator, NUM_NODES))
            for generator in qaoa_generators()
        )
        gradient = (
            qaoa_hadamard_gradient(parameters)
            if name == "Hadamard"
            else qaoa_reversed_gradient(parameters)
        )
        return gradient, observable_terms * generator_terms
    if name == "Grouped Hadamard":
        values = [
            hadamard_test_gradient_grouped(
                parameters,
                index,
                NUM_NODES,
                MODEL,
                qaoa_generators(),
                qaoa_ansatz,
            )
            for index in range(len(parameters))
        ]
    else:
        values = [
            reversed_hadamard_test_gradient_grouped(
                parameters,
                index,
                NUM_NODES,
                MODEL,
                qaoa_generators(),
                qaoa_segments,
            )
            for index in range(len(parameters))
        ]
    return [value for value, _ in values], sum(count for _, count in values)


def qaoa_budget_training_plot(
    initial_parameters: Sequence[float] = PARAMETERS,
    budget: int = CIRCUIT_BUDGET,
    learning_rate: float = LEARNING_RATE,
    output_path: str | None = None,
) -> go.Figure:
    """Compare exact training against a common circuit-execution budget."""
    methods = (
        "PSR",
        "KET PSR",
        "KET Hadamard",
        "Hadamard",
        "Reversed Hadamard",
        "Grouped Hadamard",
        "Grouped Reversed Hadamard",
    )
    figure = go.Figure()
    for name in methods:
        parameters = list(initial_parameters)
        executions = 1
        x_values = [executions]
        costs = [qaoa_expectation(parameters)]
        while executions < budget:
            gradient, gradient_executions = _gradient_with_circuit_count(
                name, parameters
            )
            if executions + gradient_executions + 1 > budget:
                break
            parameters = [
                parameter - learning_rate * derivative
                for parameter, derivative in zip(parameters, gradient)
            ]
            executions += gradient_executions + 1
            x_values.append(executions)
            costs.append(qaoa_expectation(parameters))
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=costs,
                mode="lines+markers",
                name=name,
                hovertemplate=f"{name}<br>circuits=%{{x}}<br>cost=%{{y:.12f}}<extra></extra>",
            )
        )
    figure.update_layout(
        title="QAOA QUBO exact training loss by circuit budget",
        xaxis_title="Distinct circuit executions",
        yaxis_title="Loss",
    )
    if output_path is not None:
        figure.write_html(output_path)
    return figure


def qaoa_training_plot(
    initial_parameters: Sequence[float] = PARAMETERS,
    steps: int = TRAINING_STEPS,
    learning_rate: float = LEARNING_RATE,
    output_path: str | None = None,
) -> go.Figure:
    """Plot cost evolution while training with each gradient method."""
    methods = {
        "PSR": qaoa_psr_gradient,
        "KET PSR": qaoa_native_psr_gradient,
        "KET Hadamard": qaoa_native_hadamard_gradient,
        "Hadamard": qaoa_hadamard_gradient,
        "Reversed Hadamard": qaoa_reversed_gradient,
        "Grouped Hadamard": qaoa_grouped_hadamard_gradient,
        "Grouped Reversed Hadamard": qaoa_grouped_reversed_gradient,
    }
    histories = {}
    for name, gradient_method in methods.items():
        parameters = list(initial_parameters)
        costs = [qaoa_expectation(parameters)]
        for _ in range(steps):
            gradient = gradient_method(parameters)
            parameters = [
                parameter - learning_rate * derivative
                for parameter, derivative in zip(parameters, gradient)
            ]
            costs.append(qaoa_expectation(parameters))
        histories[name] = costs

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Expected cost", "Difference from KET PSR"),
    )
    styles = [
        ("circle", "solid"),
        ("square", "dash"),
        ("diamond", "dot"),
        ("triangle-up", "dashdot"),
        ("triangle-down", "longdash"),
        ("star", "longdashdot"),
        ("hexagon", "solid"),
    ]
    for (name, costs), (symbol, dash) in zip(histories.items(), styles):
        figure.add_trace(
            go.Scatter(
                x=list(range(steps + 1)),
                y=costs,
                mode="lines+markers",
                name=name,
                marker_symbol=symbol,
                line_dash=dash,
                hovertemplate=f"{name}<br>step=%{{x}}<br>cost=%{{y:.12f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=list(range(steps + 1)),
                y=[
                    value - reference
                    for value, reference in zip(costs, histories["KET PSR"])
                ],
                mode="lines+markers",
                name=f"{name} difference",
                marker_symbol=symbol,
                line_dash=dash,
                showlegend=False,
                hovertemplate=f"{name}<br>step=%{{x}}<br>difference=%{{y:.3e}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.update_layout(
        title="QAOA QUBO training cost",
        height=800,
        xaxis2_title="Optimization step",
        yaxis_title="Expected cost",
        yaxis2_title="Cost difference",
    )
    if output_path is not None:
        figure.write_html(output_path)
    return figure


def test_qaoa_qubo_gradient_methods_agree() -> None:
    native_psr = qaoa_native_psr_gradient(PARAMETERS)
    native_hadamard = qaoa_native_hadamard_gradient(PARAMETERS)
    hadamard = qaoa_hadamard_gradient(PARAMETERS)
    reversed_hadamard = qaoa_reversed_gradient(PARAMETERS)
    grouped_hadamard = qaoa_grouped_hadamard_gradient(PARAMETERS)
    grouped_reversed = qaoa_grouped_reversed_gradient(PARAMETERS)

    assert np.allclose(hadamard, native_hadamard, atol=2e-5)
    assert np.allclose(reversed_hadamard, native_hadamard, atol=2e-5)
    assert np.allclose(grouped_hadamard, native_hadamard, atol=2e-5)
    assert np.allclose(grouped_reversed, native_hadamard, atol=2e-5)
    assert np.allclose(native_psr, native_hadamard, atol=2e-5)


def test_qaoa_qubo_gradient_performance() -> None:
    methods = {
        "PSR": lambda: qaoa_psr_gradient(PARAMETERS),
        "KET PSR": lambda: qaoa_native_psr_gradient(PARAMETERS),
        "KET Hadamard": lambda: qaoa_native_hadamard_gradient(PARAMETERS),
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
        "KET PSR",
        "KET Hadamard",
        "Hadamard",
        "Reversed Hadamard",
        "Grouped Hadamard",
        "Grouped Reversed Hadamard",
    ):
        assert np.allclose(results[name],         results["KET Hadamard"], atol=2e-5)

    print("\nQAOA QUBO gradient performance (ms/evaluation):")
    for name, method in methods.items():
        timing = benchmark_gradient(method, REPETITIONS)
        print(f"{name:>18}: {timing:8.3f} ms")


def test_qaoa_qubo_training_plot() -> None:
    figure = qaoa_training_plot(steps=1)
    assert len(figure.data) == 14
    assert all(len(trace.y) == 2 for trace in figure.data)


def test_qaoa_qubo_budget_training_plot() -> None:
    figure = qaoa_budget_training_plot(budget=20)
    assert len(figure.data) == 7
    assert all(len(trace.x) == len(trace.y) for trace in figure.data)


if __name__ == "__main__":
    test_qaoa_qubo_gradient_methods_agree()
    test_qaoa_qubo_gradient_performance()
    qaoa_training_plot(output_path="qaoa_qubo_training.html")
    qaoa_budget_training_plot(output_path="qaoa_qubo_circuit_budget.html")
