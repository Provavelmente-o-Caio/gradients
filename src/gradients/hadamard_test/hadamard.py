from collections.abc import Callable, Sequence
from functools import partial

from ket.base import Process
from ket.base.quant import Quant
from ket.gates import H, S, X, Z, obs
from ket.operations import exp_value

from gradients.utils.utils import (
    CircuitBuilder,
    GeneratorBuilder,
    GeneratorTerm,
    ObservableBuilder,
    controlled_pauli,
    group_qubit_commuting,
    grouped_expectation,
    pauli_expansion,
)


def hadamard_test_term(
    thetas: Sequence[float],
    parameter_index: int,
    term: GeneratorTerm,
    num_qubits: int,
    observable_builder: ObservableBuilder,
    circuit_builder: CircuitBuilder,
) -> float:
    """Evaluate one native KET Pauli term of a parameter gradient."""
    process = Process()
    ancilla = process.alloc()
    qubits = process.alloc(num_qubits)
    X(ancilla)
    H(ancilla)
    S(ancilla)
    coefficient, pauli_word = term

    call_index = 0

    def controlled_parameter_gate(
        theta: float,
        gate: Callable[..., object],
        target: Quant,
    ) -> None:
        nonlocal call_index
        if call_index == parameter_index:
            controlled_pauli(pauli_word, qubits, ancilla)
        call_index += 1
        gate(theta, target)

    circuit_builder(thetas, qubits, controlled_parameter_gate)
    H(ancilla)
    with obs():
        full_observable = Z(ancilla) * observable_builder(qubits)
    return coefficient * exp_value(full_observable).get()


def hadamard_test_gradient(
    thetas: Sequence[float],
    parameter_index: int,
    num_qubits: int,
    observable_builder: ObservableBuilder,
    generators: Sequence[GeneratorBuilder],
    circuit_builder: CircuitBuilder,
) -> float:
    """Sum the Hadamard-test terms for one parameter gradient.

    observable_builder : function(main_qubits) -> Pauli/Hamiltonian,
                          must be safe to call inside `with obs():`.
    """
    if not 0 <= parameter_index < len(generators):
        raise IndexError(f"Parameter index {parameter_index} is out of range")

    terms = pauli_expansion(generators[parameter_index], num_qubits)
    return sum(
        hadamard_test_term(
            thetas,
            parameter_index,
            term,
            num_qubits,
            observable_builder,
            circuit_builder,
        )
        for term in terms
    )


def ht_state(thetas, parameter_index, pauli_word, num_qubits, circuit_builder):
    process = Process()
    ancilla = process.alloc()
    qubits = process.alloc(num_qubits)
    X(ancilla)
    H(ancilla)
    S(ancilla)

    call_index = 0

    def controlled_parameter_gate(theta, gate, target):
        nonlocal call_index
        if call_index == parameter_index:
            controlled_pauli(pauli_word, qubits, ancilla)
        call_index += 1
        gate(theta, target)

    circuit_builder(thetas, qubits, controlled_parameter_gate)
    H(ancilla)
    return ancilla, qubits


def hadamard_test_gradient_grouped(
    thetas,
    parameter_index,
    num_qubits,
    observable_builder,
    generators,
    circuit_builder,
) -> tuple[float, int]:
    if not 0 <= parameter_index < len(generators):
        raise IndexError(f"Parameter index {parameter_index} is out of range")
    gen_terms = pauli_expansion(generators[parameter_index], num_qubits)
    groups = group_qubit_commuting(pauli_expansion(observable_builder, num_qubits))
    grad = 0.0
    for beta, word in gen_terms:
        build = partial(
            ht_state, thetas, parameter_index, word, num_qubits, circuit_builder
        )
        grad += beta * grouped_expectation(build, groups)
    return grad, len(gen_terms) * len(groups)
