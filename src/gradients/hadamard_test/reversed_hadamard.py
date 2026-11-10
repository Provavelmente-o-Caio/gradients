from collections.abc import Sequence
from functools import partial

from ket.base import Process
from ket.gates import H, S, X, Z, obs
from ket.operations import adj, exp_value

from gradients.utils.utils import (
    GeneratorBuilder,
    GeneratorTerm,
    MeasurementGroup,
    ObservableBuilder,
    PauliTerm,
    Segment,
    SegmentsBuilder,
    apply_segments,
    controlled_pauli,
    group_qubit_commuting,
    grouped_expectation,
    pauli_expansion,
)

adjoint_apply_segments = adj(apply_segments)


def reversed_hadamard_test_term(
    parameter_index: int,
    term: GeneratorTerm,
    num_qubits: int,
    generator_builder: GeneratorBuilder,
    segments: Sequence[Segment],
) -> float:
    """Evaluate one native KET Pauli term of a parameter gradient."""
    process = Process()
    ancilla = process.alloc()
    qubits = process.alloc(num_qubits)

    X(ancilla)
    H(ancilla)
    S(ancilla)

    suffix = segments[parameter_index + 1 :]
    coefficient, pauli_word = term

    apply_segments(segments, qubits)
    controlled_pauli(pauli_word, qubits, ancilla)
    adjoint_apply_segments(suffix, qubits)

    H(ancilla)
    with obs():
        full_observable = Z(ancilla) * generator_builder(qubits)
    return coefficient * exp_value(full_observable).get()


def reversed_hadamard_test_gradient(
    thetas: Sequence[float],
    parameter_index: int,
    num_qubits: int,
    observable_builder: ObservableBuilder,
    generators: Sequence[GeneratorBuilder],
    segments_builder: SegmentsBuilder,
) -> float:
    """Sum the Hadamard-test terms for one parameter gradient.

    observable_builder : function(main_qubits) -> Pauli/Hamiltonian,
                          must be safe to call inside `with obs():`.
    """
    if not 0 <= parameter_index < len(generators):
        raise IndexError(f"Parameter index {parameter_index} is out of range")
    terms = pauli_expansion(observable_builder, num_qubits)
    segments = segments_builder(thetas)
    return -sum(
        reversed_hadamard_test_term(
            parameter_index,
            term,
            num_qubits,
            generators[parameter_index],
            segments,
        )
        for term in terms
    )


def rht_state(thetas, parameter_index, pauli_word, num_qubits, segments_builder):
    process = Process()
    ancilla = process.alloc()
    qubits = process.alloc(num_qubits)
    X(ancilla)
    H(ancilla)
    S(ancilla)
    segments = segments_builder(thetas)
    apply_segments(segments, qubits)
    controlled_pauli(pauli_word, qubits, ancilla)
    adjoint_apply_segments(segments[parameter_index + 1 :], qubits)
    H(ancilla)
    return ancilla, qubits


def reversed_hadamard_test_gradient_grouped(
    thetas,
    parameter_index,
    num_qubits,
    observable_builder,
    generators,
    segments_builder,
    *,
    observable_terms: Sequence[PauliTerm] | None = None,
    generator_groups: Sequence[MeasurementGroup] | None = None,
) -> tuple[float, int]:
    if not 0 <= parameter_index < len(generators):
        raise IndexError(f"Parameter index {parameter_index} is out of range")
    obs_terms = (
        observable_terms
        if observable_terms is not None
        else pauli_expansion(observable_builder, num_qubits)
    )
    groups = (
        generator_groups
        if generator_groups is not None
        else group_qubit_commuting(
            pauli_expansion(generators[parameter_index], num_qubits)
        )
    )
    grad = 0.0
    for alpha, word in obs_terms:
        build = partial(
            rht_state, thetas, parameter_index, word, num_qubits, segments_builder
        )
        grad += alpha * grouped_expectation(build, groups)
    return -grad, len(obs_terms) * len(groups)
