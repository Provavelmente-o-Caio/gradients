from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import pi
from time import perf_counter

import numpy as np
from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import H, I, S, X, Y, Z, obs
from ket.operations import ctrl, exp_value, kron

type ObservableBuilder = Callable[[Quant], Hamiltonian | Pauli]
type GeneratorBuilder = Callable[[Quant], Hamiltonian | Pauli]
type ParameterGate = Callable[[float, int, Callable[..., object], Quant], None]
type CircuitBuilder = Callable[[Sequence[float], Quant, ParameterGate], None]
type Segment = Callable[[Quant], None]
type SegmentsBuilder = Callable[[Sequence[float]], list[Segment]]
type GeneratorTerm = tuple[complex, tuple[str, ...]]
type ParameterizedGate = Callable[..., object]
type PauliWord = tuple[str, ...]
type PauliTerm = tuple[float, PauliWord]
type ShiftRule = tuple[tuple[float, float], ...]


@dataclass
class MeasurementGroup:
    basis: list[str]
    terms: list[PauliTerm] = field(default_factory=list)


def group_compatible(basis: list[str], word: PauliWord) -> bool:
    return all(b == "I" or w == "I" or b == w for b, w in zip(basis, word))


def group_qubit_commuting(terms: Sequence[PauliTerm]) -> list[MeasurementGroup]:
    groups: list[MeasurementGroup] = []
    for coef, word in sorted(terms, key=lambda t: -abs(t[0])):
        for group in groups:
            if group_compatible(group.basis, word):
                group.basis = [b if b != "I" else w for b, w in zip(group.basis, word)]
                group.terms.append((coef, word))
                break
        else:
            groups.append(MeasurementGroup(list(word), [(coef, word)]))
    return groups


def rotate_to_z_basis(basis: Sequence[str], qubits: Quant) -> None:
    for i, letter in enumerate(basis):
        if letter == "X":
            H(qubits[i])
        elif letter == "Y":
            S(qubits[i])
            S(qubits[i])
            S(qubits[i])  # S† = S^3
            H(qubits[i])


def group_hamiltonian(group: MeasurementGroup, ancilla: Quant, qubits: Quant):
    """After rotation, every term is Z_ancilla times Z on its support."""
    with obs():
        total = None
        for coef, word in group.terms:
            op = Z(ancilla)
            for i, letter in enumerate(word):
                if letter != "I":
                    op = op * Z(qubits[i])
            term = coef * op
            total = term if total is None else total + term
    return total


def grouped_expectation(build_state, groups: Sequence[MeasurementGroup]) -> float:
    """One circuit execution per group. build_state() returns (ancilla, qubits)."""
    mu = 0.0
    for group in groups:
        ancilla, qubits = build_state()
        rotate_to_z_basis(group.basis, qubits)
        mu += exp_value(group_hamiltonian(group, ancilla, qubits)).get()
    return mu


@dataclass(frozen=True)
class GateInstruction:
    gate: ParameterizedGate
    target: int | tuple[int, ...] | None
    value: float
    index: int | None = None
    multiplier: float = 1.0
    shift_rule: ShiftRule | None = None

    def param_index(self) -> tuple[int | None, float]:
        return self.index, self.multiplier


def shift_param(
    gates: Sequence[GateInstruction],
    shift_inst: int,
    shift_amt: float,
) -> list[GateInstruction]:
    """Return a circuit with one gate parameter shifted."""
    return [
        replace(instruction, value=instruction.value + shift_amt)
        if instruction_index == shift_inst
        else instruction
        for instruction_index, instruction in enumerate(gates)
    ]


def spectral_shift_rule(eigenvalues: Sequence[float]) -> ShiftRule:
    """Build an exact symmetric shift rule from generator eigenvalues."""
    gaps = sorted(
        {
            abs(left - right) / 2
            for left in eigenvalues
            for right in eigenvalues
            if abs(left - right) > 1e-12
        }
    )
    if not gaps:
        return ()

    shifts = np.arange(1, len(gaps) + 1) * pi / (2 * max(gaps) + 1)
    matrix = 2 * np.sin(np.outer(gaps, shifts))
    coefficients = np.linalg.solve(matrix, np.asarray(gaps))
    return tuple(
        (float(coefficient), float(shift))
        for coefficient, shift in zip(coefficients, shifts)
    )


def benchmark_gradient(
    gradient: Callable[[], float],
    repetitions: int = 10,
) -> float:
    """Return the average runtime of a gradient evaluation in milliseconds."""
    if repetitions <= 0:
        raise ValueError("The number of repetitions must be positive")

    gradient()
    start = perf_counter()
    for _ in range(repetitions):
        gradient()
    return (perf_counter() - start) * 1000 / repetitions


def cost(
    thetas: Sequence[float],
    num_qubits: int,
    observable_builder: ObservableBuilder,
    circuit_builder: CircuitBuilder,
) -> float:
    """Return ``<M>`` for the circuit at parameters ``thetas``."""
    p = Process()
    qubits = p.alloc(num_qubits)
    circuit_builder(thetas, qubits, direct_parameter_gate)
    with obs():
        hamiltonian = observable_builder(qubits)
    return exp_value(hamiltonian).get()


def controlled_pauli(
    pauli_word: tuple[str, ...],
    qubits: Quant,
    control: Quant,
) -> None:
    gates = {"I": I, "X": X, "Y": Y, "Z": Z}
    circuit_qubits = [qubits[index] for index in range(len(qubits))]
    pauli_gates = [gates[pauli_word[index]] for index in range(len(circuit_qubits))]
    ctrl(control, kron(*pauli_gates))(*circuit_qubits)


def pauli_terms(operator: Hamiltonian | Pauli) -> Sequence[Pauli]:
    """Return KET's native Pauli decomposition for an observable."""
    return operator.terms if isinstance(operator, Hamiltonian) else (operator,)


def direct_parameter_gate(
    theta: float,
    parameter: int,
    gate: Callable[..., object],
    target: Quant,
) -> None:
    """Apply a parameterized gate in an ordinary KET circuit."""
    gate(theta, target)


def apply_segments(segments: Sequence[Segment], qubits: Quant):
    for segment in segments:
        segment(qubits)


def pauli_expansion(builder: Callable[[Quant], Hamiltonian | Pauli], num_qubits: int):
    """Return KET's native Pauli decomposition for an operator."""
    process = Process()
    qubits = process.alloc(num_qubits)
    with obs():
        operator = builder(qubits)
        terms = tuple(
            (
                term.coef.real,
                tuple(term.map.get(q.qubits[0], "I") for q in qubits),
            )
            for term in pauli_terms(operator)
        )
    terms = tuple(t for t in terms if any(p != "I" for p in t[1]))
    return terms
