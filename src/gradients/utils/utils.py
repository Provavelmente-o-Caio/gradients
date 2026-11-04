from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from time import perf_counter

from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import I, X, Y, Z, obs
from ket.operations import ctrl, exp_value, kron

type ObservableBuilder = Callable[[Quant], Hamiltonian | Pauli]
type GeneratorBuilder = Callable[[Quant], Hamiltonian | Pauli]
type ParameterGate = Callable[[float, int, Callable[..., object], Quant], None]
type CircuitBuilder = Callable[[Sequence[float], Quant, ParameterGate], None]
type Segment = Callable[[Quant], None]
type SegmentsBuilder = Callable[[Sequence[float]], list[Segment]]
type GeneratorTerm = tuple[complex, tuple[str, ...]]
type ParameterizedGate = Callable[[float, Quant], object]


@dataclass(frozen=True)
class GateInstruction:
    gate: ParameterizedGate
    target: int
    value: float
    index: int | None = None
    multiplier: float = 1.0

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
    return terms
