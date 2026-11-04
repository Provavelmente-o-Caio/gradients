from collections.abc import Callable, Sequence

from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import obs


def pauli_terms(operator: Hamiltonian | Pauli) -> Sequence[Pauli]:
    """Return KET's native Pauli decomposition for an observable."""
    return operator.terms if isinstance(operator, Hamiltonian) else (operator,)


def pauli_expansion(
    builder: Callable[[Quant], Hamiltonian | Pauli], num_qubits: int
):
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
