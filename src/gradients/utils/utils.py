from ket.expv import Hamiltonian, Pauli
from collections.abc import Sequence

def pauli_terms(operator: Hamiltonian | Pauli) -> Sequence[Pauli]:
    """Return KET's native Pauli decomposition for an observable."""
    return operator.terms if isinstance(operator, Hamiltonian) else (operator,)
