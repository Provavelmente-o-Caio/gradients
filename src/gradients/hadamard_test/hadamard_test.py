from collections.abc import Callable

from ket.base import Process
from ket.gates import RX, RY, RZ, H, I, S, X, Y, Z


def pauli_decomposition(word: str) -> Callable[..., object]:
    gates = {"I": I, "X": X, "Y": Y, "Z": Z}

    if not word or any(symbol not in gates for symbol in word):
        raise ValueError(f"Invalid Pauli word: {word!r}")

    def gate(*targets: Quant):
        if len(targets) != len(word):
            raise ValueError(
                f"Pauli word {word!r} requires {len(word)} targets, "
                f"received {len(targets)}"
            )

        for symbol, target in zip(word, targets):
            gates[symbol](target)

    return gate


def generator(gate):
    if gate is RY:
        return Y
    elif gate is RZ:
        return Z
    elif gate is RX:
        return X
    else:
        raise NotImplementedError(f"Generator for {gate} is not implemented")


# |ψ(θ)> = RY(θ)|0>
def ansatz(thetas: list[float], qubits, gates=[RY]) -> None:
    if len(thetas) != len(qubits) or len(gates) != len(qubits):
        raise ValueError("You must select a gate for each qubit")

    for i in range(len(qubits)):
        gates[i](thetas[i], qubits[i])


def hadamard_test_gradient(
        num_qubits: int,
        thetas: list[float],
        k: int, observable,
        generators,
        gate_ansatz
    ):
    process = Process()

    # QUBIT ANCILLA
    ancilla = process.alloc(1)

    # |0> -> |1>
    _ = X(ancilla)

    _ = H(ancilla)
    _ = S(ancilla)

    # |psi>
    qubits = process.alloc(num_qubits)

    # montando estado quantico
    gradient = 0.0
    try:
        gate_ansatz(thetas, qubits)
    except:
        print("Exception")

    _ = H(ancilla)

    return gradient


if __name__ == "__main__":

    def observable(q):
        return Z(q[0]) * Z(q[1])

    thetas = [0.3, 1.1, -0.4, 0.8]

    generators = [
        [(1.0, "YI")],
        [(1.0, "IY")],
        [(1.0, "YI")],
        [(1.0, "IY")]
    ]

    for k in range(len(thetas)):
        hadamard_test_gradient(2, thetas, k, generators, ansatz)
