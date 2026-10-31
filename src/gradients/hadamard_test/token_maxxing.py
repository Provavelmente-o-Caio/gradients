from collections.abc import Callable, Sequence
from time import perf_counter
from typing import TypeAlias

from ket.base import Process
from ket.base.quant import Quant
from ket.expv import Hamiltonian, Pauli
from ket.gates import CNOT, RY, H, I, S, X, Y, Z, evolve, obs
from ket.operations import ctrl, exp_value, inverse, kron

ObservableBuilder: TypeAlias = Callable[[Quant], Hamiltonian | Pauli]
GeneratorBuilder: TypeAlias = Callable[[Quant], Hamiltonian | Pauli]
ParameterGate: TypeAlias = Callable[[float, int, Callable[..., object], Quant], None]
CircuitBuilder: TypeAlias = Callable[[Sequence[float], Quant, ParameterGate], None]


def _terms(operator: Hamiltonian | Pauli) -> Sequence[Pauli]:
    """Return KET's native Pauli decomposition for an observable."""
    return operator.terms if isinstance(operator, Hamiltonian) else (operator,)


def _controlled_pauli(
    term: Pauli,
    qubits: Quant,
    control: Quant,
) -> None:
    gates = {"I": I, "X": X, "Y": Y, "Z": Z}
    circuit_qubits = [qubits[index] for index in range(len(qubits))]
    pauli_gates = [
        gates[term.map.get(qubit.qubits[0], "I")] for qubit in circuit_qubits
    ]
    ctrl(control, kron(*pauli_gates))(*circuit_qubits)


def direct_parameter_gate(
    theta: float,
    parameter: int,
    gate: Callable[..., object],
    target: Quant,
) -> None:
    """Apply a parameterized gate in an ordinary KET circuit."""
    del parameter
    gate(theta, target)


def ansatz(
    thetas: Sequence[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    """Example ansatz, independent from the gradient-estimation logic."""
    if len(thetas) != 4:
        raise ValueError("The ansatz requires exactly four parameters")

    parameter_gate(thetas[0], 0, RY, qubits[0])
    parameter_gate(thetas[1], 1, RY, qubits[1])
    CNOT(qubits[0], qubits[1])
    parameter_gate(thetas[2], 2, RY, qubits[0])
    parameter_gate(thetas[3], 3, RY, qubits[1])


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


def ket_gradient(
    thetas: Sequence[float],
    num_qubits: int,
    observable_builder: ObservableBuilder,
    circuit_builder: CircuitBuilder,
) -> list[float]:
    """Compute gradients with KET's automatic parameter-shift support."""
    process = Process(gradient=True)
    parameters = process.param(*thetas)
    parameter_list = parameters if isinstance(parameters, list) else [parameters]
    qubits = process.alloc(num_qubits)

    def parameter_gate(
        theta: float,
        parameter: int,
        gate: Callable[..., object],
        target: Quant,
    ) -> None:
        gate(theta, target)

    circuit_builder(parameters, qubits, parameter_gate)
    with obs():
        hamiltonian = observable_builder(qubits)
    exp_value(hamiltonian).get()
    return [parameter.grad for parameter in parameter_list]


def hadamard_test_term(
    thetas: Sequence[float],
    parameter_index: int,
    generator_builder: GeneratorBuilder,
    term_index: int,
    num_qubits: int,
    observable_builder: ObservableBuilder,
    circuit_builder: CircuitBuilder,
) -> float:
    """Evaluate one native KET Pauli term of a parameter gradient."""
    p = Process()
    ancilla = p.alloc()
    qubits = p.alloc(num_qubits)
    X(ancilla)
    H(ancilla)
    S(ancilla)

    with obs():
        generator = generator_builder(qubits)
        terms = _terms(generator)
        if not 0 <= term_index < len(terms):
            raise IndexError(f"Generator term index {term_index} is out of range")
        term = terms[term_index]

    def controlled_parameter_gate(
        theta: float,
        parameter: int,
        gate: Callable[..., object],
        target: Quant,
    ) -> None:
        if parameter == parameter_index:
            _controlled_pauli(term, qubits, ancilla)
        gate(theta, target)

    circuit_builder(thetas, qubits, controlled_parameter_gate)
    H(ancilla)
    with obs():
        full_observable = Z(ancilla) * observable_builder(qubits)
    return term.coef * exp_value(full_observable).get()


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
    probe = Process()
    probe_qubits = probe.alloc(num_qubits)
    with obs():
        terms = _terms(generators[parameter_index](probe_qubits))
    return sum(
        hadamard_test_term(
            thetas,
            parameter_index,
            generators[parameter_index],
            term_index,
            num_qubits,
            observable_builder,
            circuit_builder,
        )
        for term_index in range(len(terms))
    )


def mixed_generator(qubits: Quant) -> Hamiltonian:
    """A generator with multiple Pauli terms and non-binary eigenvalues."""
    with obs():
        return 0.6 * X(qubits[0]) + 0.8 * Z(qubits[1])


def mixed_generator_ansatz(
    thetas: Sequence[float],
    qubits: Quant,
    parameter_gate: ParameterGate = direct_parameter_gate,
) -> None:
    """Apply a unitary generated by a sum of Pauli operators."""
    theta = thetas[0] if isinstance(thetas, (list, tuple)) else thetas

    parameter_gate(
        theta,
        0,
        lambda theta, _target: evolve(mixed_generator(qubits) * (theta / 2)),
        qubits[0],
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


def hadamard_test_transition_amplitude(
    num_qubits: int,
    prepare_input: Callable[[Quant], None],
    prepare_output: Callable[[Quant], None],
    unitary: Callable[[Quant], None],
) -> float:
    """Estimate Re(<output|unitary|input>) with an ancilla interferometer."""
    process = Process()
    ancilla = process.alloc()
    qubits = process.alloc(num_qubits)

    H(ancilla)
    prepare_output(qubits)
    with inverse(process):
        ctrl(ancilla, prepare_output)(qubits)
    ctrl(ancilla, prepare_input)(qubits)
    ctrl(ancilla, unitary)(qubits)
    H(ancilla)

    with obs():
        return exp_value(Z(ancilla)).get()


def native_parameter_shift_transition_amplitude() -> None:
    """Document why KET's native gradient cannot evaluate this quantity."""
    raise TypeError(
        "KET's native parameter-shift API requires exp_value(Hamiltonian); "
        "a transition amplitude <phi|U|psi> is an overlap, not a Hamiltonian "
        "expectation value"
    )


if __name__ == "__main__":
    # ----------------------------------------------------------------
    # Example: 2-qubit hardware-efficient ansatz
    #   RY(t0) q0 | RY(t1) q1 -- CNOT(q0,q1) -- RY(t2) q0 | RY(t3) q1
    # Observable: Z0 Z1
    # ----------------------------------------------------------------
    NUM_QUBITS = 2

    def observable(q):
        return Z(q[0]) * Z(q[1])

    thetas = [0.3, 1.1, -0.4, 0.8]
    generators = [
        lambda q: Y(q[0]),
        lambda q: Y(q[1]),
        lambda q: Y(q[0]),
        lambda q: Y(q[1]),
    ]
    ket_parameter_shift_gradient = ket_gradient(thetas, NUM_QUBITS, observable, ansatz)

    print(f"Cost function value: {cost(thetas, NUM_QUBITS, observable, ansatz):.6f}\n")

    print(f"{'param':>6} | {'Hadamard test':>15} | {'KET shift':>12}")
    print("-" * 56)
    for k in range(len(thetas)):
        g_had = hadamard_test_gradient(
            thetas, k, NUM_QUBITS, observable, generators, ansatz
        )
        print(f"{k:>6} | {g_had:>15.6f} | {ket_parameter_shift_gradient[k]:>12.6f}")

    print("\nMixed-generator example:")
    mixed_thetas = [0.37]
    mixed_observable = lambda q: Z(q[0])
    mixed_hadamard = hadamard_test_gradient(
        mixed_thetas,
        0,
        NUM_QUBITS,
        mixed_observable,
        [mixed_generator],
        mixed_generator_ansatz,
    )
    mixed_ket = ket_gradient(
        mixed_thetas,
        NUM_QUBITS,
        mixed_observable,
        mixed_generator_ansatz,
    )
    print(f"Generator: 0.6*X(q[0]) + 0.8*Z(q[1])")
    print(f"Hadamard test: {mixed_hadamard:.6f}")
    print(f"KET gradient:  {mixed_ket[0]:.6f}")

    repetitions = 10
    hadamard_time = benchmark_gradient(
        lambda: hadamard_test_gradient(
            mixed_thetas,
            0,
            NUM_QUBITS,
            mixed_observable,
            [mixed_generator],
            mixed_generator_ansatz,
        ),
        repetitions,
    )
    ket_time = benchmark_gradient(
        lambda: ket_gradient(
            mixed_thetas,
            NUM_QUBITS,
            mixed_observable,
            mixed_generator_ansatz,
        )[0],
        repetitions,
    )
    print(f"Average runtime ({repetitions} repetitions):")
    print(f"Hadamard test: {hadamard_time:.3f} ms")
    print(f"KET gradient:  {ket_time:.3f} ms")

    print("\nTransition-amplitude example:")
    transition_amplitude = hadamard_test_transition_amplitude(
        1,
        prepare_input=lambda q: None,
        prepare_output=lambda q: H(q[0]),
        unitary=lambda q: Z(q),
    )
    print("Estimating <+|Z|0> without an expectation-value observable")
    print(f"Hadamard test: {transition_amplitude:.6f}")
    print(f"Known value:   {1 / 2**0.5:.6f}")
    try:
        native_parameter_shift_transition_amplitude()
    except TypeError as error:
        print(f"Native parameter shift: unavailable ({error})")
