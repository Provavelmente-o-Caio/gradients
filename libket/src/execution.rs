// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Execution backend abstractions for the Libket quantum computing library.
//!
//! This module defines the traits and data types that decouple the compiler
//! front-end from the concrete quantum hardware or simulator backend:
//!
//! - [`LiveExecution`]: a backend that receives gates one at a time and
//!   returns measurement results immediately (mid-circuit feedback). Use this
//!   mode when the circuit outcome influences subsequent gate choices
//!   (e.g., adaptive circuits, mid-circuit resets).
//! - [`BatchExecution`]: a backend that receives a fully-compiled native gate
//!   sequence and returns aggregate results (histograms, expectation values).
//!   Use this mode for variational algorithms, sampling, and expectation-value
//!   estimation where the full circuit is known upfront.
//! - [`NativeGateSet`]: an optional translation layer that maps the Libket
//!   gate IR to a hardware-specific gate vocabulary. If omitted, the
//!   built-in `RzRyCX` translation is used, which emits
//!   `rz`, `ry`, and `cnot` native gates.
//! - [`QuantumExecution`]: the union of both execution modes, stored inside a
//!   [`crate::process::Process`].
//!
//! ## Error handling
//!
//! All trait methods return `Result<_, KetError>`. Backends should return
//! [`KetError::ExecutionFailed`] for unrecoverable hardware faults and
//! [`KetError::ShotCountInvalid`] when a requested shot count is out of range.

use num::complex::ComplexFloat;
use serde::{Deserialize, Serialize};

use crate::{
    error::KetError,
    ir::{gate::GateInstruction, hamiltonian::Hamiltonian},
    matrix::Matrix,
};

/// A bit-string measurement outcome encoded as a vector of 64-bit words.
///
/// Each element in the outer `Vec` is one measurement shot; within a shot the
/// qubit outcomes are packed into 64-bit words (LSB = qubit 0).
pub type BitString = Vec<u64>;

/// The quantum-state snapshot returned by a `dump` operation.
///
/// The `i`-th basis state has amplitude `amplitudes_real[i] + i·amplitudes_imag[i]`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DumpData {
    /// Basis states present in the superposition (as bit-strings).
    pub basis_states: Vec<BitString>,
    /// Real parts of the corresponding probability amplitudes.
    pub amplitudes_real: Vec<f64>,
    /// Imaginary parts of the corresponding probability amplitudes.
    pub amplitudes_imag: Vec<f64>,
}

/// The result of a `sample` operation: `(bit_strings, counts)`.
///
/// Each element `bit_strings[i]` is a measured bit-string and `counts[i]` is
/// the number of times it was observed across all shots.
pub type SampleData = (Vec<BitString>, Vec<usize>);

/// A backend that executes gates as they arrive and can return mid-circuit
/// measurement results immediately.
pub trait LiveExecution {
    /// Dispatches a single logical gate instruction to the backend.
    ///
    /// # Errors
    ///
    /// Implementations may return a [`KetError`] if gate dispatch fails.
    fn compute_gate(&mut self, gate: &GateInstruction) -> Result<(), KetError>;

    /// Dispatches a sequence of already-translated native gate instructions to
    /// the backend.
    ///
    /// Called by the live-execution path when a [`NativeGateSet`] translation
    /// layer is configured. The `gates` slice contains hardware-specific
    /// instructions produced by [`NativeGateSet::translate`] or
    /// [`NativeGateSet::cnot`].
    ///
    /// # Errors
    ///
    /// Implementations may return a [`KetError`] if native gate dispatch fails.
    fn compute_native_gates(&mut self, gates: &[NativeGate]) -> Result<(), KetError>;

    /// Collapses and reads out the specified `qubits`, returning the result as
    /// a packed integer (bit `i` corresponds to `qubits[i]`).
    ///
    /// # Errors
    ///
    /// Returns a [`KetError`] if measurement fails on the backend.
    fn measure(&mut self, qubits: &[usize]) -> Result<u64, KetError>;

    /// Returns a full state-vector snapshot restricted to `qubits`.
    ///
    /// # Errors
    ///
    /// Returns a [`KetError`] if state dumping fails on the backend.
    fn dump(&mut self, qubits: &[usize]) -> Result<DumpData, KetError>;

    /// Samples the measurement distribution of `qubits` over `shots` repetitions.
    ///
    /// # Errors
    ///
    /// Returns a [`KetError`] if sampling fails on the backend.
    fn sample(&mut self, qubits: &[usize], shots: usize) -> Result<SampleData, KetError>;

    /// Computes the expectation value of `hamiltonian` with respect to the
    /// current quantum state.
    ///
    /// # Errors
    ///
    /// Returns a [`KetError`] if expectation value estimation fails on the backend.
    fn exp_value(&mut self, hamiltonian: Hamiltonian) -> Result<f64, KetError>;
}

impl std::fmt::Debug for dyn LiveExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LiveExecution")
    }
}

/// A hardware-native gate: `(name, angles, qubit_indices)`.
///
/// The `name` string identifies the gate in the target backend's vocabulary
/// (e.g. `"cnot"`, `"rx"`, `"h"`). `angles` holds rotation parameters (may be empty).
/// `qubit_indices` lists the qubits the gate acts on.
pub type NativeGate = (String, Vec<f64>, Vec<usize>);

/// A backend that receives a fully-compiled gate sequence and returns
/// aggregate results without mid-circuit feedback.
///
/// There are two families of methods: **IR-level** (`sample`, `exp_value`,
/// `gradient`) which receive [`GateInstruction`] slices directly from the
/// Libket IR, and **native-level** (`sample_native`, `exp_value_native`) which
/// receive sequences of [`NativeGate`] tuples already translated through a
/// [`NativeGateSet`].
///
/// All methods have a default implementation that returns an error, so
/// implementors only need to override the methods they support.
pub trait BatchExecution {
    /// Executes `gates` and samples the state of `qubits_to_sample` over
    /// `shots` shots, receiving gates in the Libket IR format.
    ///
    /// Implement this method when the backend can accept [`GateInstruction`]
    /// slices directly (i.e., no [`NativeGateSet`] translation is configured).
    ///
    /// # Errors
    ///
    /// Returns [`KetError::GateUnsupported`] by default. Implementations should
    /// return [`KetError::ExecutionFailed`] for backend faults or
    /// [`KetError::ShotCountInvalid`] for an out-of-range shot count.
    fn sample(
        &self,
        _gates: &[GateInstruction],
        _qubits_to_sample: &[usize],
        _shots: usize,
    ) -> Result<SampleData, KetError> {
        Err(KetError::GateUnsupported)
    }

    /// Executes `gates` and computes the expectation value of each Hamiltonian
    /// in `hamiltonian_list`, receiving gates in the Libket IR format.
    ///
    /// Implement this method when the backend can accept [`GateInstruction`]
    /// slices directly (i.e., no [`NativeGateSet`] translation is configured).
    ///
    /// # Errors
    ///
    /// Returns [`KetError::GateUnsupported`] by default. Implementations should
    /// return [`KetError::ExecutionFailed`] for backend faults.
    fn exp_value(
        &self,
        _gates: &[GateInstruction],
        _hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, KetError> {
        Err(KetError::GateUnsupported)
    }

    /// Executes `gates` and samples the state of `qubits_to_sample` over
    /// `shots` shots, receiving pre-translated native gates.
    ///
    /// Called when a [`NativeGateSet`] translation layer is configured. The
    /// `gates` slice contains hardware-specific instructions produced by
    /// [`NativeGateSet::translate`] or [`NativeGateSet::cnot`].
    ///
    /// # Errors
    ///
    /// Returns [`KetError::GateUnsupported`] by default. Implementations should
    /// return [`KetError::ExecutionFailed`] for backend faults or
    /// [`KetError::ShotCountInvalid`] for an out-of-range shot count.
    fn sample_native(
        &self,
        _gates: &[NativeGate],
        _qubits_to_sample: &[usize],
        _shots: usize,
    ) -> Result<SampleData, KetError> {
        Err(KetError::GateUnsupported)
    }

    /// Executes `gates` and computes the expectation value of each Hamiltonian
    /// in `hamiltonian_list`, receiving pre-translated native gates.
    ///
    /// Called when a [`NativeGateSet`] translation layer is configured. The
    /// `gates` slice contains hardware-specific instructions produced by
    /// [`NativeGateSet::translate`] or [`NativeGateSet::cnot`].
    ///
    /// # Errors
    ///
    /// Returns [`KetError::NativeGateUnsupported`] by default. Implementations
    /// should return [`KetError::ExecutionFailed`] for backend faults.
    fn exp_value_native(
        &self,
        _gates: &[NativeGate],
        _hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, KetError> {
        Err(KetError::NativeGateUnsupported)
    }

    /// Computes the expectation value and its gradient with respect to all
    /// circuit parameters in a single backend call.
    ///
    /// Returns `(expectation_value, gradient_vector)` where
    /// `gradient_vector[i]` is `∂⟨H⟩/∂θᵢ` for parameter `i`.
    ///
    /// This method is used when [`GradientStrategy::Native`] is selected. If
    /// the backend cannot compute gradients natively, implement
    /// [`GradientStrategy::ParameterShiftRule`] instead.
    ///
    /// # Errors
    ///
    /// Returns [`KetError::NativeGradientUnsuported`] by default. Implementations
    /// should return [`KetError::ExecutionFailed`] for backend faults.
    fn gradient(
        &self,
        _gates: &[GateInstruction],
        _hamiltonian: &Hamiltonian,
    ) -> Result<(f64, Vec<f64>), KetError> {
        Err(KetError::NativeGradientUnsuported)
    }
}

impl std::fmt::Debug for dyn BatchExecution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BatchExecution")
    }
}

/// Strategy for computing expectation values in batch mode.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum ExpValueStrategy {
    /// Delegate directly to the backend's native expectation-value primitive.
    Native,
    /// Estimate expectation values using qubit-wise commutation grouping.
    QubitWiseCommutation(usize),
    /// Estimate expectation values using the classical shadows protocol.
    ClassicalShadows {
        /// Weights for selecting the random measurement basis (X, Y, Z).
        bias: (u8, u8, u8),
        /// Number of measurement rounds.
        samples: usize,
        /// Number of shots per measurement round.
        shots: usize,
    },
}

/// Strategy for computing parameter-shift gradients.
#[derive(Debug, Clone, Copy, Deserialize)]
pub enum GradientStrategy {
    /// Gradient computation is disabled.
    None,
    /// Use the parameter-shift rule to estimate gradients analytically.
    ParameterShiftRule,
    /// Use the backend's native gradient computation.
    Native,
    /// Use the Hadamard-test method to estimate gradients.
    HadamardTest,
}

/// The execution target attached to a [`crate::process::Process`].
#[derive(Debug)]
pub enum QuantumExecution {
    /// Mid-circuit (live) execution: gates are dispatched to the QPU
    /// immediately as they are appended. Measurement results are available
    /// inline, enabling adaptive circuits.
    Live {
        /// The live-mode QPU backend.
        qpu: Box<dyn LiveExecution>,
        /// When `true`, multi-qubit gate instructions are decomposed into
        /// primitive CNOT + single-qubit gates before dispatch.
        decompose: bool,
        /// An optional translation layer for hardware-specific gate vocabularies.
        native_gate_set: Option<Box<dyn NativeGateSet>>,
    },
    /// Deferred (batch) execution: the full compiled circuit is sent at once
    /// after [`Process::execute`](crate::process::Process::execute) is called.
    Batch {
        /// The batch execution backend.
        qpu: Box<dyn BatchExecution>,
        /// An optional translation layer for hardware-specific gate vocabularies.
        /// When `None`, the built-in [`RzRyCX`] identity set is used.
        native_gate_set: Option<Box<dyn NativeGateSet>>,
        /// The gradient computation strategy for variational circuits.
        gradient: GradientStrategy,
        /// The expectation-value estimation strategy.
        exp_value: ExpValueStrategy,
        /// Optional hardware coupling graph expressed as a list of undirected
        /// edges `(i, j)`. When present, the qubit-mapping pass inserts SWAP
        /// gates to satisfy connectivity constraints.
        coupling_graph: Option<Vec<(usize, usize)>>,
        /// When `true`, multi-qubit gate instructions are decomposed into
        /// primitive CNOT + single-qubit gates before the circuit is sent
        /// to the backend.
        decompose: bool,
    },
}

/// Translates Libket IR gates into hardware-specific native gate sequences.
///
/// Implement this trait to map the compiler's intermediate representation into
/// the physical instruction set of a specific QPU or simulator. The default implementation
/// ([`RzRyCX`]) uses a ZYZ Euler decomposition and emits `rz`, `ry`, and `cnot` instructions.
pub trait NativeGateSet {
    /// Translates the 2×2 complex unitary `matrix` acting on `target` into
    /// zero or more native single-qubit gate instructions.
    ///
    /// The matrix is represented as `[[Cf64; 2]; 2]` (row-major). Backends
    /// may recognise special patterns (H, Rz, etc.) and emit optimised
    /// instructions accordingly.
    ///
    /// # Errors
    ///
    /// Returns [`KetError::NativeGateUnsupported`] if the matrix cannot be
    /// translated into the backend's instruction set.
    fn translate(&self, matrix: &Matrix, target: usize) -> Result<Vec<NativeGate>, KetError>;

    /// Translates a CNOT gate into native gate instructions.
    ///
    /// # Errors
    ///
    /// Returns [`KetError::NativeGateUnsupported`] if CNOT is not supported.
    fn cnot(&self, control: usize, target: usize) -> Result<Vec<NativeGate>, KetError>;
}

impl std::fmt::Debug for dyn NativeGateSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("NativeGateSet")
    }
}

/// The {RZ, RY, CNOT} native gateset.
///
/// For single-qubit gates the 2×2 unitary is decomposed via the ZYZ
/// Euler decomposition, emitting `rz` and `ry` native gates
/// (near-zero rotations are dropped). For CNOT
/// it emits a single `cnot` gate. For SWAP it emits three `cnot` gates.
///
/// Used automatically when no [`NativeGateSet`] is configured in
/// [`QuantumExecution::Batch`].
pub type RzRyCX = ();

impl NativeGateSet for RzRyCX {
    fn translate(&self, matrix: &Matrix, target: usize) -> Result<Vec<NativeGate>, KetError> {
        let [[a, b], [c, d]] = matrix;
        let det = (*a * *d - *b * c).arg();

        let theta = 2.0 * c.abs().atan2(a.abs());

        let ang1 = d.arg();
        let ang2 = c.arg();

        let phi = ang1 + ang2 - det;
        let lam = ang1 - ang2;

        let mut gates = Vec::new();

        const EPS: f64 = 1e-10;

        if lam.abs() > EPS {
            gates.push(("rz".to_owned(), vec![lam], vec![target]));
        }
        if theta.abs() > EPS {
            gates.push(("ry".to_owned(), vec![theta], vec![target]));
        }
        if phi.abs() > EPS {
            gates.push(("rz".to_owned(), vec![phi], vec![target]));
        }

        Ok(gates)
    }

    fn cnot(&self, c: usize, t: usize) -> Result<Vec<NativeGate>, KetError> {
        Ok(vec![("cnot".to_owned(), vec![], vec![c, t])])
    }
}
