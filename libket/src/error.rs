// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Libket quantum computing library.
//!
//! All fallible operations in Libket return a [`Result<T>`], where the error
//! variant is a [`KetError`]. Each variant carries a human-readable description
//! of the failure condition that is suitable for end-user display.
//!
//! The enum is `#[repr(i32)]` so that it can be transmitted across FFI
//! boundaries as a plain integer code. Use [`KetError::error_code`] to obtain
//! the discriminant value and [`KetError::from_error_code`] to reconstruct the
//! variant on the receiving side.

/// The exhaustive set of errors that Libket operations can produce.
///
/// Every variant has a stable, non-zero integer discriminant (with the sole
/// exception of [`KetError::Success`], which is always `0`) so that the enum
/// can be used as a C-compatible status code.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[repr(i32)]
pub enum KetError {
    /// No error occurred. This variant is only used as a sentinel value in FFI
    /// contexts where an integer `0` signals success.
    #[error("Operation completed successfully.")]
    Success = 0,

    /// A qubit was specified as a control qubit more than once in the same gate
    /// instruction. Each physical qubit may only appear once in the control set.
    ///
    /// Returned by [`crate::ir::gate::GateInstruction::control`] (and transitively
    /// by [`crate::ir::block::BasicBlock::control`]) when
    /// `control_qubits` contains a duplicate entry or overlaps with the existing
    /// control set of the instruction.
    ///
    /// # Example
    /// Calling `gate.control(&[q0, q0])` would trigger this error.
    #[error(
        "Duplicate control qubit: the same qubit index appears more than once in the \
         control set of a single gate instruction. Each qubit may be listed at most once."
    )]
    DuplicateControlQubit,

    /// A qubit was used as both a control and the target of the same gate
    /// instruction, which is physically undefined.
    ///
    /// Returned by [`crate::ir::gate::GateInstruction::control`] (and transitively
    /// by [`crate::ir::block::BasicBlock::control`]) when
    /// one of the provided `control_qubits` equals `self.target`.
    ///
    /// # Example
    /// Calling `gate.control(&[gate.target])` would trigger this error.
    #[error(
        "Control-target qubit conflict: a qubit index appears in both the control set \
         and as the target of the same gate. A qubit cannot control its own gate."
    )]
    ControlTargetConflict,

    /// An attempt was made to add control qubits to a parameterized gate.
    ///
    /// Adding control qubits to gates with `Param::Ref` parameters is not supported.
    #[error(
        "Control parameterized gate: adding control qubits to parameterized gates is not supported."
    )]
    ControlParameterizedGate,

    /// The process has already allocated the maximum number of qubits permitted
    /// by the QPU configuration and no additional qubits can be created.
    ///
    /// Returned by [`crate::process::Process::alloc`] when
    /// `qubit_counter == qpu_config.num_qubits`.
    #[error(
        "Qubit allocation limit reached: the process has already allocated all \
         qubits allowed by the QPU configuration."
    )]
    QubitLimitExceeded,

    /// A gate or measurement referenced a qubit index that exceeds the number
    /// of qubits allocated so far by the current process.
    ///
    /// Returned by [`crate::process::Process::append_block`]
    /// when the maximum qubit index
    /// referenced inside the block is ≥ `qubit_counter`. Call
    /// [`crate::process::Process::alloc`] to allocate the required qubits
    /// before appending the block.
    #[error(
        "Qubit index out of range: the gate block references a qubit that has not \
         been allocated. Call alloc() for each qubit before using it in a block."
    )]
    QubitIndexOutOfRange,

    /// An attempt was made to allocate a qubit on a process that has already
    /// been executed and is in the `Terminated` state.
    ///
    /// Returned by [`crate::process::Process::alloc`]
    /// when `process.state == ProcessState::Terminated`. Create a new
    /// [`crate::process::Process`] to run additional circuits.
    #[error(
        "Process is terminated: qubit allocation is not allowed after a process has \
         been executed. Create a new Process to run additional circuits."
    )]
    ProcessTerminated,

    /// An attempt was made to append a gate block to a process that is not
    /// in the `AcceptingGate` state.
    ///
    /// Returned by [`crate::process::Process::append_block`]
    /// when the process state is anything other than
    /// [`crate::process::ProcessState::AcceptingGate`], for example,
    /// after `sample` or `exp_value` has already been registered in
    /// batch mode (which advancesthe state to
    /// [`crate::process::ProcessState::ReadyToExecute`] or
    /// [`crate::process::ProcessState::AcceptingHamiltonian`]).
    #[error(
        "Cannot append gates: the process is not in the AcceptingGate state. \
         Gates cannot be added after a sample or exp_value request has been \
         registered in batch mode."
    )]
    GateAppendForbidden,

    /// A single-shot mid-circuit measurement was requested but the process is
    /// not configured for live execution.
    ///
    /// Returned by [`crate::process::Process::measure`]
    /// when `qpu_config.quantum_execution` is
    /// `Some(QuantumExecution::Batch { .. })` or `None`. Measurements that
    /// collapse qubit state and return a result inline are only available in
    /// [`crate::execution::QuantumExecution::Live`] mode.
    /// Use `sample` for batch-mode histograms.
    #[error(
        "Single-shot measurement is unavailable outside of Live execution mode. \
         Use 'sample' to collect a measurement histogram."
    )]
    MeasurementUnavailableInBatch,

    /// A `sample` request was made but cannot be registered in the current
    /// process state.
    ///
    /// Returned by [`crate::process::Process::sample`] in batch mode when
    /// the process state is not [`crate::process::ProcessState::AcceptingGate`]
    /// (i.e., a `sample` or `exp_value` request has already
    /// been registered), or when no execution backend is configured
    /// (`qpu_config.quantum_execution == None`). Only one `sample` call is
    /// permitted per batch execution.
    #[error(
        "Sampling unavailable: a sample request can only be registered once per batch \
         execution, and only while the process is in the AcceptingGate state. \
         Ensure an execution backend is configured and no conflicting request exists."
    )]
    SamplingUnavailable,

    /// An `exp_value` request was made but cannot be registered in the current
    /// process state.
    ///
    /// Returned by [`crate::process::Process::exp_value`] in batch mode when the
    /// process state is not [`crate::process::ProcessState::AcceptingGate`] or
    /// [`crate::process::ProcessState::AcceptingHamiltonian`], for example, after the
    /// process has transitioned to `ReadyToExecute` because a `sample` request
    /// or a gradient-enabled `exp_value` was already registered, or when no
    /// execution backend is configured.
    #[error(
        "Expectation-value unavailable: exp_value can only be registered while the process \
         is in the AcceptingGate or AcceptingHamiltonian state. It cannot be combined with \
         a pending sample request, and requires an execution backend to be configured."
    )]
    ExpectationValueUnavailable,

    /// A quantum state dump was requested but the process is not configured
    /// for live execution.
    ///
    /// Returned by [`crate::process::Process::dump`]
    /// when `qpu_config.quantum_execution` is `Some(QuantumExecution::Batch { .. })`
    /// or `None`. Full state-vector inspection is only available in
    /// [`crate::execution::QuantumExecution::Live`] mode.
    #[error(
        "State dump is unavailable outside of Live execution mode. \
         Full state-vector inspection via 'dump' requires Live execution."
    )]
    DumpUnavailableInBatch,

    #[error("unknown macro")]
    UnknownMacro,

    /// `execute` was called on a process that has no pending measurement or
    /// expectation-value request.
    ///
    /// Returned by [`crate::process::Process::execute`] when the process state
    /// is [`crate::process::ProcessState::AcceptingGate`],
    /// meaning neither `sample` nor `exp_value` has been called since the
    /// last execution. Register at least one `sample` or `exp_value` request
    /// before calling `execute` in batch mode.
    #[error(
        "No pending measurement: 'execute' requires at least one 'sample' or 'exp_value' \
         call before it can be invoked. The process is still in the AcceptingGate state."
    )]
    NoPendingMeasurement,

    /// `execute` (or an internal `compile`) was called on a process configured
    /// for live execution.
    ///
    /// Returned by [`crate::process::Process::execute`] when
    /// `qpu_config.quantum_execution` is `Some(QuantumExecution::Live { .. })`.
    /// In live mode every gate is dispatched to the backend immediately as it
    /// is appended; there is no deferred compilation step.
    #[error(
        "Explicit execution is not supported in Live mode: gates and measurements are \
         dispatched to the backend immediately as they are appended. \
         Use Batch execution to compile and run the circuit via 'execute'."
    )]
    ExplicitExecuteInLiveMode,

    /// The quantum execution backend reported an unrecoverable failure.
    ///
    /// This is a generic error code that backend implementations
    /// should return for hardware faults, decoherence events, calibration errors,
    /// or simulator assertion failures.
    /// It surfaces in Rust through [`KetError::from_error_code`] after the C
    /// callback returns a non-zero status. Consult the backend's own error
    /// channel for a more detailed diagnostic.
    #[error(
        "Execution failed: the quantum backend reported an unrecoverable error. \
         Consult the backend's diagnostic output for details."
    )]
    ExecutionFailed,

    /// The requested number of measurement shots is out of range for the
    /// active backend.
    ///
    /// Backend implementations should return this error code
    /// from their `sample` callback when `shots` is zero, exceeds an
    /// implementation-defined maximum, or otherwise falls outside the
    /// backend's accepted range. It reaches Rust via [`KetError::from_error_code`].
    #[error(
        "Invalid shot count: the number of measurement shots is out of range for \
         the active backend. Check the backend's accepted minimum and maximum."
    )]
    ShotCountInvalid,

    /// A gate produced by the [`crate::execution::NativeGateSet`] translation
    /// is not supported by the backend.
    ///
    /// Custom [`crate::execution::NativeGateSet`] implementations should
    /// return this error from `translate`, `cnot`, or `swap` when the
    /// requested operation cannot be expressed in the hardware's instruction
    /// set. It is also the recommended return code for batch-execution C
    /// callbacks that receive an unrecognized gate name.
    #[error(
        "Unsupported native gate: the gate produced by the NativeGateSet translation \
         is not recognized by the backend. Ensure the NativeGateSet is correctly \
         configured for the target hardware."
    )]
    NativeGateUnsupported,

    /// A gate instruction from the Libket IR was received by a backend that
    /// does not accept IR-level gates.
    ///
    /// Returned as the default implementation of [`crate::execution::BatchExecution::sample`]
    /// and [`crate::execution::BatchExecution::exp_value`] when the backend only
    /// supports native gates. Implementors that accept IR-level gates should
    /// override those methods instead of propagating this error.
    #[error("Unsupported gate: gates from the Libket IR is not supported.")]
    GateUnsupported,

    /// The backend does not support native expectation-value computation but
    /// [`crate::execution::ExpValueStrategy::Native`] was selected.
    ///
    /// Backend callbacks should return this error code from their
    /// `exp_value` callback when the hardware or simulator cannot compute
    /// expectation values directly. Switch to
    /// [`crate::execution::ExpValueStrategy::QubitWiseCommutation`] or
    /// [`crate::execution::ExpValueStrategy::ClassicalShadows`] to estimate
    /// expectation values through repeated sampling instead.
    #[error(
        "Native expectation-value computation is not supported by this backend. \
         Use ExpValueStrategy::QubitWiseCommutation or ExpValueStrategy::ClassicalShadows \
         to estimate expectation values via sampling."
    )]
    ExpValueNotSupported,

    /// JSON serialization or deserialization failed in the C API layer.
    ///
    /// Emitted when `serde_json` fails to serialize a Rust value to a
    /// C string, or when a JSON string received from a C callback cannot be
    /// parsed into the expected Rust type. Verify that the JSON payload
    /// conforms to the schema expected by the called function.
    #[error(
        "Serialization/deserialization error: a JSON payload could not be encoded \
         or decoded in the C API layer. Ensure the JSON string is valid and matches \
         the expected schema."
    )]
    SerdeError,

    /// The classical shadows bias tuple `(X, Y, Z)` has all three weights set
    /// to zero, making it impossible to sample a measurement basis.
    ///
    /// Returned by [`crate::process::Process::execute`] (via
    /// `execute_classical_shadows`) when
    /// [`crate::execution::ExpValueStrategy::ClassicalShadows`] is configured
    /// with `bias = (0, 0, 0)`. At least one of the three weights must be
    /// non-zero.
    #[error(
        "Invalid classical shadows bias: all three Pauli basis weights (X, Y, Z) are zero. \
         At least one weight must be non-zero to sample a measurement basis."
    )]
    InvalidBias,

    /// The backend does not support native gradient computation but
    /// [`crate::execution::GradientStrategy::Native`] was selected.
    ///
    /// Returned as the default implementation of
    /// [`crate::execution::BatchExecution::gradient`]. Switch to
    /// [`crate::execution::GradientStrategy::ParameterShiftRule`] to estimate
    /// gradients analytically via repeated circuit evaluations.
    #[error(
        "Native Gradient Unsupported: The simulator does not support gradient calculation. \
         Use ParameterShiftRule."
    )]
    NativeGradientUnsuported,

    /// An unrecognized error code was received from a C callback.
    ///
    /// Returned by [`KetError::from_error_code`] as a catch-all when the
    /// integer code does not match any known variant. This typically means a
    /// C-side backend returned a non-standard status code. Treat it as an
    /// unrecoverable backend error.
    #[error(
        "Unknown error: an unrecognized error code was returned by the backend. \
         This may indicate a version mismatch or a non-standard status code."
    )]
    UnknownError,

    /// An error occurred while parsing an `OpenQASM` program.
    #[error("OpenQASM parse error")]
    OpenQasmParseError,

    /// A semantic error occurred while processing an `OpenQASM` program.
    #[error("OpenQASM semantic error")]
    OpenQasmSemanticError,
}

impl KetError {
    /// Returns the integer discriminant of this error variant.
    ///
    /// The returned value is stable and suitable for FFI use. `Success` always
    /// returns `0`; all other variants return a positive non-zero value.
    #[must_use]
    pub const fn error_code(&self) -> i32 {
        match self {
            Self::Success => 0,
            Self::DuplicateControlQubit => 1,
            Self::ControlTargetConflict => 2,
            Self::QubitLimitExceeded => 3,
            Self::QubitIndexOutOfRange => 4,
            Self::ProcessTerminated => 5,
            Self::GateAppendForbidden => 6,
            Self::MeasurementUnavailableInBatch => 7,
            Self::SamplingUnavailable => 8,
            Self::ExpectationValueUnavailable => 9,
            Self::DumpUnavailableInBatch => 10,
            Self::UnknownMacro => 24,
            Self::NoPendingMeasurement => 11,
            Self::ExplicitExecuteInLiveMode => 12,
            Self::ExecutionFailed => 13,
            Self::ShotCountInvalid => 14,
            Self::NativeGateUnsupported => 15,
            Self::ExpValueNotSupported => 16,
            Self::SerdeError => 17,
            Self::InvalidBias => 18,
            Self::ControlParameterizedGate => 19,
            Self::GateUnsupported => 20,
            Self::NativeGradientUnsuported => 21,
            Self::OpenQasmParseError => 22,
            Self::OpenQasmSemanticError => 23,
            Self::UnknownError => i32::MAX,
        }
    }

    /// Reconstructs a `KetError` from its integer discriminant.
    ///
    /// Variants that carry additional context (such as qubit indices) are
    /// reconstructed with sentinel values (`0` for `usize` fields) because the
    /// context is not preserved in the raw integer code.
    ///
    /// # Safety
    /// The caller must guarantee that `error_code` was produced by a prior call
    /// to [`KetError::error_code`]. Passing an arbitrary integer is undefined
    /// behaviour because this function uses [`std::mem::transmute`] internally
    /// for the simple variants and pattern-matches for the contextual ones.
    #[must_use]
    pub fn from_error_code(error_code: i32) -> Self {
        match error_code {
            0 => Self::Success,
            1 => Self::DuplicateControlQubit,
            2 => Self::ControlTargetConflict,
            3 => Self::QubitLimitExceeded,
            4 => Self::QubitIndexOutOfRange,
            5 => Self::ProcessTerminated,
            6 => Self::GateAppendForbidden,
            7 => Self::MeasurementUnavailableInBatch,
            8 => Self::SamplingUnavailable,
            9 => Self::ExpectationValueUnavailable,
            10 => Self::DumpUnavailableInBatch,
            24 => Self::UnknownMacro,
            11 => Self::NoPendingMeasurement,
            12 => Self::ExplicitExecuteInLiveMode,
            13 => Self::ExecutionFailed,
            14 => Self::ShotCountInvalid,
            15 => Self::NativeGateUnsupported,
            16 => Self::ExpValueNotSupported,
            17 => Self::SerdeError,
            18 => Self::InvalidBias,
            19 => Self::ControlParameterizedGate,
            20 => Self::GateUnsupported,
            21 => Self::NativeGradientUnsuported,
            22 => Self::OpenQasmParseError,
            23 => Self::OpenQasmSemanticError,
            _ => Self::UnknownError,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KetError;

    #[test]
    fn success_is_zero() {
        assert_eq!(KetError::Success.error_code(), 0);
    }

    #[test]
    fn error_codes_are_distinct() {
        let variants = vec![
            KetError::Success,
            KetError::DuplicateControlQubit,
            KetError::ControlTargetConflict,
            KetError::QubitLimitExceeded,
            KetError::QubitIndexOutOfRange,
            KetError::ProcessTerminated,
            KetError::GateAppendForbidden,
            KetError::MeasurementUnavailableInBatch,
            KetError::SamplingUnavailable,
            KetError::ExpectationValueUnavailable,
            KetError::DumpUnavailableInBatch,
            KetError::UnknownMacro,
            KetError::NoPendingMeasurement,
            KetError::ExplicitExecuteInLiveMode,
            KetError::ExecutionFailed,
            KetError::ShotCountInvalid,
            KetError::NativeGateUnsupported,
            KetError::ExpValueNotSupported,
            KetError::SerdeError,
            KetError::InvalidBias,
            KetError::ControlParameterizedGate,
            KetError::GateUnsupported,
            KetError::NativeGradientUnsuported,
            KetError::OpenQasmParseError,
            KetError::OpenQasmSemanticError,
            KetError::UnknownError,
        ];
        let mut codes: Vec<i32> = variants.iter().map(|e| e.error_code()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(
            codes.len(),
            variants.len(),
            "Duplicate error codes detected"
        );
    }

    #[test]
    fn roundtrip_simple_variants() {
        let variants = vec![
            KetError::Success,
            KetError::DuplicateControlQubit,
            KetError::ProcessTerminated,
            KetError::GateAppendForbidden,
            KetError::MeasurementUnavailableInBatch,
            KetError::SamplingUnavailable,
            KetError::ExpectationValueUnavailable,
            KetError::DumpUnavailableInBatch,
            KetError::UnknownMacro,
            KetError::NoPendingMeasurement,
            KetError::ExplicitExecuteInLiveMode,
            KetError::ExpValueNotSupported,
        ];
        for v in &variants {
            let code = v.error_code();
            let reconstructed = KetError::from_error_code(code);
            assert_eq!(reconstructed.error_code(), code);
        }
    }
}
