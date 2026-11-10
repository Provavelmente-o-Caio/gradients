// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Quantum process: the top-level unit of compilation and execution.
//!
//! A [`Process`] holds a [`BasicBlock`] of gate instructions together with
//! measurement requests and drives them through compilation (qubit mapping,
//! gate decomposition) and execution on a [`QuantumExecution`] backend.
//!
//! ## State machine
//!
//! A `Process` advances through the following states:
//!
//! ```text
//! AcceptingGate
//!   ├── append_block / alloc / param      (stays AcceptingGate)
//!   ├── exp_value (batch, no gradient)  → AcceptingHamiltonian
//!   ├── exp_value (batch, with gradient)→ ReadyToExecute
//!   └── sample (batch)                  → ReadyToExecute
//! AcceptingHamiltonian
//!   ├── exp_value (batch, no gradient)     (stays AcceptingHamiltonian)
//!   └── execute                          → Terminated
//! ReadyToExecute
//!   └── execute                          → Terminated
//! Terminated
//!   └── (no further mutations allowed)
//! ```
//!
//! In **live** mode ([`QuantumExecution::Live`]) gates are dispatched to the
//! QPU backend immediately; `execute` is not used. In **batch** mode
//! ([`QuantumExecution::Batch`]) the full circuit is compiled and sent at once
//! when [`Process::execute`] is called.

mod classical_shadow;
mod qwc;

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::error::KetError;
use crate::execution::{
    BatchExecution, DumpData, ExpValueStrategy, GradientStrategy, NativeGate, NativeGateSet,
    QuantumExecution, SampleData,
};
use crate::ir::gate::GateInstruction;
use crate::ir::{
    gate::{DecomposedGate, Param, QuantumGate},
    hamiltonian::Hamiltonian,
};
use crate::mapping;
use crate::matrix::Matrix;
use crate::process::classical_shadow::execute_classical_shadows;
use crate::process::qwc::execute_qwc_exp_value_live;

/// Configuration of the quantum processing unit (QPU) available to this process.
#[derive(Debug)]
pub struct QPUConfig {
    /// Maximum number of logical qubits that may be allocated by this process.
    /// Attempting to allocate beyond this limit returns
    /// [`KetError::QubitLimitExceeded`].
    pub num_qubits: usize,
    /// The execution backend (live or batch) used to run circuits.
    /// When `None`, gate sequences can be built but execution will fail.
    pub quantum_execution: Option<QuantumExecution>,
}

/// Internal state machine for a [`Process`].
///
/// The state advances monotonically; once `Terminated` no further mutations
/// are possible. Transitions are triggered by [`Process::sample`],
/// [`Process::exp_value`], and [`Process::execute`].
#[derive(Debug, Default, Clone, Copy)]
pub enum ProcessState {
    /// The process is accepting gate instructions via [`Process::append_block`].
    /// This is the initial state.
    #[default]
    AcceptingGate,
    /// At least one Hamiltonian expectation-value request has been registered
    /// via [`Process::exp_value`] (batch mode, no gradient). Additional
    /// Hamiltonians may still be added.
    AcceptingHamiltonian,
    /// All measurements are registered and the process is ready to execute.
    /// No further gates or Hamiltonian registrations are accepted.
    ReadyToExecute,
    /// The process has been executed and is in a terminal state. No further
    /// instructions can be appended or execution triggered.
    Terminated,
}

/// A quantum process: the top-level compilation and execution unit.
///
/// Create a [`Process`] with [`Process::new`], allocate qubits with
/// [`Process::alloc`], append gate sequences with [`Process::append_block`],
/// and trigger execution with [`Process::execute`] (batch mode) or read
/// results immediately after each gate/measurement in live mode.
///
/// Each process follows a linear [`ProcessState`] machine and can be executed
/// at most once. Results are cached and retrieved via [`Process::read_sample`],
/// [`Process::read_exp_value`], and [`Process::read_gradient`].
#[derive(Debug)]
pub struct Process {
    /// The accumulated gate sequence for this process.
    main_block: crate::ir::block::Block,
    /// QPU configuration (qubit count and execution backend).
    qpu_config: QPUConfig,
    /// Number of qubits that have been allocated so far.
    qubit_counter: usize,
    /// Logical qubit indices and shot count for a pending `sample` request.
    qubits_to_sample: Option<(Vec<usize>, usize)>,
    /// Cached sample result after execution.
    sample: Option<SampleData>,
    /// List of Hamiltonians registered for expectation-value computation,
    /// in registration order.
    hamiltonian_list: Vec<Hamiltonian>,
    /// Cached expectation-value results after execution, one per Hamiltonian.
    exp_value: Option<Vec<f64>>,
    /// Concrete parameter values for variational circuits, in registration order.
    parameters: Vec<f64>,
    /// Cached gradient values after execution (parameter-shift rule),
    /// one per registered parameter.
    grad: Option<Vec<f64>>,
    /// Current state of the process state machine.
    state: ProcessState,
    /// Angle threshold (in radians) below which a rotation is treated as the
    /// identity and dropped during gate simplification.
    epsilon: f64,
    /// Index into `main_block.gates` of the next gate to dispatch in live mode.
    live_pc: usize,
}

/// A borrowed view of a compiled gate sequence, dispatching to either the
/// Libket IR path or the pre-translated native-gate path.
///
/// This enum is threaded through `execute_sample`, `execute_exp_value`, and
/// the QWC/classical-shadows helpers so that the same logic works regardless
/// of whether a [`NativeGateSet`] is configured.
pub(super) enum GateList<'a> {
    /// IR-level gates: passed directly to [`BatchExecution::sample`] /
    /// [`BatchExecution::exp_value`] / [`BatchExecution::gradient`].
    Ir { gates: &'a [GateInstruction] },
    /// Pre-translated native gates: passed to [`BatchExecution::sample_native`]
    /// / [`BatchExecution::exp_value_native`].
    ///
    /// The `native_gate_set` reference is kept so that basis-change gates
    /// appended by QWC measurement circuits can also be translated on the fly.
    Native {
        gates: &'a [NativeGate],
        native_gate_set: &'a dyn NativeGateSet,
    },
}

/// An owned version of [`GateList`] produced by measurement-circuit
/// construction (e.g., QWC basis-change appending).
pub(super) enum GateListOwned {
    /// Owned IR-level gate sequence.
    Ir { gates: Vec<GateInstruction> },
    /// Owned native gate sequence together with a reference to the gate-set
    /// translator (needed for further basis-change translation).
    Native { gates: Vec<NativeGate> },
}

impl Process {
    /// Creates a new [`Process`] from the given [`QPUConfig`].
    ///
    /// The process starts in the [`ProcessState::AcceptingGate`] state with an
    /// empty gate sequence, no allocated qubits, and an angle-cancellation
    /// threshold (`epsilon`) of `1e-10` radians.
    #[must_use]
    pub fn new(qpu_config: QPUConfig) -> Self {
        Self {
            main_block: crate::ir::block::Block::new(),
            qubit_counter: 0,
            qpu_config,
            qubits_to_sample: None,
            sample: None,
            hamiltonian_list: Vec::new(),
            exp_value: None,
            parameters: Vec::new(),
            grad: None,
            state: ProcessState::default(),
            epsilon: 1e-10,
            live_pc: 0,
        }
    }

    /// Allocates the next available logical qubit and returns its index.
    ///
    /// # Errors
    /// - [`KetError::ProcessTerminated`]: the process has already been executed.
    /// - [`KetError::QubitLimitExceeded`]: all qubits permitted by the QPU
    ///   configuration have been allocated.
    pub const fn alloc(&mut self) -> Result<usize, KetError> {
        if matches!(self.state, ProcessState::Terminated) {
            Err(KetError::ProcessTerminated)
        } else if self.qubit_counter < self.qpu_config.num_qubits {
            let index = self.qubit_counter;
            self.qubit_counter += 1;
            Ok(index)
        } else {
            Err(KetError::QubitLimitExceeded)
        }
    }

    /// Appends a [`BasicBlock`] of gate instructions to this process.
    ///
    /// If required, gate decomposition is performed in parallel before
    /// the block is merged into the main circuit. In live mode every
    /// gate is sent to the QPU backend immediately.
    ///
    /// # Errors
    /// - [`KetError::GateAppendForbidden`]: the process is not in the
    ///   gate-accepting state.
    /// - [`KetError::QubitIndexOutOfRange`]: the block references a qubit
    ///   that has not yet been allocated.
    pub fn append_block(&mut self, block: crate::ir::block::Block) -> Result<(), KetError> {
        if !matches!(self.state, ProcessState::AcceptingGate) {
            return Err(KetError::GateAppendForbidden);
        }

        if block
            .max_qubit_index()
            .is_some_and(|index| index >= self.qubit_counter)
        {
            return Err(KetError::QubitIndexOutOfRange);
        }

        self.main_block.append_block_with_epsilon_and_min_index(
            block,
            Some(self.epsilon),
            self.live_pc,
        );
        Ok(())
    }

    #[must_use]
    fn need_to_decompose(&self) -> bool {
        matches!(
            self.qpu_config.quantum_execution,
            Some(
                QuantumExecution::Batch {
                    coupling_graph: Some(_),
                    ..
                } | QuantumExecution::Batch {
                    native_gate_set: Some(_),
                    ..
                } | QuantumExecution::Batch {
                    decompose: true,
                    ..
                } | QuantumExecution::Live {
                    decompose: true,
                    ..
                }
            )
        )
    }

    #[must_use]
    /// Returns a reference to the accumulated gate sequence of this process.
    ///
    /// Useful for introspection or serialization before execution.
    pub fn main_block(&self) -> &crate::ir::block::Block {
        &self.main_block
    }

    /// Flattens the main block without simplifying gates across the live
    /// barrier (`live_pc`), so the result matches what the backend executed.
    ///
    /// # Errors
    /// Propagates any [`KetError`] raised while instantiating template blocks.
    pub fn flat_main_block(&self) -> Result<crate::ir::block::BasicBlock, KetError> {
        self.main_block.flatten_with_min_index(self.live_pc)
    }

    /// Performs a single-shot mid-circuit measurement of `qubits`.
    ///
    /// # Errors
    /// - [`KetError::MeasurementUnavailableInBatch`]: the process is not
    ///   running in live execution mode.
    pub fn measure(&mut self, qubits: &[usize]) -> Result<u64, KetError> {
        self.live_execute()?;

        if let Some(QuantumExecution::Live { qpu, .. }) = &mut self.qpu_config.quantum_execution {
            Ok(qpu.measure(qubits)?)
        } else {
            Err(KetError::MeasurementUnavailableInBatch)
        }
    }

    /// Dumps the quantum state of `qubits` (live mode only).
    ///
    /// # Errors
    /// - [`KetError::DumpUnavailableInBatch`]: the process is not running in
    ///   live execution mode.
    pub fn dump(&mut self, qubits: &[usize]) -> Result<DumpData, KetError> {
        self.live_execute()?;

        if let Some(QuantumExecution::Live { qpu, .. }) = &mut self.qpu_config.quantum_execution {
            Ok(qpu.dump(qubits)?)
        } else {
            Err(KetError::DumpUnavailableInBatch)
        }
    }

    /// Drains any newly appended gates to the live backend.
    ///
    /// Iterates from `live_pc` (the position of the last dispatched gate) to
    /// the end of `main_block.gates`, sending each gate to the QPU via
    /// [`crate::execution::LiveExecution::compute_gate`]. A no-op in batch mode.
    ///
    /// The main block is flattened without simplifying across `live_pc`, and
    /// then replaced by that flat block. This keeps `live_pc` a valid index
    /// into the main block, so gates appended later can never cancel or merge
    /// with gates the backend has already executed.
    ///
    /// # Errors
    /// Propagates any [`KetError`] returned by the underlying backend.
    fn live_execute(&mut self) -> Result<(), KetError> {
        if !matches!(
            self.qpu_config.quantum_execution,
            Some(QuantumExecution::Live { .. })
        ) {
            return Ok(());
        }

        let mut flat_block = self.flat_main_block()?;
        self.main_block = crate::ir::block::Block {
            elements: vec![crate::ir::block::BlockElement::Basic(flat_block.clone())],
        };

        if self.need_to_decompose() {
            flat_block.gates.iter_mut().for_each(|gate| {
                gate.decompose((self.qubit_counter, self.qpu_config.num_qubits));
            });
        }

        if let Some(QuantumExecution::Live {
            qpu,
            native_gate_set,
            ..
        }) = &mut self.qpu_config.quantum_execution
        {
            while self.live_pc < flat_block.gates.len() {
                let gate = &flat_block.gates[self.live_pc];

                if let Some(native_gate_set) = native_gate_set {
                    let translated = if let Some(decomposed) = &gate.decomposed {
                        Self::translate_circuit(
                            &decomposed.iter().map(Into::into).collect::<Vec<_>>(),
                            native_gate_set.as_ref(),
                        )
                    } else {
                        Self::translate_circuit(
                            std::slice::from_ref(gate),
                            native_gate_set.as_ref(),
                        )
                    }?;

                    qpu.compute_native_gates(&translated)?;
                } else if let Some(decomposed) = &gate.decomposed {
                    for gate in decomposed {
                        qpu.compute_gate(&gate.into())?;
                    }
                } else {
                    qpu.compute_gate(gate)?;
                }
                self.live_pc += 1;
            }
        }
        Ok(())
    }

    /// Requests a measurement sample of `qubits` over `shots` repetitions.
    ///
    /// **Live mode:** The sample is taken immediately and returned as
    /// `Ok(Some(SampleData))`.
    ///
    /// **Batch mode:** The request is recorded and `Ok(None)` is returned.
    /// The process transitions to `ProcessState::ReadyToExecute`. Call
    /// [`Process::execute`] to dispatch the circuit and then retrieve the
    /// result with [`Process::read_sample`].
    ///
    /// # Errors
    /// - [`KetError::SamplingUnavailable`]: a conflicting measurement has
    ///   already been registered, or no execution backend is configured.
    pub fn sample(
        &mut self,
        qubits: &[usize],
        shots: usize,
    ) -> Result<Option<SampleData>, KetError> {
        self.live_execute()?;

        match &mut self.qpu_config.quantum_execution {
            Some(QuantumExecution::Live { qpu, .. }) => Ok(Some(qpu.sample(qubits, shots)?)),
            Some(QuantumExecution::Batch { .. }) => {
                if matches!(self.state, ProcessState::AcceptingGate) {
                    self.state = ProcessState::ReadyToExecute;
                    self.qubits_to_sample = Some((qubits.to_owned(), shots));
                    Ok(None)
                } else {
                    Err(KetError::SamplingUnavailable)
                }
            }
            None => Err(KetError::SamplingUnavailable),
        }
    }

    /// Returns a reference to the cached [`SampleData`] from the most recent
    /// execution, or `None` if no sample result is available yet.
    #[must_use]
    pub fn read_sample(&self) -> Option<&SampleData> {
        self.sample.as_ref()
    }

    /// Requests the expectation value `⟨ψ|H|ψ⟩` of `hamiltonian`.
    ///
    /// **Live mode:** The expectation value is computed immediately and
    /// returned as `Ok(Some(f64))`.
    ///
    /// **Batch mode (no gradient):** The Hamiltonian is appended to the
    /// internal list. The process stays in (or transitions to)
    /// [`ProcessState::AcceptingHamiltonian`], allowing additional
    /// Hamiltonians to be registered before execution.
    ///
    /// **Batch mode (with gradient):** The Hamiltonian is registered and the
    /// process immediately transitions to [`ProcessState::ReadyToExecute`]
    /// because the parameter-shift rule requires exactly one observable.
    ///
    /// In both batch cases `Ok(None)` is returned; call [`Process::execute`]
    /// followed by [`Process::read_exp_value`] to retrieve the results.
    ///
    /// # Errors
    /// - [`KetError::ExpectationValueUnavailable`]: the process is in an
    ///   incompatible state (e.g., already terminated or a sample request has
    ///   been registered), or no execution backend is configured.
    pub fn exp_value(&mut self, hamiltonian: Hamiltonian) -> Result<Option<f64>, KetError> {
        self.live_execute()?;

        match &mut self.qpu_config.quantum_execution {
            Some(QuantumExecution::Live { qpu, .. }) => Ok(Some(execute_qwc_exp_value_live(
                qpu.as_mut(),
                &hamiltonian,
            )?)),
            Some(QuantumExecution::Batch { gradient, .. }) => {
                if matches!(
                    self.state,
                    ProcessState::AcceptingGate | ProcessState::AcceptingHamiltonian
                ) {
                    self.state = if matches!(gradient, GradientStrategy::None) {
                        ProcessState::AcceptingHamiltonian
                    } else {
                        ProcessState::ReadyToExecute
                    };

                    self.hamiltonian_list.push(hamiltonian);
                    Ok(None)
                } else {
                    Err(KetError::ExpectationValueUnavailable)
                }
            }
            None => Err(KetError::ExpectationValueUnavailable),
        }
    }

    /// Returns a slice of the cached expectation-value results from the most
    /// recent execution, or `None` if no results are available yet.
    ///
    /// The slice contains one value per [`Hamiltonian`] registered via
    /// [`Process::exp_value`], in registration order.
    #[must_use]
    pub fn read_exp_value(&self) -> Option<&[f64]> {
        self.exp_value.as_deref()
    }

    /// Returns a slice of the cached gradient values from the most recent
    /// execution, or `None` if gradient computation was not enabled or has
    /// not yet been performed.
    ///
    /// The slice contains one value per registered parameter (in registration
    /// order). Each element is the partial derivative of the expectation value
    /// with respect to that parameter.
    #[must_use]
    pub fn read_gradient(&self) -> Option<&[f64]> {
        self.grad.as_deref()
    }

    /// Registers a new differentiable parameter with initial value `param`
    /// and returns a [`Param::Ref`] token for embedding in gate rotation angles.
    pub fn param(&mut self, param: f64) -> Param {
        let index = self.parameters.len();
        self.parameters.push(param);

        Param::Ref {
            index,
            multiplier: 1.0,
            value: param,
        }
    }

    /// Compiles and executes the accumulated circuit (batch mode only).
    ///
    /// # Errors
    /// - [`KetError::NoPendingMeasurement`]: no `sample` or `exp_value`
    ///   request has been registered before calling `execute`.
    /// - [`KetError::ExplicitExecuteInLiveMode`]: the process is in live
    ///   execution mode, where gates are dispatched immediately.
    pub fn execute(&mut self) -> Result<(), KetError> {
        if matches!(self.state, ProcessState::Terminated) {
            return Ok(());
        }

        if !matches!(
            self.state,
            ProcessState::AcceptingHamiltonian | ProcessState::ReadyToExecute
        ) {
            return Err(KetError::NoPendingMeasurement);
        }

        let mut flat_block = self.main_block.flatten()?;
        if self.need_to_decompose() {
            flat_block.gates.iter_mut().for_each(|gate| {
                gate.decompose((self.qubit_counter, self.qpu_config.num_qubits));
            });
        }

        let (final_gates, mapped_hamiltonian_list, mapped_qubits_to_sample) =
            if let Some(QuantumExecution::Batch {
                coupling_graph: Some(coupling_graph),
                ..
            }) = self.qpu_config.quantum_execution.as_ref()
            {
                let (gates, hamiltonian, qubits_to_sample) = mapping::map_circuit(
                    &flat_block,
                    &self.hamiltonian_list,
                    &self.qubits_to_sample.clone().unwrap_or((vec![], 0)).0,
                    coupling_graph,
                    self.qpu_config.num_qubits,
                )?;
                let qubits_to_sample = self
                    .qubits_to_sample
                    .clone()
                    .map(|(_, shots)| (qubits_to_sample, shots));

                (gates, hamiltonian, qubits_to_sample)
            } else {
                (
                    flat_block.gates,
                    std::mem::take(&mut self.hamiltonian_list),
                    self.qubits_to_sample.take(),
                )
            };

        let mut quantum_execution = self.qpu_config.quantum_execution.take();

        if let Some(QuantumExecution::Batch {
            qpu,
            native_gate_set,
            gradient,
            exp_value,
            ..
        }) = &mut quantum_execution
        {
            if let Some((qubits_to_sample, shots)) = &mapped_qubits_to_sample {
                if let Some(native_gate_set) = native_gate_set {
                    let gates = GateList::Native {
                        gates: &Self::translate_circuit(&final_gates, native_gate_set.as_ref())?,
                        native_gate_set: native_gate_set.as_ref(),
                    };
                    self.sample = Some(Self::execute_sample(
                        qpu.as_ref(),
                        gates,
                        qubits_to_sample,
                        *shots,
                    )?);
                } else {
                    let gates = GateList::Ir {
                        gates: &final_gates,
                    };
                    self.sample = Some(Self::execute_sample(
                        qpu.as_ref(),
                        gates,
                        qubits_to_sample,
                        *shots,
                    )?);
                }
            } else if !mapped_hamiltonian_list.is_empty() {
                self.grad = match gradient {
                    GradientStrategy::None => None,
                    GradientStrategy::Native => {
                        let (exp_result, grad) =
                            qpu.gradient(&final_gates, &mapped_hamiltonian_list[0])?;
                        self.exp_value = Some(vec![exp_result]);
                        Some(grad)
                    }
                    GradientStrategy::ParameterShiftRule => Some(Self::parameter_shift(
                        qpu.as_ref(),
                        native_gate_set.as_ref().map(std::convert::AsRef::as_ref),
                        self.qpu_config.num_qubits,
                        &final_gates,
                        &self.parameters,
                        &mapped_hamiltonian_list,
                        exp_value,
                    )?),
                    GradientStrategy::HadamardTest => Some(Self::hadamard_test_gradient(
                        qpu.as_ref(),
                        native_gate_set.as_ref().map(std::convert::AsRef::as_ref),
                        self.qpu_config.num_qubits,
                        &final_gates,
                        &self.parameters,
                        &mapped_hamiltonian_list,
                        exp_value,
                    )?),
                };

                if self.exp_value.is_none() {
                    if let Some(native_gate_set) = native_gate_set {
                        let gates = GateList::Native {
                            gates: &Self::translate_circuit(
                                &final_gates,
                                native_gate_set.as_ref(),
                            )?,
                            native_gate_set: native_gate_set.as_ref(),
                        };

                        self.exp_value = Some(Self::execute_exp_value(
                            qpu.as_ref(),
                            self.qpu_config.num_qubits,
                            gates,
                            &mapped_hamiltonian_list,
                            exp_value,
                        )?);
                    } else {
                        let gates = GateList::Ir {
                            gates: &final_gates,
                        };
                        self.exp_value = Some(Self::execute_exp_value(
                            qpu.as_ref(),
                            self.qpu_config.num_qubits,
                            gates,
                            &mapped_hamiltonian_list,
                            exp_value,
                        )?);
                    }
                }
            }
        } else {
            return Err(KetError::ExplicitExecuteInLiveMode);
        }

        self.qpu_config.quantum_execution = quantum_execution;

        self.state = ProcessState::Terminated;

        Ok(())
    }

    /// Delegates a sampling request to the batch backend.
    fn execute_sample(
        qpu: &(impl BatchExecution + ?Sized),
        gates: GateList,
        qubits_to_sample: &[usize],
        shots: usize,
    ) -> Result<SampleData, KetError> {
        match gates {
            GateList::Ir { gates } => qpu.sample(gates, qubits_to_sample, shots),
            GateList::Native { gates, .. } => qpu.sample_native(gates, qubits_to_sample, shots),
        }
    }

    /// Dispatches an expectation-value computation to the appropriate strategy.
    ///
    /// Selects among [`ExpValueStrategy::Native`],
    /// [`ExpValueStrategy::ClassicalShadows`], and
    /// [`ExpValueStrategy::QubitWiseCommutation`] based on `exp_value_strategy`.
    fn execute_exp_value(
        qpu: &(impl BatchExecution + ?Sized),
        num_qubits: usize,
        gates: GateList,
        hamiltonian_list: &[Hamiltonian],
        exp_value_strategy: &ExpValueStrategy,
    ) -> Result<Vec<f64>, KetError> {
        match exp_value_strategy {
            ExpValueStrategy::Native => {
                Self::execute_native_exp_value(qpu, gates, hamiltonian_list)
            }
            ExpValueStrategy::ClassicalShadows {
                bias,
                samples,
                shots,
            } => execute_classical_shadows(
                qpu,
                num_qubits,
                gates,
                hamiltonian_list,
                *samples,
                *shots,
                *bias,
            ),
            ExpValueStrategy::QubitWiseCommutation(shots) => {
                qwc::execute_qwc(qpu, num_qubits, gates, hamiltonian_list, *shots)
            }
        }
    }

    /// Delegates expectation-value computation directly to the backend's
    /// native primitive ([`BatchExecution::exp_value`]).
    fn execute_native_exp_value(
        qpu: &(impl BatchExecution + ?Sized),
        gates: GateList,
        hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, KetError> {
        qwc::execute_qwc_exp_value(qpu, gates, hamiltonian_list)
    }

    fn translate_circuit(
        gates: &[GateInstruction],
        native_gate_set: &(impl NativeGateSet + ?Sized),
    ) -> Result<Vec<NativeGate>, KetError> {
        let mut physical_circuit = Vec::new();
        let mut pending_sq = HashMap::<usize, Matrix>::new();

        macro_rules! flush_sq {
            ($phys:expr) => {{
                if let Some(m) = pending_sq.remove(&$phys) {
                    if !crate::matrix::is_identity(&m) {
                        physical_circuit.append(&mut native_gate_set.translate(&m, $phys)?);
                    }
                }
            }};
        }

        for gate in gates {
            if let Some(decomposed) = &gate.decomposed {
                for gate in decomposed {
                    match gate {
                        DecomposedGate::U(gate, target) => {
                            let m = gate.matrix();
                            let entry = pending_sq
                                .entry(*target)
                                .or_insert_with(crate::matrix::identity);
                            *entry = crate::matrix::mat_mul(&m, entry);
                        }
                        DecomposedGate::CNOT(control, target) => {
                            flush_sq!(*control);
                            flush_sq!(*target);
                            physical_circuit.append(&mut native_gate_set.cnot(*control, *target)?);
                        }
                    }
                }
            } else if gate.control.len() == 1 && matches!(gate.gate, QuantumGate::PauliX) {
                let control = *gate.control.iter().next().unwrap();
                let target = gate.target;
                flush_sq!(control);
                flush_sq!(target);
                physical_circuit.append(&mut native_gate_set.cnot(control, target)?);
            } else {
                if !gate.control.is_empty() {
                    println!(
                        "PANIC! gate: {:?}, control len: {}",
                        gate.gate,
                        gate.control.len()
                    );
                }
                assert!(
                    gate.control.is_empty(),
                    "controlled gates should be decomposed at this pont"
                );

                let m = gate.gate.matrix();
                let entry = pending_sq
                    .entry(gate.target)
                    .or_insert_with(crate::matrix::identity);
                *entry = crate::matrix::mat_mul(&m, entry);
            }
        }

        let remaining_phys: Vec<usize> = pending_sq.keys().copied().collect();
        for phys in remaining_phys {
            flush_sq!(phys);
        }

        Ok(physical_circuit)
    }

    fn shift_param(
        gates: &[GateInstruction],
        shift_inst: usize,
        shift_amt: f64,
    ) -> Vec<GateInstruction> {
        gates
            .par_iter()
            .enumerate()
            .map(|(inst_idx, gate)| {
                let mut gate = gate.clone();

                if inst_idx == shift_inst {
                    gate.gate = match &gate.gate {
                        QuantumGate::RotationX(p) => {
                            QuantumGate::RotationX(Param::Value(p.value() + shift_amt))
                        }
                        QuantumGate::RotationY(p) => {
                            QuantumGate::RotationY(Param::Value(p.value() + shift_amt))
                        }
                        QuantumGate::RotationZ(p) => {
                            QuantumGate::RotationZ(Param::Value(p.value() + shift_amt))
                        }
                        QuantumGate::Phase(p) => {
                            QuantumGate::Phase(Param::Value(p.value() + shift_amt))
                        }
                        other => *other,
                    };
                }
                gate
            })
            .collect()
    }

    #[must_use]
    /// Returns the current [`ProcessState`] of this process.
    pub fn status(&self) -> ProcessState {
        self.state
    }

    fn parameter_shift(
        qpu: &(impl BatchExecution + ?Sized),
        native_gate_set: Option<&dyn NativeGateSet>,
        num_qubits: usize,
        gates: &[GateInstruction],
        parameters: &[f64],
        hamiltonian: &[Hamiltonian],
        exp_value_strategy: &ExpValueStrategy,
    ) -> Result<Vec<f64>, KetError> {
        (0..parameters.len())
            .into_iter()
            .map(|param_idx| -> Result<f64, KetError> {
                let mut multipliers = Vec::new();
                for (inst_idx, gate) in gates.iter().enumerate() {
                    if let Some((idx, mult)) = gate.gate.param_index() {
                        if idx == param_idx {
                            multipliers.push((inst_idx, mult));
                        }
                    }
                }
                if multipliers.is_empty() {
                    return Ok(0.0);
                }
                let mut total_grad = 0.0;
                for (inst_idx, mult) in multipliers {
                    let circuit_plus = Self::shift_param(gates, inst_idx, FRAC_PI_2);
                    let circuit_plus = if let Some(native_gate_set) = native_gate_set {
                        GateList::Native {
                            gates: &Self::translate_circuit(&circuit_plus, native_gate_set)?,
                            native_gate_set,
                        }
                    } else {
                        GateList::Ir {
                            gates: &circuit_plus,
                        }
                    };
                    let e_plus = Self::execute_exp_value(
                        qpu,
                        num_qubits,
                        circuit_plus,
                        hamiltonian,
                        exp_value_strategy,
                    )?[0];

                    let circuit_minus = Self::shift_param(gates, inst_idx, -FRAC_PI_2);
                    let circuit_minus = if let Some(native_gate_set) = native_gate_set {
                        GateList::Native {
                            gates: &Self::translate_circuit(&circuit_minus, native_gate_set)?,
                            native_gate_set,
                        }
                    } else {
                        GateList::Ir {
                            gates: &circuit_minus,
                        }
                    };
                    let e_minus = Self::execute_exp_value(
                        qpu,
                        num_qubits,
                        circuit_minus,
                        hamiltonian,
                        exp_value_strategy,
                    )?[0];

                    total_grad += mult * (e_plus - e_minus) / 2.0;
                }
                Ok(total_grad)
            })
            .collect()
    }

    fn hadamard_test_term(
        qpu: &(impl BatchExecution + ?Sized),
        native_gate_set: Option<&dyn NativeGateSet>,
        num_qubits: usize,
        gates: &[GateInstruction],
        parameter_index: usize,
        parameter_gate_index: usize,
        generator: &[(QuantumGate, usize)],
        generator_coefficient: f64,
        hamiltonian: &[Hamiltonian],
        exp_value_strategy: &ExpValueStrategy,
    ) -> Result<f64, KetError> {
        let ancilla = num_qubits;
        let mut test_gates = vec![
            GateInstruction::new(QuantumGate::PauliX, ancilla),
            GateInstruction::new(QuantumGate::Hadamard, ancilla),
            GateInstruction::new(QuantumGate::s(), ancilla),
        ];

        for (gate_index, gate) in gates.iter().enumerate() {
            if gate.gate.param_index().is_some_and(|(index, _)| {
                index == parameter_index && gate_index == parameter_gate_index
            }) {
                for (generator_gate, target) in generator {
                    let controlled = crate::controlled_gate(*generator_gate, &[ancilla], *target)?;
                    test_gates.extend(controlled.gates);
                }
            }
            test_gates.push(gate.clone());
        }
        test_gates.push(GateInstruction::new(QuantumGate::Hadamard, ancilla));

        let mut full_observable = Hamiltonian::new();
        for observable in hamiltonian {
            for (pauli_string, coefficient) in observable
                .pauli_strings
                .iter()
                .zip(&observable.coefficients)
            {
                let mut pauli_string = pauli_string.clone();
                pauli_string.push(crate::ir::hamiltonian::PauliTerm {
                    pauli: crate::ir::hamiltonian::Pauli::PauliZ,
                    qubit: ancilla,
                });
                full_observable.add_pauli_string(pauli_string, *coefficient);
            }
        }

        let gates = if let Some(native_gate_set) = native_gate_set {
            GateList::Native {
                gates: &Self::translate_circuit(&test_gates, native_gate_set)?,
                native_gate_set,
            }
        } else {
            GateList::Ir { gates: &test_gates }
        };

        Ok(generator_coefficient
            * Self::execute_exp_value(
                qpu,
                num_qubits + 1,
                gates,
                &[full_observable],
                exp_value_strategy,
            )?[0])
    }

    fn hadamard_test_gradient(
        qpu: &(impl BatchExecution + ?Sized),
        native_gate_set: Option<&dyn NativeGateSet>,
        num_qubits: usize,
        gates: &[GateInstruction],
        parameters: &[f64],
        hamiltonian: &[Hamiltonian],
        exp_value_strategy: &ExpValueStrategy,
    ) -> Result<Vec<f64>, KetError> {
        (0..parameters.len())
            .map(|parameter_index| {
                gates
                    .iter()
                    .enumerate()
                    .filter_map(|(gate_index, gate)| {
                        let (index, multiplier) = gate.gate.param_index()?;
                        if index != parameter_index {
                            return None;
                        }

                        let (generator, generator_sign) =
                            Self::generator_for_gate(gates, gate_index)?;
                        Some(Self::hadamard_test_term(
                            qpu,
                            native_gate_set,
                            num_qubits,
                            gates,
                            parameter_index,
                            gate_index,
                            &generator,
                            generator_sign * multiplier / 2.0,
                            hamiltonian,
                            exp_value_strategy,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(|terms| terms.into_iter().sum())
            })
            .collect()
    }

    /// Recover the Pauli generator of a parameterized gate.
    fn generator_for_gate(
        gates: &[GateInstruction],
        gate_index: usize,
    ) -> Option<(Vec<(QuantumGate, usize)>, f64)> {
        let gate = &gates[gate_index];
        let (axis, target) = match gate.gate {
            QuantumGate::RotationX(_) => (QuantumGate::PauliX, gate.target),
            QuantumGate::RotationY(_) => (QuantumGate::PauliY, gate.target),
            QuantumGate::RotationZ(_) | QuantumGate::Phase(_) => (QuantumGate::PauliZ, gate.target),
            _ => return None,
        };

        let mut generator = vec![(axis, target)];
        let mut sign = 1.0;
        if matches!(axis, QuantumGate::PauliZ) {
            let max_controls = gate_index.min(gates.len() - gate_index - 1);
            let mut ladder = None;
            for count in (1..=max_controls).rev() {
                let controls: Vec<_> = (1..=count)
                    .map(|offset| &gates[gate_index - offset])
                    .collect();
                if controls.iter().any(|candidate| {
                    candidate.gate != QuantumGate::PauliX
                        || candidate.control.len() != 1
                        || candidate.target == target
                }) {
                    continue;
                }

                let matches_inverse = controls.iter().enumerate().all(|(offset, candidate)| {
                    let inverse = &gates[gate_index + 1 + offset];
                    inverse.gate == QuantumGate::PauliX
                        && inverse.target == candidate.target
                        && inverse.control == candidate.control
                });
                if matches_inverse {
                    ladder = Some((
                        gate_index - count,
                        controls
                            .iter()
                            .map(|candidate| *candidate.control.iter().next().unwrap())
                            .collect::<Vec<_>>(),
                    ));
                    break;
                }
            }

            if let Some((ladder_start, controls)) = ladder {
                let mut body_qubits = controls;
                body_qubits.push(target);

                for qubit in body_qubits {
                    let basis = gates[..ladder_start].iter().rev().find_map(|candidate| {
                        if candidate.target != qubit || !candidate.control.is_empty() {
                            return None;
                        }

                        match candidate.gate {
                            QuantumGate::Hadamard => Some((QuantumGate::PauliX, 1.0)),
                            QuantumGate::RotationX(Param::Value(angle))
                                if (angle - FRAC_PI_2).abs() <= 1e-10 =>
                            {
                                Some((QuantumGate::PauliY, -1.0))
                            }
                            _ => None,
                        }
                    });

                    if let Some((pauli, pauli_sign)) = basis {
                        if qubit == target {
                            generator[0].0 = pauli;
                        } else if let Some(term) =
                            generator.iter_mut().find(|(_, index)| *index == qubit)
                        {
                            term.0 = pauli;
                        } else {
                            generator.push((pauli, qubit));
                        }
                        sign *= pauli_sign;
                    } else if qubit != target {
                        generator.push((QuantumGate::PauliZ, qubit));
                    }
                }
            }
        }
        Some((generator, sign))
    }
}
