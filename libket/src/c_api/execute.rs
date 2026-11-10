// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! C API bindings for quantum execution backends.
//!
//! This module defines the FFI-compatible structs ([`CLiveExecution`],
//! [`CBatchExecution`], [`CNativeGateSet`]) that allow external C/C++ quantum
//! simulators or hardware drivers to plug into the Libket execution pipeline.
//!
//! ## JSON serialization contract
//!
//! All data exchanged between Rust and the C callbacks is serialized as JSON:
//!
//! - **Rust → C**: gate instructions, Hamiltonians, and gate matrices are
//!   serialized with [`serde_json`] into null-terminated C strings before
//!   being passed to the corresponding callback.
//! - **C → Rust**: JSON strings returned by C callbacks (e.g., sample counts,
//!   state dumps, native gate sequences) are parsed back with [`serde_json`]
//!   on the Rust side. If parsing fails, [`KetError::SerdeError`] is returned.
//!
//! Heap-allocated strings returned by C callbacks are **not** freed by Rust;
//! the C backend is responsible for their lifetime (typically they are freed
//! via [`super::ket_string_delete`] by the caller of the top-level C API).

use std::{
    ffi::{c_char, CStr, CString},
    ptr::null,
};

use crate::{
    c_api::c_ok,
    error::KetError,
    execution::{
        BatchExecution, DumpData, ExpValueStrategy, GradientStrategy, LiveExecution, NativeGate,
        NativeGateSet, QuantumExecution, SampleData,
    },
    ir::{gate::GateInstruction, hamiltonian::Hamiltonian},
    matrix::Matrix,
    process::QPUConfig,
};

/// FFI-compatible function-pointer table for a *live* (gate-at-a-time) execution backend.
///
/// In live mode each gate is dispatched to the hardware or simulator
/// immediately as it is applied to the process, and measurement results are
/// available without an explicit `ket_process_execute` call.
///
/// Construct a `CLiveExecution` value on the C side, fill in all function
/// pointers, then call [`ket_quantum_execution_live`] to create the
/// corresponding [`QuantumExecution`] variant.
///
/// All callbacks must return `0` on success or a non-zero libket error code on
/// failure.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CLiveExecution {
    /// Dispatch a single quantum gate to the backend.
    ///
    /// `json` is a null-terminated JSON string encoding the gate instruction
    /// (type, parameters, target qubit, etc.).  Returns `0` on success.
    pub compute_gate: fn(json: *const c_char) -> i32,
    /// Dispatch a sequence of pre-translated native gate instructions to the backend.
    ///
    /// `json` is a null-terminated JSON string encoding the array of native
    /// gate instructions (each a `[name, angles, qubits]` triple). Returns `0`
    /// on success.
    pub compute_native_gates: fn(json: *const c_char) -> i32,

    /// Perform a single-shot measurement of `len` qubits.
    ///
    /// `qubits` points to an array of `len` qubit indices.  On success the
    /// measurement result is written to `*result` as a bitmask where bit `i`
    /// corresponds to `qubits[i]`.  Returns `0` on success.
    pub measure: fn(qubits: *const usize, len: usize, result: &mut u64) -> i32,

    /// Retrieve the full quantum state (probability amplitudes) of `len` qubits.
    ///
    /// `qubits` points to an array of `len` qubit indices.  On success
    /// `*result_json` is set to a heap-allocated null-terminated JSON string
    /// encoding the state; Rust will parse it and then it is the C side's
    /// responsibility to manage the lifetime.  Returns `0` on success.
    pub dump: fn(qubits: *const usize, len: usize, result_json: &mut *const c_char) -> i32,

    /// Perform a repeated measurement sample of `len` qubits over `shots` shots.
    ///
    /// `qubits` points to an array of `len` qubit indices.  On success
    /// `*result_json` is set to a heap-allocated null-terminated JSON string
    /// with the sample counts.  Returns `0` on success.
    pub sample:
        fn(qubits: *const usize, len: usize, shots: usize, result_json: &mut *const c_char) -> i32,

    /// Compute the expectation value of a Hamiltonian on the current quantum state.
    ///
    /// `hamiltonian_json` is a null-terminated JSON string encoding the
    /// Hamiltonian.  On success `*result` is set to the computed expectation
    /// value.  Returns `0` on success.
    pub exp_value: fn(hamiltonian_json: *const c_char, result: &mut f64) -> i32,
}

impl LiveExecution for CLiveExecution {
    fn compute_gate(&mut self, gate: &GateInstruction) -> Result<(), KetError> {
        let Ok(json) = serde_json::to_string(gate) else {
            return Err(KetError::SerdeError);
        };

        let Ok(json) = CString::new(json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.compute_gate)(json.as_ptr()));

        match err {
            KetError::Success => Ok(()),
            err => Err(err),
        }
    }

    fn compute_native_gates(&mut self, gates: &[NativeGate]) -> Result<(), KetError> {
        let Ok(gates) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };

        let Ok(gates) = CString::new(gates) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.compute_native_gates)(gates.as_ptr()));
        match err {
            KetError::Success => Ok(()),
            err => Err(err),
        }
    }

    fn measure(&mut self, qubits: &[usize]) -> Result<u64, KetError> {
        let mut result = 42;

        let err =
            KetError::from_error_code((self.measure)(qubits.as_ptr(), qubits.len(), &mut result));

        match err {
            KetError::Success => Ok(result),
            err => Err(err),
        }
    }

    fn sample(&mut self, qubits: &[usize], shots: usize) -> Result<SampleData, KetError> {
        let mut result_json = null();

        let err = KetError::from_error_code((self.sample)(
            qubits.as_ptr(),
            qubits.len(),
            shots,
            &mut result_json,
        ));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(result_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };

                match serde_json::from_str::<SampleData>(result_json) {
                    Ok(sample_date) => Ok(sample_date),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }

    fn exp_value(&mut self, hamiltonian: Hamiltonian) -> Result<f64, KetError> {
        let mut result = f64::NAN;

        let Ok(json) = serde_json::to_string(&hamiltonian) else {
            return Err(KetError::SerdeError);
        };

        let Ok(json) = CString::new(json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.exp_value)(json.as_ptr(), &mut result));

        match err {
            KetError::Success => Ok(result),
            err => Err(err),
        }
    }

    fn dump(&mut self, qubits: &[usize]) -> Result<DumpData, KetError> {
        let mut result_json = null();

        let err =
            KetError::from_error_code((self.dump)(qubits.as_ptr(), qubits.len(), &mut result_json));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(result_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };

                match serde_json::from_str::<DumpData>(result_json) {
                    Ok(dump_data) => Ok(dump_data),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }
}

/// FFI-compatible function-pointer table for a *batch* execution backend.
///
/// In batch mode the entire compiled circuit is serialized and handed to the
/// backend in a single call when `ket_process_execute` is invoked, rather than
/// dispatching gate-by-gate.
///
/// Construct a `CBatchExecution` value on the C side, fill in all function
/// pointers, then call [`ket_quantum_execution_batch`] to create the
/// corresponding [`QuantumExecution`] variant.
///
/// All callbacks must return `0` on success or a non-zero libket error code on
/// failure.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CBatchExecution {
    pub sample: fn(
        gates_json: *const c_char,
        qubits_to_sample: *const usize,
        qubits_to_sample_len: usize,
        shots: usize,
        sample_json: &mut *const c_char,
    ) -> i32,

    pub exp_value: fn(
        gates_json: *const c_char,
        hamiltonian_list_json: *const c_char,
        result: *mut f64,
    ) -> i32,

    pub sample_native: fn(
        gates_json: *const c_char,
        qubits_to_sample: *const usize,
        qubits_to_sample_len: usize,
        shots: usize,
        sample_json: &mut *const c_char,
    ) -> i32,

    pub exp_value_native: fn(
        gates_json: *const c_char,
        hamiltonian_list_json: *const c_char,
        result: *mut f64,
    ) -> i32,

    pub gradient: fn(
        gates_json: *const c_char,
        hamiltonian_list_json: *const c_char,
        exp_result: &mut f64,
        grad_json: &mut *const c_char,
    ) -> i32,
}

impl BatchExecution for CBatchExecution {
    fn sample(
        &self,
        gates: &[GateInstruction],
        qubits_to_sample: &[usize],
        shots: usize,
    ) -> Result<SampleData, KetError> {
        let mut sample_json = std::ptr::null();

        let Ok(gates_json) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gates_json) = CString::new(gates_json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.sample)(
            gates_json.as_ptr(),
            qubits_to_sample.as_ptr(),
            qubits_to_sample.len(),
            shots,
            &mut sample_json,
        ));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(sample_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };

                match serde_json::from_str::<SampleData>(result_json) {
                    Ok(sample_data) => Ok(sample_data),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }

    fn exp_value(
        &self,
        gates: &[GateInstruction],
        hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, KetError> {
        let mut result = vec![f64::NAN; hamiltonian_list.len()];

        let Ok(gates_json) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gates_json) = CString::new(gates_json) else {
            return Err(KetError::SerdeError);
        };

        let Ok(ham_json) = serde_json::to_string(&hamiltonian_list) else {
            return Err(KetError::SerdeError);
        };
        let Ok(ham_json) = CString::new(ham_json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.exp_value)(
            gates_json.as_ptr(),
            ham_json.as_ptr(),
            result.as_mut_ptr(),
        ));

        match err {
            KetError::Success => Ok(result),
            err => Err(err),
        }
    }

    fn sample_native(
        &self,
        gates: &[NativeGate],
        qubits_to_sample: &[usize],
        shots: usize,
    ) -> Result<SampleData, KetError> {
        let mut sample_json = std::ptr::null();

        let Ok(gates_json) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gates_json) = CString::new(gates_json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.sample_native)(
            gates_json.as_ptr(),
            qubits_to_sample.as_ptr(),
            qubits_to_sample.len(),
            shots,
            &mut sample_json,
        ));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(sample_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };

                match serde_json::from_str::<SampleData>(result_json) {
                    Ok(sample_data) => Ok(sample_data),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }

    fn exp_value_native(
        &self,
        gates: &[NativeGate],
        hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, KetError> {
        let mut result = vec![f64::NAN; hamiltonian_list.len()];

        let Ok(gates_json) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gates_json) = CString::new(gates_json) else {
            return Err(KetError::SerdeError);
        };

        let Ok(ham_json) = serde_json::to_string(&hamiltonian_list) else {
            return Err(KetError::SerdeError);
        };
        let Ok(ham_json) = CString::new(ham_json) else {
            return Err(KetError::SerdeError);
        };

        let err = KetError::from_error_code((self.exp_value_native)(
            gates_json.as_ptr(),
            ham_json.as_ptr(),
            result.as_mut_ptr(),
        ));

        match err {
            KetError::Success => Ok(result),
            err => Err(err),
        }
    }

    fn gradient(
        &self,
        gates: &[GateInstruction],
        hamiltonian: &Hamiltonian,
    ) -> Result<(f64, Vec<f64>), KetError> {
        let mut exp_result = 0.0;

        let Ok(gates_json) = serde_json::to_string(gates) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gates_json) = CString::new(gates_json) else {
            return Err(KetError::SerdeError);
        };

        let Ok(ham_json) = serde_json::to_string(hamiltonian) else {
            return Err(KetError::SerdeError);
        };
        let Ok(ham_json) = CString::new(ham_json) else {
            return Err(KetError::SerdeError);
        };

        let mut grad_json = std::ptr::null();

        let err = KetError::from_error_code((self.gradient)(
            gates_json.as_ptr(),
            ham_json.as_ptr(),
            &mut exp_result,
            &mut grad_json,
        ));

        match err {
            KetError::Success => {
                let Ok(grad_json) = { unsafe { CStr::from_ptr(grad_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };

                match serde_json::from_str::<Vec<f64>>(grad_json) {
                    Ok(grad_json) => Ok((exp_result, grad_json)),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }
}

/// FFI-compatible function-pointer table for translating abstract gates into
/// backend-native instructions.
///
/// The compiler calls these callbacks during the decomposition and routing
/// passes to convert high-level gate representations into the instruction set
/// understood by the target hardware or simulator.
///
/// Construct a `CNativeGateSet` value on the C side, fill in all function
/// pointers, then pass it to [`ket_quantum_execution_batch`].  Passing `NULL`
/// instead uses identity translation (no decomposition).
///
/// All callbacks must return `0` on success or a non-zero libket error code on
/// failure.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct CNativeGateSet {
    /// Translate a single-qubit gate (given as a 2×2 unitary matrix) into
    /// native gate instructions.
    ///
    /// `gate_json` is a null-terminated JSON string encoding the gate as a
    /// flat 2×2 complex matrix `[[re,im,re,im],[re,im,re,im]]`.  `target` is
    /// the target qubit index.  On success `*native_gate_json` is set to a
    /// heap-allocated null-terminated JSON string encoding the native gate
    /// sequence.  Returns `0` on success.
    pub translate:
        fn(gate_json: *const c_char, target: usize, native_gate_json: &mut *const c_char) -> i32,

    /// Produce the native gate sequence that implements a CNOT between two qubits.
    ///
    /// The first argument is the control qubit index and the second is the
    /// target qubit index.  On success `*native_gate_json` is set to a
    /// heap-allocated null-terminated JSON string encoding the native CNOT
    /// decomposition.  Returns `0` on success.
    pub cnot: fn(control: usize, target: usize, native_gate_json: &mut *const c_char) -> i32,
}

impl NativeGateSet for CNativeGateSet {
    fn translate(&self, matrix: &Matrix, target: usize) -> Result<Vec<NativeGate>, KetError> {
        let matrix_json = [
            [
                matrix[0][0].re,
                matrix[0][0].im,
                matrix[0][1].re,
                matrix[0][1].im,
            ],
            [
                matrix[1][0].re,
                matrix[1][0].im,
                matrix[1][1].re,
                matrix[1][1].im,
            ],
        ];
        let Ok(gate) = serde_json::to_string(&matrix_json) else {
            return Err(KetError::SerdeError);
        };
        let Ok(gate) = CString::new(gate) else {
            return Err(KetError::SerdeError);
        };

        let mut result_json = null();

        let err =
            KetError::from_error_code((self.translate)(gate.as_ptr(), target, &mut result_json));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(result_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };
                match serde_json::from_str(result_json) {
                    Ok(result) => Ok(result),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }

    fn cnot(&self, control: usize, target: usize) -> Result<Vec<NativeGate>, KetError> {
        let mut result_json = null();

        let err = KetError::from_error_code((self.cnot)(control, target, &mut result_json));

        match err {
            KetError::Success => {
                let Ok(result_json) = { unsafe { CStr::from_ptr(result_json) } }.to_str() else {
                    return Err(KetError::SerdeError);
                };
                match serde_json::from_str(result_json) {
                    Ok(result) => Ok(result),
                    Err(_) => Err(KetError::SerdeError),
                }
            }
            err => Err(err),
        }
    }
}

/// Creates a live-mode [`QuantumExecution`] backed by the provided callback table.
///
/// In live mode each gate is dispatched to the backend immediately as it is
/// applied to the process. The `decompose` flag controls whether multi-qubit
/// gates are decomposed into single-qubit and two-qubit native gates before
/// being dispatched.
///
/// # Parameters
/// - `live`: Pointer to a caller-owned [`CLiveExecution`] struct whose
///   function pointers implement the backend. The struct is cloned; the caller
///   retains ownership of the original.
/// - `decompose`: If `true`, enables multi-qubit gate decomposition before
///   dispatching each gate to the backend.
/// - `quantum_execution`: Output: receives the pointer to the newly allocated
///   [`QuantumExecution`] object. Ownership is transferred to the caller.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_quantum_execution_live(
    num_qubits: usize,
    live: &CLiveExecution,
    decompose: bool,
    native_gate_set: *const CNativeGateSet,
    qpu_config: &mut *mut QPUConfig,
) -> i32 {
    let native_gate_set = if native_gate_set.is_null() {
        None
    } else {
        Some(Box::new(unsafe { &*native_gate_set }.clone()) as Box<dyn NativeGateSet>)
    };

    let execution = QuantumExecution::Live {
        qpu: Box::new(live.clone()),
        decompose,
        native_gate_set,
    };

    *qpu_config = Box::into_raw(Box::new(QPUConfig {
        num_qubits,
        quantum_execution: Some(execution),
    }));

    c_ok()
}

/// Creates a batch-mode [`QuantumExecution`] backed by the provided callback tables.
///
/// In batch mode the full compiled circuit is handed to the backend in a
/// single call when `ket_process_execute` is invoked. The `native_gate_set`
/// callback table is used during the decomposition and routing passes to
/// translate abstract gates into hardware-native instructions.
///
/// # Parameters
/// - `batch`: Pointer to a caller-owned [`CBatchExecution`] struct. The struct
///   is cloned; the caller retains ownership of the original.
/// - `native_gate_set`: Pointer to a caller-owned [`CNativeGateSet`] struct,
///   or `NULL` to use identity translation (no decomposition into native gates).
/// - `gradient_strategy_json`: JSON-encoded [`GradientStrategy`]. When it is
///   `HadamardTest`, `num_qubits` must include one additional physical qubit
///   for the temporary Hadamard-test ancilla.
/// - `coupling_graph_json`: A null-terminated JSON string that is either
///   `null` (no connectivity constraint) or a JSON array of `[usize, usize]`
///   edge pairs describing the QPU coupling graph used for qubit routing.
/// - `quantum_execution`: Output: receives the pointer to the newly allocated
///   [`QuantumExecution`] object. Ownership is transferred to the caller.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_quantum_execution_batch(
    num_qubits: usize,
    batch: &CBatchExecution,
    native_gate_set: *const CNativeGateSet,
    decompose: bool,
    gradient_strategy_json: *const c_char,
    coupling_graph_json: *const c_char,
    exp_value_strategy_json: *const c_char,
    qpu_config: &mut *mut QPUConfig,
) -> i32 {
    let native_gate_set: Option<Box<dyn NativeGateSet>> = unsafe {
        if native_gate_set.is_null() {
            None
        } else {
            Some(Box::new((*native_gate_set).clone()))
        }
    };

    let coupling_graph = from_json!(coupling_graph_json, Option<Vec<(usize, usize)>>);
    let exp_value = from_json!(exp_value_strategy_json, ExpValueStrategy);
    let gradient_strategy = from_json!(gradient_strategy_json, GradientStrategy);

    let execution = QuantumExecution::Batch {
        qpu: Box::new(batch.clone()),
        native_gate_set,
        gradient: gradient_strategy,
        exp_value,
        coupling_graph,
        decompose,
    };

    *qpu_config = Box::into_raw(Box::new(QPUConfig {
        num_qubits,
        quantum_execution: Some(execution),
    }));

    c_ok()
}
