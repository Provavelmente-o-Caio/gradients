// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::missing_safety_doc, clippy::not_unsafe_ptr_arg_deref)]

//! C API for interacting with `Process` operations.

use std::ffi::{c_char, CStr, CString};

use crate::{
    c_api::{c_error, c_ok},
    error::KetError,
    ir::{block::Block, gate::Param, hamiltonian::Hamiltonian},
    process::{Process, ProcessState, QPUConfig},
};

/// Allocates a new [`Process`] using the provided QPU configuration.
///
/// `qpu_config` is consumed by this call: ownership is transferred to the
/// newly created `Process` and the pointer must not be used afterwards.
///
/// # Parameters
/// - `qpu_config`: Pointer to a [`QPUConfig`].
/// - `process`: Output: receives the pointer to the newly allocated
///   [`Process`]. Must eventually be freed with [`ket_process_delete`].
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_new(qpu_config: *mut QPUConfig, process: &mut *mut Process) -> i32 {
    let qpu_config = unsafe { *Box::from_raw(qpu_config) };
    *process = Box::into_raw(Box::new(Process::new(qpu_config)));
    c_ok()
}

/// Allocates a new [`QPUConfig`] for a QPU with the specified number of physical qubits.
///
/// The newly created configuration has **no execution backend** attached. A
/// `QuantumExecution` backend must be set (via `ket_quantum_execution_live`
/// or `ket_quantum_execution_batch`) before passing the config to
/// [`ket_process_new`].
///
/// # Parameters
/// - `num_qubits`: The number of physical qubits available on the QPU.
/// - `qpu_config`: Output: receives the pointer to the newly allocated
///   [`QPUConfig`]. Ownership is transferred to the caller until it is
///   consumed by [`ket_process_new`].
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_config_new(num_qubits: usize, qpu_config: &mut *mut QPUConfig) -> i32 {
    *qpu_config = Box::into_raw(Box::new(QPUConfig {
        num_qubits,
        quantum_execution: None,
    }));
    c_ok()
}

/// Deallocates the memory associated with a [`Process`].
///
/// # Parameters
/// - `process`: Pointer to the [`Process`] to free.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_delete(process: *mut Process) -> i32 {
    unsafe {
        let _ = Box::from_raw(process);
    }
    c_ok()
}

/// Allocates a new logical qubit in the process.
///
/// Returns an error if the process is in a state that does not accept new
/// qubits (e.g., `ProcessTerminated`) or if the QPU qubit limit has been
/// reached (`QubitLimitExceeded`).
///
/// # Parameters
/// - `process`: The process in which the qubit is allocated.
/// - `qubit`: Output: receives the index of the newly allocated logical qubit.
///
/// # Returns
/// `0` on success, non-zero error code on failure (including
/// `ProcessTerminated` and `QubitLimitExceeded`).
#[no_mangle]
pub extern "C" fn ket_process_alloc(process: &mut Process, qubit: &mut usize) -> i32 {
    *qubit = c_error!(process.alloc());
    c_ok()
}

/// Appends a [`Block`] of gates to the process.
///
/// The `block` is **consumed** by this call and must not be used afterwards.
/// Returns an error if the process state does not allow gate appending
/// (`GateAppendForbidden`) or if a qubit index referenced in the block is
/// out of range (`QubitIndexOutOfRange`).
///
/// # Parameters
/// - `process`: The process to which the block is appended.
/// - `block`: The block to consume and append. Ownership is transferred.
///
/// # Returns
/// `0` on success, non-zero error code on failure (including
/// `GateAppendForbidden` and `QubitIndexOutOfRange`).
#[no_mangle]
pub extern "C" fn ket_process_append_block(process: &mut Process, block: *mut Block) -> i32 {
    let block = unsafe { *Box::from_raw(block) };
    c_error(process.append_block(block))
}

/// Returns the accumulated gate instructions of the process as a JSON string.
///
/// The returned string is a JSON array of gate instruction objects representing
/// the gates in the process's main block. The string is heap-allocated and the
/// caller **must** free it with [`super::ket_string_delete`].
///
/// # Parameters
/// - `process`: The process whose gate list is serialized.
/// - `block`: Output: set to a pointer to a heap-allocated null-terminated
///   JSON string.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_gates_json(process: &Process, block: &mut *mut c_char) -> i32 {
    *block = to_json!(&process.flat_main_block().unwrap().gates).into_raw();
    c_ok()
}

/// Performs a single-shot mid-circuit measurement of `qubits`.
///
/// This function is only valid in **live mode**. Any gates accumulated since
/// the last measurement are dispatched to the backend before the measurement
/// is performed.
///
/// # Parameters
/// - `process`: The process on which the measurement is performed.
/// - `qubits`: Pointer to an array of qubit indices to measure.
/// - `qubits_len`: Number of elements in `qubits`.
/// - `result`: Output: the measurement result encoded as a bitmask
///   (bit `i` corresponds to `qubits[i]`).
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_measure(
    process: &mut Process,
    qubits: *const usize,
    qubits_len: usize,
    result: &mut u64,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(qubits, qubits_len) };
    *result = c_error!(process.measure(qubits));
    c_ok()
}

/// Dumps the quantum state of `qubits` (live mode only).
///
/// Returns the full probability amplitude vector for the specified qubits as a
/// heap-allocated JSON string. The caller **must** free it with
/// [`super::ket_string_delete`].
///
/// # Parameters
/// - `process`: The process whose state is dumped.
/// - `qubits`: Pointer to an array of qubit indices to dump.
/// - `qubits_len`: Number of elements in `qubits`.
/// - `dump_json`: Output: set to a pointer to a heap-allocated
///   null-terminated JSON string encoding the quantum state.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_dump(
    process: &mut Process,
    qubits: *const usize,
    qubits_len: usize,
    dump_json: &mut *const c_char,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(qubits, qubits_len) };
    *dump_json = to_json!(&c_error!(process.dump(qubits))).into_raw();

    c_ok()
}

/// Requests a measurement sample of `qubits` over `shots` repetitions.
///
/// In **live mode**, the circuit is executed immediately and the result is
/// returned as a heap-allocated JSON string (caller must free with
/// [`super::ket_string_delete`]). In **batch mode**, the request is recorded for the
/// next [`ket_process_execute`] call and `*sample_json` is set to a JSON
/// `null` string; the actual data becomes available after execution and can be
/// retrieved with [`ket_process_read_sample`].
///
/// # Parameters
/// - `process`: The process on which the sample is requested.
/// - `qubits`: Pointer to an array of qubit indices to sample.
/// - `qubits_len`: Number of elements in `qubits`.
/// - `shots`: Number of repetitions.
/// - `sample_json`: Output: set to a pointer to a heap-allocated
///   null-terminated JSON string (caller must free with [`super::ket_string_delete`]).
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_sample(
    process: &mut Process,
    qubits: *const usize,
    qubits_len: usize,
    shots: usize,
    sample_json: &mut *const c_char,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(qubits, qubits_len) };
    *sample_json = to_json!(&c_error!(process.sample(qubits, shots))).into_raw();
    c_ok()
}

/// Returns a serialized JSON array of expectation values from the most recent execution.
///
/// If no execution has been performed yet, the returned JSON string contains
/// `null`. The string is heap-allocated; the caller **must** free it with
/// [`super::ket_string_delete`].
///
/// # Parameters
/// - `process`: The process from which expectation values are read.
/// - `result_json`: Output: set to a pointer to a heap-allocated
///   null-terminated JSON string.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_read_exp_value(
    process: &Process,
    result_json: &mut *const c_char,
) -> i32 {
    *result_json = to_json!(&process.read_exp_value()).into_raw();
    c_ok()
}

/// Returns a serialized JSON array of parameter-shift gradients from the most recent execution.
///
/// If no execution has been performed yet, the returned JSON string contains
/// `null`. The string is heap-allocated; the caller **must** free it with
/// [`super::ket_string_delete`].
///
/// # Parameters
/// - `process`: The process from which gradients are read.
/// - `result_json`: Output: set to a pointer to a heap-allocated
///   null-terminated JSON string.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_read_gradient(
    process: &Process,
    result_json: &mut *const c_char,
) -> i32 {
    *result_json = to_json!(&process.read_gradient()).into_raw();
    c_ok()
}

/// Registers a new differentiable parameter with the given initial value.
///
/// The returned `param_index` uniquely identifies this parameter within the
/// process. Use it to construct `Param::Ref` values when building
/// parameterised gate instructions. When the parameter value changes between
/// executions, gates that reference the same index are automatically updated
/// during the next compilation pass.
///
/// # Parameters
/// - `process`: The process in which the parameter is registered.
/// - `param`: The initial numeric value of the parameter.
/// - `param_index`: Output: receives the index that identifies this parameter
///   within the process.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_param(
    process: &mut Process,
    param: f64,
    param_index: &mut usize,
) -> i32 {
    let param = process.param(param);
    if let Param::Ref { index, .. } = param {
        *param_index = index;
    }
    c_ok()
}

/// Compiles and executes the accumulated circuit.
///
/// This is the primary entry point for **batch mode** execution. Calling it in
/// live mode returns an `ExplicitExecuteInLiveMode` error. If no pending
/// measurement or expectation-value requests have been recorded,
/// `NoPendingMeasurement` is returned. If the process is already in the
/// `Terminated` state, this call is idempotent and returns success.
///
/// # Parameters
/// - `process`: The process to compile and execute.
///
/// # Returns
/// `0` on success, non-zero error code on failure (including
/// `NoPendingMeasurement` and `ExplicitExecuteInLiveMode`).
#[no_mangle]
pub extern "C" fn ket_process_execute(process: &mut Process) -> i32 {
    c_error(process.execute())
}

/// Reads the cached sample data from a [`Process`].
///
/// Returns the sample counts recorded during the most recent execution as a
/// heap-allocated JSON string. If no execution has been performed yet, the
/// returned JSON string contains `null`. The caller **must** free the string
/// with [`super::ket_string_delete`].
///
/// # Parameters
/// - `process`: The process from which sample data is read.
/// - `sample_json`: Output: set to a pointer to a heap-allocated
///   null-terminated JSON string.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_read_sample(
    process: &mut Process,
    sample_json: &mut *const c_char,
) -> i32 {
    *sample_json = to_json!(&process.read_sample()).into_raw();

    c_ok()
}

/// Requests the expectation value computation for a given [`Hamiltonian`] (passed as JSON) on a [`Process`].
///
/// In **live mode**, the expectation value is computed immediately:
/// `*some_result` is set to `true` and the value is written to `*result`.
/// In **batch mode**, the request is recorded for the next
/// [`ket_process_execute`] call: `*some_result` is set to `false` and
/// `*result` is left unmodified until execution completes and
/// [`ket_process_read_exp_value`] is called.
///
/// # Parameters
/// - `process`: The process on which the expectation value is computed.
/// - `hamiltonian_json`: A null-terminated JSON string encoding the
///   Hamiltonian.
/// - `result`: Output: the computed expectation value (valid only when
///   `*some_result` is `true`).
/// - `some_result`: Output: set to `true` if the expectation value was
///   computed immediately (live mode), or `false` if the request was
///   deferred (batch mode).
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_exp_value(
    process: &mut Process,
    hamiltonian_json: *const c_char,
    result: &mut f64,
    some_result: &mut bool,
) -> i32 {
    let hamiltonian = from_json!(hamiltonian_json, Hamiltonian);
    match c_error!(process.exp_value(hamiltonian)) {
        Some(value) => {
            *result = value;
            *some_result = true;
        }
        None => {
            *some_result = false;
        }
    }
    c_ok()
}

/// Converts a [`ProcessState`] into the corresponding static C string.
///
/// Maps each variant to a null-terminated `&'static CStr`:
/// - [`ProcessState::AcceptingGate`] → `"AcceptingGate"`
/// - [`ProcessState::AcceptingHamiltonian`] → `"AcceptingHamiltonian"`
/// - [`ProcessState::ReadyToExecute`] → `"ReadyToExecute"`
/// - [`ProcessState::Terminated`] → `"Terminated"`
///
/// The returned pointer is valid for the lifetime of the program and must
/// **not** be freed.
impl From<ProcessState> for &'static CStr {
    fn from(value: ProcessState) -> Self {
        match value {
            ProcessState::AcceptingGate => c"AcceptingGate",
            ProcessState::AcceptingHamiltonian => c"AcceptingHamiltonian",
            ProcessState::ReadyToExecute => c"ReadyToExecute",
            ProcessState::Terminated => c"Terminated",
        }
    }
}

/// Returns the current status of the process as a static C string.
///
/// The string is one of:
/// - `"AcceptingGate"`: the process is ready to accept gate blocks.
/// - `"AcceptingHamiltonian"`: all gates have been appended; expectation-value
///   requests can now be submitted.
/// - `"ReadyToExecute"`: execution has been requested and the circuit is ready.
/// - `"Terminated"`: execution has completed and results are available.
///
/// The returned pointer points to a static string and must **not** be freed.
///
/// # Parameters
/// - `process`: The process whose status is queried.
/// - `status`: Output: set to a pointer to a static null-terminated string.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_status(process: &Process, status: &mut *const c_char) -> i32 {
    let process_status: &CStr = process.status().into();
    *status = process_status.as_ptr();
    c_ok()
}

/// Registers a new parameter with the given initial value in the process.
///
/// The parameter's index within the process parameter list is written to `index`.
///
/// # Parameters
/// - `process`: The process to register the parameter in.
/// - `value`: The initial value of the parameter.
/// - `index`: Output: set to the registered parameter's index.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub extern "C" fn ket_process_set_parameter(
    process: &mut Process,
    value: f64,
    index: &mut usize,
) -> i32 {
    let Param::Ref {
        index: param_index, ..
    } = process.param(value)
    else {
        unreachable!()
    };
    *index = param_index;

    c_ok()
}

/// Parses an `OpenQASM` 2.0 source string and returns a heap-allocated
/// [`Block`] containing the resulting gate sequence.
///
/// Physical qubit indices are mapped to the `qreg` declarations in the source
/// in order. The total number of entries in `qubits` must be ≥ the sum of all
/// `qreg` sizes declared in `qasm`.
///
/// `measure` statements are **not** executed. Their qubit-to-cbit mapping is
/// recorded so that subsequent `if` statements are converted to controlled
/// quantum gates instead of classical branches.
///
/// On success, a newly heap-allocated `Block` is written to `*block_out`.
/// The caller takes ownership and must eventually free it with
/// [`ket_block_delete`].
///
/// # Parameters
/// - `qubits`     : Pointer to an array of `size_t` physical qubit indices.
/// - `qubits_len` : Number of entries in `qubits`.
/// - `qasm`       : Null-terminated C string with the `OpenQASM` 2.0 source.
/// - `block_out`  : Output: on success, set to a pointer to the new `Block`.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[cfg(feature = "qasm")]
#[no_mangle]
pub unsafe extern "C" fn ket_parse_openqasm(
    qubits: *const usize,
    qubits_len: usize,
    qasm: *const c_char,
    block_out: &mut *mut Block,
) -> i32 {
    if qasm.is_null() || (qubits_len > 0 && qubits.is_null()) {
        return KetError::OpenQasmParseError.error_code();
    }

    let qasm_str = match CStr::from_ptr(qasm).to_str() {
        Ok(s) => s,
        Err(_) => return KetError::OpenQasmParseError.error_code(),
    };

    let qubit_slice = if qubits_len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(qubits, qubits_len)
    };

    match crate::qasm::parse_openqasm(qubit_slice, qasm_str) {
        Ok(bb) => {
            let mut block = Block::new();
            block.append_basic_block(bb);
            *block_out = Box::into_raw(Box::new(block));
            c_ok()
        }
        Err(e) => e.error_code(),
    }
}

/// Creates a BasicBlock that implements the Quantum Fourier Transform (QFT) on the given qubits.
///
/// On success, a newly heap-allocated `Block` is written to `*block_out`.
/// The caller takes ownership and must eventually free it with [`ket_block_delete`].
///
/// # Parameters
/// - `qubits`     : Pointer to an array of `size_t` physical qubit indices. The MSQ should be at index 0.
/// - `qubits_len` : Number of entries in `qubits`.
/// - `do_swap`    : Boolean flag indicating whether to add SWAP gates at the end of the circuit to reverse the qubit order.
/// - `block_out`  : Output: on success, set to a pointer to the new `Block`.
///
/// # Returns
/// `0` on success, non-zero error code on failure.
#[no_mangle]
pub unsafe extern "C" fn ket_qft(
    qubits: *const usize,
    qubits_len: usize,
    do_swap: bool,
    block_out: &mut *mut Block,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(qubits, qubits_len) };

    *block_out = Box::into_raw(Box::new(c_error!(crate::qft(qubits, do_swap))));

    c_ok()
}
