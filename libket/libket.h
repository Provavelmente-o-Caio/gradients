/*
 * SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa<evandro@quantuloop.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * A straight-line sequence of quantum gate instructions.
 *
 * A [`BasicBlock`] is created empty and gates are appended one at a time via
 * [`BasicBlock::append_gate`] or in bulk via [`BasicBlock::append_block`].
 * Each append operation attempts lightweight algebraic simplifications:
 * adjacent inverse gate pairs are cancelled and consecutive same-axis
 * rotation gates are merged into one.
 *
 * The block also maintains a dependency index (`qubits_op`) consumed by the
 * qubit-allocation and circuit-mapping passes.
 */
typedef struct BasicBlock BasicBlock;

/**
 * A quantum process: the top-level compilation and execution unit.
 *
 * Create a [`Process`] with [`Process::new`], allocate qubits with
 * [`Process::alloc`], append gate sequences with [`Process::append_block`],
 * and trigger execution with [`Process::execute`] (batch mode) or read
 * results immediately after each gate/measurement in live mode.
 *
 * Each process follows a linear [`ProcessState`] machine and can be executed
 * at most once. Results are cached and retrieved via [`Process::read_sample`],
 * [`Process::read_exp_value`], and [`Process::read_gradient`].
 */
typedef struct Process Process;

/**
 * Configuration of the quantum processing unit (QPU) available to this process.
 */
typedef struct QPUConfig QPUConfig;

/**
 * FFI-compatible function-pointer table for a *live* (gate-at-a-time) execution backend.
 *
 * In live mode each gate is dispatched to the hardware or simulator
 * immediately as it is applied to the process, and measurement results are
 * available without an explicit `ket_process_execute` call.
 *
 * Construct a `CLiveExecution` value on the C side, fill in all function
 * pointers, then call [`ket_quantum_execution_live`] to create the
 * corresponding [`QuantumExecution`] variant.
 *
 * All callbacks must return `0` on success or a non-zero libket error code on
 * failure.
 */
typedef struct CLiveExecution {
  /**
   * Dispatch a single quantum gate to the backend.
   *
   * `json` is a null-terminated JSON string encoding the gate instruction
   * (type, parameters, target qubit, etc.).  Returns `0` on success.
   */
  int32_t (*compute_gate)(const char *json);
  /**
   * Dispatch a sequence of pre-translated native gate instructions to the backend.
   *
   * `json` is a null-terminated JSON string encoding the array of native
   * gate instructions (each a `[name, angles, qubits]` triple). Returns `0`
   * on success.
   */
  int32_t (*compute_native_gates)(const char *json);
  /**
   * Perform a single-shot measurement of `len` qubits.
   *
   * `qubits` points to an array of `len` qubit indices.  On success the
   * measurement result is written to `*result` as a bitmask where bit `i`
   * corresponds to `qubits[i]`.  Returns `0` on success.
   */
  int32_t (*measure)(const uintptr_t *qubits, uintptr_t len, uint64_t *result);
  /**
   * Retrieve the full quantum state (probability amplitudes) of `len` qubits.
   *
   * `qubits` points to an array of `len` qubit indices.  On success
   * `*result_json` is set to a heap-allocated null-terminated JSON string
   * encoding the state; Rust will parse it and then it is the C side's
   * responsibility to manage the lifetime.  Returns `0` on success.
   */
  int32_t (*dump)(const uintptr_t *qubits, uintptr_t len, const char **result_json);
  /**
   * Perform a repeated measurement sample of `len` qubits over `shots` shots.
   *
   * `qubits` points to an array of `len` qubit indices.  On success
   * `*result_json` is set to a heap-allocated null-terminated JSON string
   * with the sample counts.  Returns `0` on success.
   */
  int32_t (*sample)(const uintptr_t *qubits,
                    uintptr_t len,
                    uintptr_t shots,
                    const char **result_json);
  /**
   * Compute the expectation value of a Hamiltonian on the current quantum state.
   *
   * `hamiltonian_json` is a null-terminated JSON string encoding the
   * Hamiltonian.  On success `*result` is set to the computed expectation
   * value.  Returns `0` on success.
   */
  int32_t (*exp_value)(const char *hamiltonian_json, double *result);
} CLiveExecution;

/**
 * FFI-compatible function-pointer table for translating abstract gates into
 * backend-native instructions.
 *
 * The compiler calls these callbacks during the decomposition and routing
 * passes to convert high-level gate representations into the instruction set
 * understood by the target hardware or simulator.
 *
 * Construct a `CNativeGateSet` value on the C side, fill in all function
 * pointers, then pass it to [`ket_quantum_execution_batch`].  Passing `NULL`
 * instead uses identity translation (no decomposition).
 *
 * All callbacks must return `0` on success or a non-zero libket error code on
 * failure.
 */
typedef struct CNativeGateSet {
  /**
   * Translate a single-qubit gate (given as a 2×2 unitary matrix) into
   * native gate instructions.
   *
   * `gate_json` is a null-terminated JSON string encoding the gate as a
   * flat 2×2 complex matrix `[[re,im,re,im],[re,im,re,im]]`.  `target` is
   * the target qubit index.  On success `*native_gate_json` is set to a
   * heap-allocated null-terminated JSON string encoding the native gate
   * sequence.  Returns `0` on success.
   */
  int32_t (*translate)(const char *gate_json, uintptr_t target, const char **native_gate_json);
  /**
   * Produce the native gate sequence that implements a CNOT between two qubits.
   *
   * The first argument is the control qubit index and the second is the
   * target qubit index.  On success `*native_gate_json` is set to a
   * heap-allocated null-terminated JSON string encoding the native CNOT
   * decomposition.  Returns `0` on success.
   */
  int32_t (*cnot)(uintptr_t control, uintptr_t target, const char **native_gate_json);
} CNativeGateSet;

/**
 * FFI-compatible function-pointer table for a *batch* execution backend.
 *
 * In batch mode the entire compiled circuit is serialized and handed to the
 * backend in a single call when `ket_process_execute` is invoked, rather than
 * dispatching gate-by-gate.
 *
 * Construct a `CBatchExecution` value on the C side, fill in all function
 * pointers, then call [`ket_quantum_execution_batch`] to create the
 * corresponding [`QuantumExecution`] variant.
 *
 * All callbacks must return `0` on success or a non-zero libket error code on
 * failure.
 */
typedef struct CBatchExecution {
  int32_t (*sample)(const char *gates_json,
                    const uintptr_t *qubits_to_sample,
                    uintptr_t qubits_to_sample_len,
                    uintptr_t shots,
                    const char **sample_json);
  int32_t (*exp_value)(const char *gates_json, const char *hamiltonian_list_json, double *result);
  int32_t (*sample_native)(const char *gates_json,
                           const uintptr_t *qubits_to_sample,
                           uintptr_t qubits_to_sample_len,
                           uintptr_t shots,
                           const char **sample_json);
  int32_t (*exp_value_native)(const char *gates_json,
                              const char *hamiltonian_list_json,
                              double *result);
  int32_t (*gradient)(const char *gates_json,
                      const char *hamiltonian_list_json,
                      double *exp_result,
                      const char **grad_json);
} CBatchExecution;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Retrieves the human-readable error message for a given integer error code.
 *
 * On success the function heap-allocates a null-terminated C string and
 * writes its address into `*error_message`. The caller must free this string
 * with [`ket_string_delete`] when done.
 *
 * # Returns
 * `0` on success; a non-zero error code if the C string could not be
 * allocated (e.g., if the message contains interior null bytes).
 */
int32_t ket_error_message(int32_t error_code, const char **error_message);

/**
 * Frees a heap-allocated C string that was previously written by Libket.
 *
 * Every `const char **` output parameter written by Libket functions
 * (`ket_error_message`, `ket_process_gates_json`, `ket_process_dump`, etc.)
 * points to memory that must be released through this function. Passing a
 * pointer that was not produced by Libket, or freeing the same pointer twice,
 * is undefined behaviour.
 *
 * # Returns
 * Always returns `0`.
 */
int32_t ket_string_delete(char *ptr);

/**
 * Returns a pointer and byte-length for a static build-information string.
 *
 * The string is embedded at compile time and contains the crate name,
 * version, Rust compiler version, and compilation target triple. It is
 * **not** null-terminated; use `size` to bound reads. The pointer is valid
 * for the lifetime of the program and must **not** be freed.
 *
 * # Parameters
 * - `msg` : Set to the address of the first byte of the build-info string.
 * - `size`: Set to the byte length of the build-info string.
 *
 * # Returns
 * Always returns `0`.
 */
int32_t ket_build_info(const uint8_t **msg, uintptr_t *size);

/**
 * Allocates a new, empty [`BasicBlock`] and writes its pointer to `block`.
 */
int32_t ket_block_new(struct BasicBlock **block);

/**
 * Deallocates the memory associated with a [`BasicBlock`].
 */
int32_t ket_block_delete(struct BasicBlock *block);

/**
 * Appends a new quantum gate (provided as a JSON payload) to a [`BasicBlock`].
 */
int32_t ket_block_append_gate(struct BasicBlock *block, const char *gate_json, uintptr_t target);

/**
 * Creates a new [`BasicBlock`] that is the adjoint (inverse) of the provided block.
 */
int32_t ket_block_inverse(const struct BasicBlock *block, struct BasicBlock **inverted_block);

/**
 * Creates a new [`BasicBlock`] resulting from adding control qubits to all gates in the block.
 */
int32_t ket_block_control(const struct BasicBlock *block,
                          const uintptr_t *control_qubits,
                          uintptr_t control_qubits_len,
                          struct BasicBlock **controlled_block);

/**
 * Enables approximate decomposition strategies for all gates in the given [`BasicBlock`].
 */
int32_t ket_block_enable_approximated_decomposition(struct BasicBlock *block);

/**
 * Locks the control qubits of all gates in the given [`BasicBlock`], preventing further additions.
 */
int32_t ket_block_lock_control(struct BasicBlock *block);

/**
 * Appends the contents of `other` block to `block` and consumes `other`.
 */
int32_t ket_block_append_block(struct BasicBlock *block, struct BasicBlock *other);

/**
 * Adds a global phase to the [`BasicBlock`].
 */
int32_t ket_block_add_global_phase(struct BasicBlock *block, double phase);

/**
 * Serializes the properties of the [`BasicBlock`] into a JSON payload and writes it to the buffer.
 */
int32_t ket_block_proprieties_json(const struct BasicBlock *block, const char **proprieties);

/**
 * Marks the [`BasicBlock`] as implementing a diagonal gate.
 */
int32_t ket_block_set_as_diagonal(struct BasicBlock *block);

/**
 * Marks the [`BasicBlock`] as implementing a permutation gate.
 */
int32_t ket_block_set_as_permutation(struct BasicBlock *block);

/**
 * Creates a live-mode [`QuantumExecution`] backed by the provided callback table.
 *
 * In live mode each gate is dispatched to the backend immediately as it is
 * applied to the process. The `decompose` flag controls whether multi-qubit
 * gates are decomposed into single-qubit and two-qubit native gates before
 * being dispatched.
 *
 * # Parameters
 * - `live`: Pointer to a caller-owned [`CLiveExecution`] struct whose
 *   function pointers implement the backend. The struct is cloned; the caller
 *   retains ownership of the original.
 * - `decompose`: If `true`, enables multi-qubit gate decomposition before
 *   dispatching each gate to the backend.
 * - `quantum_execution`: Output: receives the pointer to the newly allocated
 *   [`QuantumExecution`] object. Ownership is transferred to the caller.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_quantum_execution_live(uintptr_t num_qubits,
                                   const struct CLiveExecution *live,
                                   bool decompose,
                                   const struct CNativeGateSet *native_gate_set,
                                   struct QPUConfig **qpu_config);

/**
 * Creates a batch-mode [`QuantumExecution`] backed by the provided callback tables.
 *
 * In batch mode the full compiled circuit is handed to the backend in a
 * single call when `ket_process_execute` is invoked. The `native_gate_set`
 * callback table is used during the decomposition and routing passes to
 * translate abstract gates into hardware-native instructions.
 *
 * # Parameters
 * - `batch`: Pointer to a caller-owned [`CBatchExecution`] struct. The struct
 *   is cloned; the caller retains ownership of the original.
 * - `native_gate_set`: Pointer to a caller-owned [`CNativeGateSet`] struct,
 *   or `NULL` to use identity translation (no decomposition into native gates).
 * - `gradient`: If `true`, enables parameter-shift rule gradient computation
 *   during execution.
 * - `coupling_graph_json`: A null-terminated JSON string that is either
 *   `null` (no connectivity constraint) or a JSON array of `[usize, usize]`
 *   edge pairs describing the QPU coupling graph used for qubit routing.
 * - `quantum_execution`: Output: receives the pointer to the newly allocated
 *   [`QuantumExecution`] object. Ownership is transferred to the caller.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_quantum_execution_batch(uintptr_t num_qubits,
                                    const struct CBatchExecution *batch,
                                    const struct CNativeGateSet *native_gate_set,
                                    bool decompose,
                                    const char *gradient_strategy_json,
                                    const char *coupling_graph_json,
                                    const char *exp_value_strategy_json,
                                    struct QPUConfig **qpu_config);

/**
 * Allocates a new [`Process`] using the provided QPU configuration.
 *
 * `qpu_config` is consumed by this call: ownership is transferred to the
 * newly created `Process` and the pointer must not be used afterwards.
 *
 * # Parameters
 * - `qpu_config`: Pointer to a [`QPUConfig`].
 * - `process`: Output: receives the pointer to the newly allocated
 *   [`Process`]. Must eventually be freed with [`ket_process_delete`].
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_new(struct QPUConfig *qpu_config, struct Process **process);

/**
 * Allocates a new [`QPUConfig`] for a QPU with the specified number of physical qubits.
 *
 * The newly created configuration has **no execution backend** attached. A
 * `QuantumExecution` backend must be set (via `ket_quantum_execution_live`
 * or `ket_quantum_execution_batch`) before passing the config to
 * [`ket_process_new`].
 *
 * # Parameters
 * - `num_qubits`: The number of physical qubits available on the QPU.
 * - `qpu_config`: Output: receives the pointer to the newly allocated
 *   [`QPUConfig`]. Ownership is transferred to the caller until it is
 *   consumed by [`ket_process_new`].
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_config_new(uintptr_t num_qubits, struct QPUConfig **qpu_config);

/**
 * Deallocates the memory associated with a [`Process`].
 *
 * # Parameters
 * - `process`: Pointer to the [`Process`] to free.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_delete(struct Process *process);

/**
 * Allocates a new logical qubit in the process.
 *
 * Returns an error if the process is in a state that does not accept new
 * qubits (e.g., `ProcessTerminated`) or if the QPU qubit limit has been
 * reached (`QubitLimitExceeded`).
 *
 * # Parameters
 * - `process`: The process in which the qubit is allocated.
 * - `qubit`: Output: receives the index of the newly allocated logical qubit.
 *
 * # Returns
 * `0` on success, non-zero error code on failure (including
 * `ProcessTerminated` and `QubitLimitExceeded`).
 */
int32_t ket_process_alloc(struct Process *process, uintptr_t *qubit);

/**
 * Appends a [`BasicBlock`] of gates to the process.
 *
 * The `block` is **consumed** by this call and must not be used afterwards.
 * Returns an error if the process state does not allow gate appending
 * (`GateAppendForbidden`) or if a qubit index referenced in the block is
 * out of range (`QubitIndexOutOfRange`).
 *
 * # Parameters
 * - `process`: The process to which the block is appended.
 * - `block`: The block to consume and append. Ownership is transferred.
 *
 * # Returns
 * `0` on success, non-zero error code on failure (including
 * `GateAppendForbidden` and `QubitIndexOutOfRange`).
 */
int32_t ket_process_append_block(struct Process *process, struct BasicBlock *block);

/**
 * Returns the accumulated gate instructions of the process as a JSON string.
 *
 * The returned string is a JSON array of gate instruction objects representing
 * the gates in the process's main block. The string is heap-allocated and the
 * caller **must** free it with [`super::ket_string_delete`].
 *
 * # Parameters
 * - `process`: The process whose gate list is serialized.
 * - `block`: Output: set to a pointer to a heap-allocated null-terminated
 *   JSON string.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_gates_json(const struct Process *process, char **block);

/**
 * Performs a single-shot mid-circuit measurement of `qubits`.
 *
 * This function is only valid in **live mode**. Any gates accumulated since
 * the last measurement are dispatched to the backend before the measurement
 * is performed.
 *
 * # Parameters
 * - `process`: The process on which the measurement is performed.
 * - `qubits`: Pointer to an array of qubit indices to measure.
 * - `qubits_len`: Number of elements in `qubits`.
 * - `result`: Output: the measurement result encoded as a bitmask
 *   (bit `i` corresponds to `qubits[i]`).
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_measure(struct Process *process,
                            const uintptr_t *qubits,
                            uintptr_t qubits_len,
                            uint64_t *result);

/**
 * Dumps the quantum state of `qubits` (live mode only).
 *
 * Returns the full probability amplitude vector for the specified qubits as a
 * heap-allocated JSON string. The caller **must** free it with
 * [`super::ket_string_delete`].
 *
 * # Parameters
 * - `process`: The process whose state is dumped.
 * - `qubits`: Pointer to an array of qubit indices to dump.
 * - `qubits_len`: Number of elements in `qubits`.
 * - `dump_json`: Output: set to a pointer to a heap-allocated
 *   null-terminated JSON string encoding the quantum state.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_dump(struct Process *process,
                         const uintptr_t *qubits,
                         uintptr_t qubits_len,
                         const char **dump_json);

/**
 * Requests a measurement sample of `qubits` over `shots` repetitions.
 *
 * In **live mode**, the circuit is executed immediately and the result is
 * returned as a heap-allocated JSON string (caller must free with
 * [`super::ket_string_delete`]). In **batch mode**, the request is recorded for the
 * next [`ket_process_execute`] call and `*sample_json` is set to a JSON
 * `null` string; the actual data becomes available after execution and can be
 * retrieved with [`ket_process_read_sample`].
 *
 * # Parameters
 * - `process`: The process on which the sample is requested.
 * - `qubits`: Pointer to an array of qubit indices to sample.
 * - `qubits_len`: Number of elements in `qubits`.
 * - `shots`: Number of repetitions.
 * - `sample_json`: Output: set to a pointer to a heap-allocated
 *   null-terminated JSON string (caller must free with [`super::ket_string_delete`]).
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_sample(struct Process *process,
                           const uintptr_t *qubits,
                           uintptr_t qubits_len,
                           uintptr_t shots,
                           const char **sample_json);

/**
 * Returns a serialized JSON array of expectation values from the most recent execution.
 *
 * If no execution has been performed yet, the returned JSON string contains
 * `null`. The string is heap-allocated; the caller **must** free it with
 * [`super::ket_string_delete`].
 *
 * # Parameters
 * - `process`: The process from which expectation values are read.
 * - `result_json`: Output: set to a pointer to a heap-allocated
 *   null-terminated JSON string.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_read_exp_value(const struct Process *process, const char **result_json);

/**
 * Returns a serialized JSON array of parameter-shift gradients from the most recent execution.
 *
 * If no execution has been performed yet, the returned JSON string contains
 * `null`. The string is heap-allocated; the caller **must** free it with
 * [`super::ket_string_delete`].
 *
 * # Parameters
 * - `process`: The process from which gradients are read.
 * - `result_json`: Output: set to a pointer to a heap-allocated
 *   null-terminated JSON string.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_read_gradient(const struct Process *process, const char **result_json);

/**
 * Registers a new differentiable parameter with the given initial value.
 *
 * The returned `param_index` uniquely identifies this parameter within the
 * process. Use it to construct `Param::Ref` values when building
 * parameterised gate instructions. When the parameter value changes between
 * executions, gates that reference the same index are automatically updated
 * during the next compilation pass.
 *
 * # Parameters
 * - `process`: The process in which the parameter is registered.
 * - `param`: The initial numeric value of the parameter.
 * - `param_index`: Output: receives the index that identifies this parameter
 *   within the process.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_param(struct Process *process, double param, uintptr_t *param_index);

/**
 * Compiles and executes the accumulated circuit.
 *
 * This is the primary entry point for **batch mode** execution. Calling it in
 * live mode returns an `ExplicitExecuteInLiveMode` error. If no pending
 * measurement or expectation-value requests have been recorded,
 * `NoPendingMeasurement` is returned. If the process is already in the
 * `Terminated` state, this call is idempotent and returns success.
 *
 * # Parameters
 * - `process`: The process to compile and execute.
 *
 * # Returns
 * `0` on success, non-zero error code on failure (including
 * `NoPendingMeasurement` and `ExplicitExecuteInLiveMode`).
 */
int32_t ket_process_execute(struct Process *process);

/**
 * Reads the cached sample data from a [`Process`].
 *
 * Returns the sample counts recorded during the most recent execution as a
 * heap-allocated JSON string. If no execution has been performed yet, the
 * returned JSON string contains `null`. The caller **must** free the string
 * with [`super::ket_string_delete`].
 *
 * # Parameters
 * - `process`: The process from which sample data is read.
 * - `sample_json`: Output: set to a pointer to a heap-allocated
 *   null-terminated JSON string.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_read_sample(struct Process *process, const char **sample_json);

/**
 * Requests the expectation value computation for a given [`Hamiltonian`] (passed as JSON) on a [`Process`].
 *
 * In **live mode**, the expectation value is computed immediately:
 * `*some_result` is set to `true` and the value is written to `*result`.
 * In **batch mode**, the request is recorded for the next
 * [`ket_process_execute`] call: `*some_result` is set to `false` and
 * `*result` is left unmodified until execution completes and
 * [`ket_process_read_exp_value`] is called.
 *
 * # Parameters
 * - `process`: The process on which the expectation value is computed.
 * - `hamiltonian_json`: A null-terminated JSON string encoding the
 *   Hamiltonian.
 * - `result`: Output: the computed expectation value (valid only when
 *   `*some_result` is `true`).
 * - `some_result`: Output: set to `true` if the expectation value was
 *   computed immediately (live mode), or `false` if the request was
 *   deferred (batch mode).
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_exp_value(struct Process *process,
                              const char *hamiltonian_json,
                              double *result,
                              bool *some_result);

/**
 * Returns the current status of the process as a static C string.
 *
 * The string is one of:
 * - `"AcceptingGate"`: the process is ready to accept gate blocks.
 * - `"AcceptingHamiltonian"`: all gates have been appended; expectation-value
 *   requests can now be submitted.
 * - `"ReadyToExecute"`: execution has been requested and the circuit is ready.
 * - `"Terminated"`: execution has completed and results are available.
 *
 * The returned pointer points to a static string and must **not** be freed.
 *
 * # Parameters
 * - `process`: The process whose status is queried.
 * - `status`: Output: set to a pointer to a static null-terminated string.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_status(const struct Process *process, const char **status);

/**
 * Registers a new parameter with the given initial value in the process.
 *
 * The parameter's index within the process parameter list is written to `index`.
 *
 * # Parameters
 * - `process`: The process to register the parameter in.
 * - `value`: The initial value of the parameter.
 * - `index`: Output: set to the registered parameter's index.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_process_set_parameter(struct Process *process, double value, uintptr_t *index);

#if defined(FEATURE_QASM)
/**
 * Parses an `OpenQASM` 2.0 source string and returns a heap-allocated
 * [`BasicBlock`] containing the resulting gate sequence.
 *
 * Physical qubit indices are mapped to the `qreg` declarations in the source
 * in order. The total number of entries in `qubits` must be ≥ the sum of all
 * `qreg` sizes declared in `qasm`.
 *
 * `measure` statements are **not** executed. Their qubit-to-cbit mapping is
 * recorded so that subsequent `if` statements are converted to controlled
 * quantum gates instead of classical branches.
 *
 * On success, a newly heap-allocated `BasicBlock` is written to `*block_out`.
 * The caller takes ownership and must eventually free it with
 * [`ket_block_delete`].
 *
 * # Parameters
 * - `qubits`     : Pointer to an array of `size_t` physical qubit indices.
 * - `qubits_len` : Number of entries in `qubits`.
 * - `qasm`       : Null-terminated C string with the `OpenQASM` 2.0 source.
 * - `block_out`  : Output: on success, set to a pointer to the new `BasicBlock`.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_parse_openqasm(const uintptr_t *qubits,
                           uintptr_t qubits_len,
                           const char *qasm,
                           struct BasicBlock **block_out);
#endif

/**
 * Creates a BasicBlock that implements the Quantum Fourier Transform (QFT) on the given qubits.
 *
 * On success, a newly heap-allocated `BasicBlock` is written to `*block_out`.
 * The caller takes ownership and must eventually free it with [`ket_block_delete`].
 *
 * # Parameters
 * - `qubits`     : Pointer to an array of `size_t` physical qubit indices. The MSQ should be at index 0.
 * - `qubits_len` : Number of entries in `qubits`.
 * - `do_swap`    : Boolean flag indicating whether to add SWAP gates at the end of the circuit to reverse the qubit order.
 * - `block_out`  : Output: on success, set to a pointer to the new `BasicBlock`.
 *
 * # Returns
 * `0` on success, non-zero error code on failure.
 */
int32_t ket_qft(const uintptr_t *qubits,
                uintptr_t qubits_len,
                bool do_swap,
                struct BasicBlock **block_out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
