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

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Create a [`QPUConfig`] for the requested simulator and write the heap
 * pointer into `*qpu_config`.
 *
 * # Parameters
 *
 * * `num_qubits`: number of qubits to allocate.
 * * `simulator`: NUL-terminated C string naming the backend.  Recognised
 *   values (case-insensitive, whitespace ignored): `"dense"`, `"densev2"`,
 *   `"densegpu"`, `"densecuda"`, `"sparse"`, `"sparsev2"`.
 * * `use_live`: `true` for live (gate-by-gate) mode; `false` for batch mode.
 * * `gradient`: enable the parameter-shift-rule gradient strategy (batch
 *   mode only).
 * * `coupling_graph_json`: JSON-encoded `Vec<[usize; 2]>` describing allowed
 *   two-qubit pairs, or a JSON `null` for all-to-all connectivity.
 * * `qpu_config`: out-parameter; set to the newly allocated config on
 *   success.
 *
 * # Returns
 *
 * A `KetError` integer code: `0` on success, non-zero on failure.
 *
 * # Safety
 *
 * * `simulator` must be a valid, NUL-terminated C string.
 * * `coupling_graph_json` must be a valid, NUL-terminated C string containing
 *   either a JSON array of two-element integer arrays or `null`.
 * * `qpu_config` must be a non-null, aligned, writable pointer.
 */
int32_t kbw_make_configuration(uintptr_t num_qubits,
                               const char *simulator,
                               bool use_live,
                               bool decompose,
                               const char *gradient_strategy_json,
                               const char *coupling_graph_json,
                               void **qpu_config);

/**
 * Return a static string describing the KBW build.
 *
 * Sets `*msg` to the UTF-8 bytes of the build-info string and `*size` to
 * its byte length.  The pointer is valid for the lifetime of the process.
 *
 * Returns a `KetError` integer code (always `0` = success).
 */
int32_t kbw_build_info(const uint8_t **msg, uintptr_t *size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
