// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Recursive network decomposition for multi-controlled general unitary gates.
//!
//! The [`network`] function decomposes a gate `U` with `n` controls into a
//! binary tree of smaller controlled operations, using ancilla qubits to
//! "carry" the accumulated control.  Each leaf of the tree is a single-
//! controlled `CU` gate implemented via [`u2::cu2`].

use itertools::chain;
use num::Integer;

use crate::ir::gate::{DecomposedGate, QuantumGate};

use super::{
    u2,
    x::{c2to4x, CXMode},
};

/// Decomposes a multi-controlled gate into a gate network using ancilla qubits.
///
/// # Parameters
/// - `gate`        : the single-qubit gate to apply conditionally.
/// - `control`     : the control qubit indices (`n ≥ 1`).
/// - `aux_qubits` : ancilla qubits available to the network; the required
///   count depends on `cx_mode`.
/// - `target`     : the target qubit index.
/// - `cx_mode`    : whether to group controls into groups of 2 ([`CXMode::C2X`])
///   or 3 ([`CXMode::C3X`]).
/// - `approximated`: whether to permit approximate sub-gate decompositions.
///
/// # Returns
/// A flat list of `(gate, target, optional_control)` primitive gate triples.
pub fn network(
    gate: QuantumGate,
    control: &[usize],
    aux_qubits: &[usize],
    target: usize,
    cx_mode: CXMode,
    approximated: bool,
) -> Vec<DecomposedGate> {
    let n = control.len();
    let n_cx = match cx_mode {
        CXMode::C2X => 2,
        CXMode::C3X => 3,
    };

    if matches!(gate, QuantumGate::PauliX) && n <= n_cx {
        return c2to4x::c1to4x(control, target, approximated);
    } else if n == 1 {
        return u2::cu2(gate.matrix(), control[0], target);
    } else if n == 2 && matches!(cx_mode, CXMode::C3X) {
        return chain!(
            c2to4x::c1to4x(control, aux_qubits[0], true),
            u2::cu2(gate.matrix(), aux_qubits[0], target),
            c2to4x::c1to4x(control, aux_qubits[0], true)
        )
        .collect();
    }

    let (num_groups, rem) = n.div_rem(&n_cx);

    let left_instructions = (0..num_groups)
        .flat_map(|i| c2to4x::c1to4x(&control[n_cx * i..n_cx * i + n_cx], aux_qubits[i], true));

    let new_ctrl = if rem != 0 {
        &control[control.len() - rem..]
            .iter()
            .copied()
            .chain(aux_qubits[..num_groups].iter().copied())
            .collect::<Vec<_>>()
    } else {
        &aux_qubits[..num_groups]
    };

    left_instructions
        .clone()
        .chain(network(
            gate,
            new_ctrl,
            &aux_qubits[num_groups..],
            target,
            cx_mode,
            approximated,
        ))
        .chain(left_instructions.rev().map(|gate| gate.inverse()))
        .collect()
}
