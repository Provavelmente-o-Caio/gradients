// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Multi-controlled Pauli-X decompositions using an ancilla V-chain.
//!
//! V-chain decompositions achieve a low gate count by using a ladder of
//! Toffoli (or C3X) gates to "accumulate" the control conditions onto a sequence
//! of clean or dirty ancilla qubits.

use itertools::chain;

use crate::{decompose::AuxMode, ir::gate::DecomposedGate};

use super::{c2to4x, CXMode};

fn v_chain_action(
    control: &[usize],
    aux_qubit: &[usize],
    target: usize,
    approximated: u8,
    left_right: (bool, bool),
    cx_mode: CXMode,
) -> Vec<DecomposedGate> {
    let n = control.len();
    if n <= match cx_mode {
        CXMode::C2X => 2,
        CXMode::C3X => 3,
    } {
        return c2to4x::c1to4x(
            control,
            target,
            if matches!(cx_mode, CXMode::C3X) {
                false
            } else {
                approximated == 0
            },
        );
    }

    let a = aux_qubit.len();

    let c_n_index = n - match cx_mode {
        CXMode::C2X => 1,
        CXMode::C3X => 2,
    };

    let edge_ctrl: Vec<usize> = control[c_n_index..]
        .iter()
        .copied()
        .chain([*aux_qubit.last().unwrap()])
        .collect();

    let edge = c2to4x::c1to4x(&edge_ctrl, target, approximated == 0);
    let approximated = if approximated != 0 {
        approximated - 1
    } else {
        0
    };

    match left_right {
        (true, true) => chain![
            edge.clone(),
            v_chain_action(
                &control[..c_n_index],
                &aux_qubit[..a - 1],
                *aux_qubit.last().unwrap(),
                approximated,
                left_right,
                cx_mode,
            ),
            edge
        ]
        .collect(),
        (true, false) => chain![
            edge,
            v_chain_action(
                &control[..c_n_index],
                &aux_qubit[..a - 1],
                *aux_qubit.last().unwrap(),
                approximated,
                left_right,
                cx_mode,
            ),
        ]
        .collect(),
        (false, true) => chain![
            v_chain_action(
                &control[..c_n_index],
                &aux_qubit[..a - 1],
                *aux_qubit.last().unwrap(),
                approximated,
                left_right,
                cx_mode,
            ),
            edge
        ]
        .collect(),
        (false, false) => panic!("invalid parameters for v_chain_c2x_action"),
    }
}

/// Decomposes a multi-controlled Pauli-X gate using an ancilla V-chain.
///
/// Builds a ladder of `C²X` (or `C³X` depending on `cx_mode`) gates that uses
/// `aux_qubits` to compute the conjunction of the `control` qubits.
///
/// # Parameters
/// - `control`     : the control qubit indices.
/// - `aux_qubits`  : the ancilla chain; requires `n-2` for `C2X` and `(n-3)/2` for `C3X`.
/// - `target`      : the target qubit index.
/// - `aux_mode`    : whether the ancillae are `Clean` or `Dirty`.
/// - `cx_mode`     : whether to use `C2X` or `C3X` primitives in the chain.
/// - `approximated`: whether to permit approximate sub-decompositions.
pub fn v_chain(
    control: &[usize],
    aux_qubits: &[usize],
    target: usize,
    aux_mode: AuxMode,
    cx_mode: CXMode,
    approximated: bool,
) -> Vec<DecomposedGate> {
    let n = control.len();
    if n <= 3 {
        return c2to4x::c1to4x(control, target, approximated);
    }

    let a = aux_qubits.len();
    let (action, reset) = match aux_mode {
        AuxMode::Clean => ((false, true), (true, false)),
        AuxMode::Dirty => ((true, true), (true, true)),
    };

    chain![
        v_chain_action(
            control,
            aux_qubits,
            target,
            match cx_mode {
                CXMode::C2X => 1,
                CXMode::C3X => 2,
            },
            action,
            cx_mode,
        ),
        v_chain_action(
            &control[..n - match cx_mode {
                CXMode::C2X => 1,
                CXMode::C3X => 2,
            }],
            &aux_qubits[..a - 1],
            aux_qubits[a - 1],
            match cx_mode {
                CXMode::C2X => 0,
                CXMode::C3X => 1,
            },
            reset,
            cx_mode,
        )
    ]
    .collect()
}
