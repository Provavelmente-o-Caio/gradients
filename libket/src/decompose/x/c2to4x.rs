// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Exact and approximate fixed-size multi-controlled Pauli-X decompositions.
//!
//! Provides the primitive multi-controlled NOT gates (C2X, C3X, C4X) used as
//! base cases for the larger recursive decompositions.
//!
//! - `cnot`  : fundamental CNOT gate.
//! - `c2x`   : Toffoli gate (exact and approximate).
//! - `c3x`   : 3-controlled NOT (exact and approximate).
//! - `c4x`   : 4-controlled NOT.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, FRAC_PI_8};

use itertools::chain;

use crate::ir::gate::{DecomposedGate, Param, QuantumGate};

/// Emits a single primitive CNOT instruction.
pub const fn cnot(control: usize, target: usize) -> DecomposedGate {
    DecomposedGate::CNOT(control, target)
}

fn c2x(control_0: usize, control_1: usize, target: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::Hadamard, target).into(),
        cnot(control_1, target),
        (QuantumGate::td(), target).into(),
        cnot(control_0, target),
        (QuantumGate::t(), target).into(),
        cnot(control_1, target),
        (QuantumGate::td(), target).into(),
        cnot(control_0, target),
        (QuantumGate::t(), control_1).into(),
        (QuantumGate::t(), target).into(),
        cnot(control_0, control_1),
        (QuantumGate::Hadamard, target).into(),
        (QuantumGate::t(), control_0).into(),
        (QuantumGate::td(), control_1).into(),
        cnot(control_0, control_1),
    ]
}

fn c3x(control_0: usize, control_1: usize, control_2: usize, target: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::Hadamard, target).into(),
        (QuantumGate::sqrt_t(), control_0).into(),
        (QuantumGate::sqrt_t(), control_1).into(),
        (QuantumGate::sqrt_t(), control_2).into(),
        (QuantumGate::sqrt_t(), target).into(),
        cnot(control_0, control_1),
        (QuantumGate::sqrt_td(), control_1).into(),
        cnot(control_0, control_1),
        cnot(control_1, control_2),
        (QuantumGate::sqrt_td(), control_2).into(),
        cnot(control_0, control_2),
        (QuantumGate::sqrt_t(), control_2).into(),
        cnot(control_1, control_2),
        (QuantumGate::sqrt_td(), control_2).into(),
        cnot(control_0, control_2),
        cnot(control_2, target),
        (QuantumGate::sqrt_td(), target).into(),
        cnot(control_1, target),
        (QuantumGate::sqrt_t(), target).into(),
        cnot(control_2, target),
        (QuantumGate::sqrt_td(), target).into(),
        cnot(control_0, target),
        (QuantumGate::sqrt_t(), target).into(),
        cnot(control_2, target),
        (QuantumGate::sqrt_td(), target).into(),
        cnot(control_1, target),
        (QuantumGate::sqrt_t(), target).into(),
        cnot(control_2, target),
        (QuantumGate::sqrt_td(), target).into(),
        cnot(control_0, target),
        (QuantumGate::Hadamard, target).into(),
    ]
}

fn c4x(
    control_0: usize,
    control_1: usize,
    control_2: usize,
    control_3: usize,
    target: usize,
) -> Vec<DecomposedGate> {
    let c3x_ap = c3x_ap(control_0, control_1, control_2, control_3);

    chain![
        [(QuantumGate::Hadamard, target).into()],
        cp(FRAC_PI_2, control_3, target),
        [(QuantumGate::Hadamard, target).into()],
        c3x_ap.clone(),
        [(QuantumGate::Hadamard, target).into()],
        cp(-FRAC_PI_2, control_3, target),
        [(QuantumGate::Hadamard, target).into()],
        c3x_ap.iter().rev().map(DecomposedGate::inverse),
        c3sx(control_0, control_1, control_2, target)
    ]
    .collect()
}

/// Emits a controlled-phase `CP(λ)` gadget.
pub fn cp(lambda: f64, control: usize, target: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::Phase(Param::Value(lambda / 2.0)), control).into(),
        cnot(control, target),
        (QuantumGate::Phase(Param::Value(-lambda / 2.0)), target).into(),
        cnot(control, target),
        (QuantumGate::Phase(Param::Value(lambda / 2.0)), target).into(),
    ]
}

fn c3sx(
    control_0: usize,
    control_1: usize,
    control_2: usize,
    target: usize,
) -> Vec<DecomposedGate> {
    chain![
        [(QuantumGate::Hadamard, target).into()],
        cp(FRAC_PI_8, control_0, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_0, control_1),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(-FRAC_PI_8, control_1, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_0, control_1),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(FRAC_PI_8, control_1, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_1, control_2),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(-FRAC_PI_8, control_2, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_0, control_2),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(FRAC_PI_8, control_2, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_1, control_2),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(-FRAC_PI_8, control_2, target),
        [
            (QuantumGate::Hadamard, target).into(),
            cnot(control_0, control_2),
            (QuantumGate::Hadamard, target).into()
        ],
        cp(FRAC_PI_8, control_2, target),
        [(QuantumGate::Hadamard, target).into()],
    ]
    .collect()
}

fn c3x_ap(
    control_0: usize,
    control_1: usize,
    control_2: usize,
    target: usize,
) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::Hadamard, target).into(),
        (QuantumGate::t(), target).into(),
        cnot(control_2, target),
        (QuantumGate::td(), target).into(),
        (QuantumGate::Hadamard, target).into(),
        cnot(control_0, target),
        (QuantumGate::t(), target).into(),
        cnot(control_1, target),
        (QuantumGate::td(), target).into(),
        cnot(control_0, target),
        (QuantumGate::t(), target).into(),
        cnot(control_1, target),
        (QuantumGate::td(), target).into(),
        (QuantumGate::Hadamard, target).into(),
        (QuantumGate::t(), target).into(),
        cnot(control_2, target),
        (QuantumGate::td(), target).into(),
        (QuantumGate::Hadamard, target).into(),
    ]
}

fn c2x_ap(control_0: usize, control_1: usize, target: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::RotationY(Param::Value(-FRAC_PI_4)), target).into(),
        cnot(control_0, target),
        (QuantumGate::RotationY(Param::Value(-FRAC_PI_4)), target).into(),
        cnot(control_1, target),
        (QuantumGate::RotationY(Param::Value(FRAC_PI_4)), target).into(),
        cnot(control_0, target),
        (QuantumGate::RotationY(Param::Value(FRAC_PI_4)), target).into(),
    ]
}

/// Dispatches to the appropriate primitive multi-controlled X decomposition.
///
/// Supports 1 to 4 control qubits. If `approximated` is true, allows the use
/// of approximate C2X and C3X decompositions to reduce gate count at the cost
/// of circuit error.
///
/// # Panics
/// Panics if the number of controls is 0 or greater than 4.
pub fn c1to4x(control: &[usize], target: usize, approximated: bool) -> Vec<DecomposedGate> {
    match control.len() {
        1 => vec![cnot(control[0], target)],
        2 => {
            if approximated {
                c2x_ap(control[0], control[1], target)
            } else {
                c2x(control[0], control[1], target)
            }
        }
        3 => {
            if approximated {
                c3x_ap(control[0], control[1], control[2], target)
            } else {
                c3x(control[0], control[1], control[2], target)
            }
        }
        4 => c4x(control[0], control[1], control[2], control[3], target),
        n => panic!("C{n}X not supported"),
    }
}
