// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! SU(2)-based multi-controlled gate decomposition.
//!
//! Provides two decomposition strategies for an arbitrary multi-controlled
//! single-qubit gate `CⁿU` where `U ∈ SU(2)`:
//!
//! - **Linear depth** ([`linear`]): uses a sequence of `CZ` and `CU` gadgets
//!   that grows linearly in the number of controls.  No ancilla depth overhead.
//! - **Logarithmic depth** ([`decompose`] with [`DepthMode::Log`]): employs
//!   recursive ZYZ splitting, reducing circuit depth at the cost of a higher
//!   total CNOT count.
//!
//! - [`decompose`]: the general implementation for any unitary. It uses a
//!   V-chain construction and is reused by [`crate::decompose::u2::su2_rewrite_hadamard`].

use std::{f64::consts::FRAC_PI_4, f64::consts::PI};

use itertools::chain;

use crate::{
    decompose::{util::matrix_dot, x::c2to4x::cnot},
    ir::gate::{DecomposedGate, Param, QuantumGate},
    matrix::Matrix,
};

use super::{u2::cu2, util::zyz, x, AuxMode, DepthMode};

/// Implements a three-qubit gate: C²XP(a, b, target): a Toffoli-like gadget
/// used inside [`cz_ap`] to build controlled-Z ladders without ancillae.
fn c2xp(a: usize, b: usize, c: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::Hadamard, b).into(),
        (QuantumGate::td(), b).into(),
        cnot(a, b),
        (QuantumGate::t(), b).into(),
        cnot(c, b),
        (QuantumGate::td(), b).into(),
        cnot(a, b),
        (QuantumGate::t(), b).into(),
        (QuantumGate::Hadamard, b).into(),
    ]
}

/// Implements C²IZ(a, b, target): a doubly-controlled iZ gadget using T/T† gates.
fn c2iz(a: usize, b: usize, c: usize) -> Vec<DecomposedGate> {
    vec![
        (QuantumGate::td(), c).into(),
        cnot(a, c),
        (QuantumGate::t(), c).into(),
        cnot(b, c),
        (QuantumGate::td(), c).into(),
        cnot(a, c),
        (QuantumGate::t(), c).into(),
        cnot(b, c),
    ]
}

/// Decomposes `CⁿZ` (multi-controlled Pauli-Z) using ancilla-assisted V-chain.
///
/// `control` holds the control qubits and `aux_qubits` provides the ancilla
/// qubits.  The ancillae must contain `|control| - 1` qubits.
pub fn cz_ap(control: &[usize], aux_qubits: &[usize], target: usize) -> Vec<DecomposedGate> {
    let n = control.len();
    let a = aux_qubits.len();
    if n == 1 {
        vec![
            (QuantumGate::Hadamard, target).into(),
            cnot(control[0], target),
            (QuantumGate::Hadamard, target).into(),
        ]
    } else if n == 2 {
        c2iz(control[0], control[1], target)
    } else {
        chain![
            c2xp(
                *control.last().unwrap(),
                *aux_qubits.last().unwrap(),
                target,
            ),
            cz_ap(
                &control[..n - 1],
                &aux_qubits[..a - 1],
                *aux_qubits.last().unwrap(),
            ),
            c2xp(
                *control.last().unwrap(),
                *aux_qubits.last().unwrap(),
                target,
            )
        ]
        .collect()
    }
}

/// Linear-depth decomposition of `CⁿU` for `U ∈ SU(2)`, using no ancilla.
///
/// Implements the generalised Gray-code / recursive halving scheme from the
/// literature.  The circuit depth grows linearly in `control.len()`.
pub(super) fn linear(gate: QuantumGate, control: &[usize], target: usize) -> Vec<DecomposedGate> {
    let (a4, theta) = match gate {
        QuantumGate::RotationX(theta) => (vec![], theta.value()),
        QuantumGate::RotationY(theta) => {
            (vec![QuantumGate::Hadamard, QuantumGate::s()], theta.value())
        }
        QuantumGate::RotationZ(theta) => (vec![QuantumGate::Hadamard], theta.value()),
        QuantumGate::Hadamard => (vec![QuantumGate::RotationY(Param::Value(FRAC_PI_4))], PI),
        _ => panic!("Not an SU2 gate"),
    };

    let a2 = QuantumGate::RotationX(Param::Value(-theta / 4.0));
    let a3 = a2.inverse();
    let a1: Vec<_> = chain![
        [a3],
        a4.iter()
            .rev()
            .map(super::super::ir::gate::QuantumGate::inverse)
    ]
    .collect();

    let n = control.len() / 2;
    let c1 = &control[..n];
    let c2 = &control[n..];

    let cz_ap_c1_c2 = cz_ap(c1, c2, target);
    let cz_ap_c2_c1 = cz_ap(c2, c1, target);

    chain![
        a4.iter().map(|gate| (*gate, target).into()),
        cz_ap_c1_c2.clone(),
        [(a2, target).into()],
        cz_ap_c2_c1.clone(),
        [(a3, target).into()],
        cz_ap_c1_c2.iter().rev().map(DecomposedGate::inverse),
        [(a2, target).into()],
        cz_ap_c2_c1.iter().rev().map(DecomposedGate::inverse),
        a1.iter().map(|gate| (*gate, target).into()),
    ]
    .collect()
}

/// Logarithmic-depth decomposition of `CⁿU` via ZYZ splitting.
///
/// Decomposes `U` into `A·B` sub-unitaries using the ZYZ decomposition and
/// recurses on `C^{n-1}X` with a dirty ancilla, halving the depth at each
/// level.
fn log(
    matrix: Matrix,
    control: &[usize],
    target: usize,
    approximated: bool,
) -> Vec<DecomposedGate> {
    let n = control.len();
    let (_, beta, gamma, delta) = zyz(matrix);

    let c = QuantumGate::RotationZ(Param::Value((delta - beta) / 2.0)).matrix();
    let b = matrix_dot(
        &QuantumGate::RotationZ(Param::Value(-(delta + beta) / 2.0)).matrix(),
        &QuantumGate::RotationY(Param::Value(-gamma / 2.0)).matrix(),
    );
    let a = matrix_dot(
        &QuantumGate::RotationY(Param::Value(gamma / 2.0)).matrix(),
        &QuantumGate::RotationZ(Param::Value(beta)).matrix(),
    );

    cu2(c, control[n - 1], target)
        .into_iter()
        .chain(x::single_aux::decompose(
            &control[..n - 1],
            control[n - 1],
            target,
            AuxMode::Dirty,
            DepthMode::Log,
            approximated,
        ))
        .chain(cu2(b, control[n - 1], target))
        .chain(x::single_aux::decompose(
            &control[..n - 1],
            control[n - 1],
            target,
            AuxMode::Dirty,
            DepthMode::Log,
            approximated,
        ))
        .chain(cu2(a, control[n - 1], target))
        .collect()
}

/// Decomposes a multi-controlled SU(2) gate.
///
/// Selects between the logarithmic-depth ([`DepthMode::Log`]) and
/// linear-depth ([`DepthMode::Linear`]) strategies.
///
/// # Parameters
/// - `gate`       : the single-qubit gate to decompose.
/// - `control`    : control qubit indices.
/// - `target`     : target qubit index.
/// - `depth_mode` : depth/CNOT-count trade-off.
/// - `approximated`: whether approximate sub-decompositions are permitted.
pub fn decompose(
    gate: QuantumGate,
    control: &[usize],
    target: usize,
    depth_mode: DepthMode,
    approximated: bool,
) -> Vec<DecomposedGate> {
    match depth_mode {
        DepthMode::Log => log(gate.su2_matrix(), control, target, approximated),
        DepthMode::Linear => linear(gate, control, target),
    }
}
