// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Controlled single-qubit (U(2)) decomposition utilities.
//!
//! Provides:
//! - [`cu2`]: exact one-control-qubit decomposition via ZYZ.
//! - [`linear_depth`]: linear-depth multi-controlled decomposition requiring
//!   no ancilla qubits.
//! - [`su2_rewrite_hadamard`]: special-case multi-controlled Hadamard
//!   decomposition using one ancilla and the SU(2) ZYZ ladder.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use itertools::chain;

use crate::{
    decompose::{su2::cz_ap, x::c2to4x::cnot},
    ir::gate::{DecomposedGate, Param, QuantumGate},
    matrix::Matrix,
};

use super::util::{exp_gate, zyz};

/// Decomposes a single-controlled U(2) gate into CNOT + single-qubit gates.
///
/// Uses the ZYZ decomposition: `U = e^{iα} · Rz(β) · Ry(γ) · Rz(δ)`, then
/// rewrites `CU` as four single-qubit gates and two CNOTs.
///
/// # Parameters
/// - `matrix` : the 2×2 unitary matrix to apply.
/// - `control`: the control qubit.
/// - `target` : the target qubit.
pub fn cu2(matrix: Matrix, control: usize, target: usize) -> Vec<DecomposedGate> {
    let (alpha, beta, gamma, delta) = zyz(matrix);

    vec![
        (QuantumGate::Phase(Param::Value(alpha)), control).into(),
        (
            QuantumGate::RotationZ(Param::Value((delta - beta) / 2.0)),
            target,
        )
            .into(),
        cnot(control, target),
        (
            QuantumGate::RotationZ(Param::Value(-(delta + beta) / 2.0)),
            target,
        )
            .into(),
        (QuantumGate::RotationY(Param::Value(-gamma / 2.0)), target).into(),
        cnot(control, target),
        (QuantumGate::RotationY(Param::Value(gamma / 2.0)), target).into(),
        (QuantumGate::RotationZ(Param::Value(beta)), target).into(),
    ]
}

/// Generates one step of the multi-controlled U(2) linear-depth ladder.
///
/// Used internally by [`linear_depth`] to build the forward and reverse
/// halves of the recursion.  Each step emits a set of `CU` gates whose
/// combined action approximates the `n`-th root of `matrix`.
fn mcu2_step(matrix: Matrix, qubits: &[usize], first: bool, inverse: bool) -> Vec<DecomposedGate> {
    let mut instructions = Vec::new();

    let start = usize::from(inverse);

    let mut qubit_pairs: Vec<(usize, usize)> = (0..qubits.len())
        .enumerate()
        .flat_map(|(i, t)| {
            if i > start {
                (start..i).map(|c| (c, t)).collect::<Vec<(usize, usize)>>()
            } else {
                vec![]
            }
        })
        .collect();

    qubit_pairs.sort_by_key(|(c, t)| c + t);
    if !inverse {
        qubit_pairs.reverse();
    }

    for (control, target) in qubit_pairs {
        let exponent = i32::try_from(target - control).unwrap();
        let exponent = if control == 0 { exponent - 1 } else { exponent };
        let param = 2.0_f64.powi(exponent);
        let signal = control == 0 && !first;
        let signal = signal ^ inverse;
        if target == qubits.len() - 1 && first {
            let gate = exp_gate(matrix, 1.0 / param, signal);
            instructions.extend(cu2(gate, qubits[control], qubits[target]));
        } else {
            instructions.extend(cu2(
                QuantumGate::RotationX(Param::Value(
                    std::f64::consts::PI * (if signal { -1.0 } else { 1.0 }) / param,
                ))
                .matrix(),
                qubits[control],
                qubits[target],
            ));
        }
    }

    instructions
}

/// Decomposes `CⁿU` into a linear-depth CNOT-optimal circuit without ancillae.
///
/// The algorithm emits four passes of [`mcu2_step`] gadgets: two forward and
/// two reverse, achieving exact multi-controlled-U application in O(n) depth.
///
/// # Parameters
/// - `gate`   : the gate to apply under multi-control.
/// - `control`: the control qubit indices.
/// - `target` : the target qubit index.
pub fn linear_depth(gate: QuantumGate, control: &[usize], target: usize) -> Vec<DecomposedGate> {
    let matrix = gate.matrix();
    let mut control_target = control.to_vec();
    control_target.push(target);

    chain![
        mcu2_step(matrix, &control_target, true, false),
        mcu2_step(matrix, &control_target, true, true),
        mcu2_step(matrix, control, false, false),
        mcu2_step(matrix, control, false, true),
    ]
    .collect()
}

/// Doubly-controlled `(CZ)^{1/2}` gadget applied to two targets simultaneously,
/// used as a building block inside [`su2_rewrite_hadamard`].
fn czz_ap(
    control: &[usize],
    aux_qubits: &[usize],
    target1: usize,
    target2: usize,
) -> Vec<DecomposedGate> {
    chain![
        [cnot(target2, target1)],
        cz_ap(control, aux_qubits, target1),
        [cnot(target2, target1)],
    ]
    .collect()
}

/// Decomposes a multi-controlled Hadamard gate using one ancilla qubit.
///
/// Rewrites `CⁿH` by expressing it as two simultaneous SU(2) evolutions on
/// `target` and `aux_qubit`, coupled via the `czz_ap` gadget.  The ancilla
/// must be in an arbitrary (possibly dirty) state.
///
/// # Parameters
/// - `control`  : the control qubit indices.
/// - `aux_qubit`: one dirty ancilla qubit.
/// - `target`   : the target qubit for the Hadamard.
pub fn su2_rewrite_hadamard(
    control: &[usize],
    aux_qubit: usize,
    target: usize,
) -> Vec<DecomposedGate> {
    let a4_1 = QuantumGate::RotationY(Param::Value(FRAC_PI_4));
    let a4_2 = QuantumGate::Hadamard;
    let theta1 = PI;
    let theta2 = -2.0 * FRAC_PI_2;

    let a2_1 = QuantumGate::RotationX(Param::Value(-theta1 / 4.0));
    let a3_1 = a2_1.inverse();
    let a1_1 = [a3_1, a4_1.inverse()];

    let a2_2 = QuantumGate::RotationX(Param::Value(-theta2 / 4.0));
    let a3_2 = a2_2.inverse();
    let a1_2 = [a3_2, a4_2.inverse()];

    let n = control.len() / 2;
    let c1 = &control[..n];
    let c2 = &control[n..];

    let czz_c1_c2_ap = czz_ap(c1, c2, target, aux_qubit);
    let czz_c2_c1_ap = czz_ap(c2, c1, target, aux_qubit);

    chain![
        [(a4_1, target).into(), (a4_2, aux_qubit).into()],
        czz_c1_c2_ap.clone(),
        [(a2_1, target).into(), (a2_2, aux_qubit).into()],
        czz_c2_c1_ap.clone(),
        [(a3_1, target).into(), (a3_2, aux_qubit).into()],
        czz_c1_c2_ap.iter().rev().map(DecomposedGate::inverse),
        [(a2_1, target).into(), (a2_2, aux_qubit).into()],
        czz_c2_c1_ap.iter().rev().map(DecomposedGate::inverse),
        a1_1.iter().map(|gate| (*gate, target).into()),
        a1_2.iter().map(|gate| (*gate, aux_qubit).into()),
    ]
    .collect()
}
