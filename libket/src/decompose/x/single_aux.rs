// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Multi-controlled Pauli-X decompositions using a single ancilla qubit.
//!
//! Provides two single-ancilla strategies for `CⁿX`:
//! - **Linear depth** (`linear`): uses the SU(2) decomposition of a full
//!   rotation to borrow an ancilla.
//! - **Logarithmic depth** (`log`): splits the controls into sub-registers
//!   to parallelise the decomposition tree, reducing depth at the cost of gate
//!   count.

use std::f64::consts::PI;

use itertools::chain;
use num::integer::Roots;

use crate::{
    decompose::{su2, AuxMode, DepthMode},
    ir::gate::{DecomposedGate, Param, QuantumGate},
};

use super::{c2to4x, v_chain::v_chain, CXMode};

fn linear(
    control: &[usize],
    aux_qubit: usize,
    target: usize,
    approximated: bool,
) -> Vec<DecomposedGate> {
    chain![
        [(QuantumGate::Hadamard, target).into()],
        su2::decompose(
            QuantumGate::RotationX(Param::Value(2.0 * PI)),
            chain![control.iter().copied(), [target]]
                .collect::<Vec<_>>()
                .as_ref(),
            aux_qubit,
            DepthMode::Linear,
            approximated
        ),
        [(QuantumGate::Hadamard, target).into()],
    ]
    .collect()
}

pub(super) fn log3_registers(control: &[usize]) -> (Vec<&[usize]>, &[usize], &[usize]) {
    let n = control.len();
    let p = n.sqrt();
    let r0 = &control[..2 * p];
    let mut registers = vec![r0];
    for i in 1..=p {
        let begin = (1 + i) * p;
        let end = (2 + i) * p;
        if end < n {
            registers.push(&control[begin..end]);
        } else {
            registers.push(&control[begin..]);
            assert!(!registers.last().unwrap().is_empty());
            break;
        }
    }
    let b = registers.len() - 1;
    let r0star = &r0[..b];
    let r0b = &r0[b..];

    (registers, r0star, r0b)
}

pub(super) fn log(
    control: &[usize],
    aux_qubit: usize,
    more_aux: Option<usize>,
    target: usize,
    aux_mode: AuxMode,
    approximated: bool,
) -> Vec<DecomposedGate> {
    let n = control.len();
    if n == 4 {
        return if let Some(more_aux) = more_aux {
            v_chain(
                control,
                &[aux_qubit, more_aux],
                target,
                AuxMode::Dirty,
                CXMode::C2X,
                approximated,
            )
        } else {
            return c2to4x::c1to4x(control, target, approximated);
        };
    } else if n <= 3 {
        return c2to4x::c1to4x(control, target, approximated);
    }

    let (registers, r0star, r0b) = log3_registers(control);
    let b = r0star.len();

    let c_r0_a = log(
        registers[0],
        target,
        Some(registers[1][0]),
        aux_qubit,
        AuxMode::Dirty,
        true,
    );

    let mut prod_i = Vec::new();
    for i in 1..=b {
        prod_i.extend(log(
            registers[i],
            r0b[i - 1],
            None,
            r0star[i - 1],
            AuxMode::Dirty,
            true,
        ));
    }

    let x_r0star: Vec<_> = r0star
        .iter()
        .copied()
        .map(|qubit| (QuantumGate::PauliX, qubit).into())
        .collect();

    let r0star_aux: Vec<_> = r0star.iter().copied().chain([aux_qubit]).collect();

    let c_r0star_aux_t = log(
        &r0star_aux,
        r0b[0],
        Some(registers[1][0]),
        target,
        aux_mode,
        false,
    );

    let clean_aux_gates = chain![
        c_r0_a.clone(),
        prod_i.clone(),
        x_r0star.clone(),
        c_r0star_aux_t.clone(),
        x_r0star.clone(),
        prod_i.clone(),
        c_r0_a,
    ];

    match aux_mode {
        AuxMode::Clean => clean_aux_gates.collect(),
        AuxMode::Dirty => chain![
            clean_aux_gates,
            prod_i.clone(),
            x_r0star.clone(),
            c_r0star_aux_t,
            x_r0star,
            prod_i,
        ]
        .collect(),
    }
}

/// Decomposes a multi-controlled Pauli-X gate using exactly one ancilla qubit.
///
/// Selects between the logarithmic-depth and linear-depth strategies based
/// on `depth_mode`.  The linear strategy requires a clean ancilla, while the
/// logarithmic strategy can accommodate a dirty ancilla depending on `aux_mode`.
pub fn decompose(
    control: &[usize],
    aux_qubit: usize,
    target: usize,
    aux_mode: AuxMode,
    depth_mode: DepthMode,
    approximated: bool,
) -> Vec<DecomposedGate> {
    match depth_mode {
        DepthMode::Log => log(control, aux_qubit, None, target, aux_mode, approximated),
        DepthMode::Linear => linear(control, aux_qubit, target, approximated),
    }
}
