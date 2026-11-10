// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Libket: a quantum circuit compiler library.
//!
//! Libket translates high-level quantum gate sequences into hardware-native
//! instruction streams. It provides:
//!
//! - A rich intermediate representation ([`ir`]) with algebraic gate
//!   optimisations and multi-controlled gate decomposition.
//! - Pluggable execution backends ([`execution`]) for both live (mid-circuit
//!   feedback) and batch (deferred compilation) execution modes.
//! - A process abstraction ([`process`]) that drives the full compilation and
//!   execution pipeline.
//! - A structured, FFI-friendly error type ([`error::KetError`]).

use crate::{
    error::KetError,
    ir::{block::BasicBlock, gate::QuantumGate},
};

#[cfg(feature = "c_api")]
pub mod c_api;
mod decompose;
pub mod error;
pub mod execution;
pub mod ir;
mod mapping;
mod matrix;
pub mod process;

#[cfg(feature = "qasm")]
pub mod qasm;

#[cfg(test)]
mod tests;

/// Return a block with a controlled gate.
///
/// # Errors
///
/// Propagates any [`KetError`] that occurs during the `control` operation.
pub fn controlled_gate(
    gate: QuantumGate,
    control: &[usize],
    target: usize,
) -> Result<BasicBlock, KetError> {
    let mut quantum_code = BasicBlock::new();
    quantum_code.append_gate(gate, target);
    quantum_code.control(control)
}

/// Return a block with a SWAP gate.
///
/// # Errors
///
/// Propagates any [`KetError`] that occurs during the `controlled_gate` operation.
pub fn swap_gate(a: usize, b: usize) -> Result<BasicBlock, KetError> {
    let mut quantum_code = BasicBlock::new();
    quantum_code.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
    quantum_code.append_block(controlled_gate(QuantumGate::PauliX, &[b], a)?);
    quantum_code.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
    Ok(quantum_code)
}

/// Return a block with the compute-action-uncompute.
///
/// # Errors
///
/// Propagates any [`KetError`] that occurs during `compute` or `action` block generation.
pub fn compute_action_uncompute_gate<Fc, Fa>(
    mut compute: Fc,
    mut action: Fa,
) -> Result<BasicBlock, KetError>
where
    Fc: FnMut(BasicBlock) -> Result<BasicBlock, KetError>,
    Fa: FnMut(BasicBlock) -> Result<BasicBlock, KetError>,
{
    let mut quantum_code = BasicBlock::new();

    let compute = compute(BasicBlock::new())?;
    let uncompute = compute.inverse();

    let action = action(BasicBlock::new())?;

    quantum_code.append_block(compute);
    quantum_code.append_block(action);
    quantum_code.append_block(uncompute);

    Ok(quantum_code)
}

/// Constructs a Quantum Fourier Transform (QFT) circuit over the specified qubits.
///
/// The QFT is implemented recursively using Hadamard and controlled-phase gates.
///
/// # Arguments
///
/// * `qubits` - A slice of qubit indices over which to apply the QFT. The Most Significant Qubit (MSQ) should be at index 0 and the Least Significant Qubit (LSQ) at the last index.
/// * `do_swap` - If `true`, append SWAP gates at the end of the circuit to reverse the order of the qubits. This is standard in QFT implementations to ensure the output order matches the input order.
///
/// # Returns
///
/// Returns a `Result` containing the `BasicBlock` with the QFT instructions, or a `KetError` if an error occurs.
pub fn qft(qubits: &[usize], do_swap: bool) -> Result<crate::ir::block::Block, KetError> {
    let mut block = crate::ir::block::Block::new();

    block.append_template(
        crate::ir::block::TemplateBlock {
            name: "QFT".to_string(),
            qubits: qubits.to_vec(),
            controls: Vec::new(),
            control_locked: false,
            enable_approximated_decomposition: false,
            is_adjoint: false,
            is_self_inverse: false,
            is_diagonal: false,
            is_permutation: false,
        },
        0,
    );

    if do_swap {
        block.append_template(
            crate::ir::block::TemplateBlock {
                name: "QFT_SWAP".to_string(),
                qubits: qubits.to_vec(),
                controls: Vec::new(),
                control_locked: false,
                enable_approximated_decomposition: false,
                is_adjoint: false,
                is_self_inverse: true,
                is_diagonal: false,
                is_permutation: false,
            },
            0,
        );
    }

    Ok(block)
}

pub(crate) fn qft_internal(qubits: &[usize]) -> Result<BasicBlock, KetError> {
    let mut block = BasicBlock::new();
    if qubits.len() == 1 {
        block.append_gate(QuantumGate::Hadamard, qubits[0]);
        Ok(block)
    } else {
        let init = &qubits[..qubits.len() - 1];
        let last = qubits.last().unwrap();

        block.append_gate(QuantumGate::Hadamard, *last);

        for (i, ctrl_qubit) in init.iter().rev().enumerate() {
            block.append_block(controlled_gate(
                QuantumGate::Phase(ir::gate::Param::Value(
                    std::f64::consts::PI / (1 << (i + 1)) as f64,
                )),
                &[*ctrl_qubit],
                *last,
            )?);
        }

        block.append_block(qft_internal(init)?);

        Ok(block)
    }
}
