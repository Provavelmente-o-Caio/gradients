// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::missing_safety_doc, clippy::not_unsafe_ptr_arg_deref)]

//! C API for interacting with `BasicBlock` operations.

use std::ffi::{c_char, CString};

use crate::{
    c_api::c_ok,
    error::KetError,
    ir::{block::Block, gate::QuantumGate},
};

/// Allocates a new, empty [`BasicBlock`] and writes its pointer to `block`.
#[no_mangle]
pub extern "C" fn ket_block_new(block: &mut *mut Block) -> i32 {
    *block = Box::into_raw(Box::new(Block::new()));
    c_ok()
}

/// Deallocates the memory associated with a [`BasicBlock`].
#[no_mangle]
pub extern "C" fn ket_block_delete(block: *mut Block) -> i32 {
    let _ = unsafe { *Box::from_raw(block) };
    c_ok()
}

/// Appends a new quantum gate (provided as a JSON payload) to a [`BasicBlock`].
#[no_mangle]
pub extern "C" fn ket_block_append_gate(
    block: &mut Block,
    gate_json: *const c_char,
    target: usize,
) -> i32 {
    let gate = from_json!(gate_json, QuantumGate);
    block.append_gate(gate, target);
    c_ok()
}

/// Creates a new [`BasicBlock`] that is the adjoint (inverse) of the provided block.
#[no_mangle]
pub extern "C" fn ket_block_inverse(block: &Block, inverted_block: &mut *mut Block) -> i32 {
    *inverted_block = Box::into_raw(Box::new(block.inverse()));
    c_ok()
}

/// Creates a new [`BasicBlock`] resulting from adding control qubits to all gates in the block.
#[no_mangle]
pub extern "C" fn ket_block_control(
    block: &Block,
    control_qubits: *const usize,
    control_qubits_len: usize,
    controlled_block: &mut *mut Block,
) -> i32 {
    let qubits = unsafe { std::slice::from_raw_parts(control_qubits, control_qubits_len) };
    match block.control(qubits) {
        Ok(block) => {
            *controlled_block = Box::into_raw(Box::new(block));
            c_ok()
        }
        Err(err) => err.error_code(),
    }
}

/// Enables approximate decomposition strategies for all gates in the given [`BasicBlock`].
#[no_mangle]
pub extern "C" fn ket_block_enable_approximated_decomposition(block: &mut Block) -> i32 {
    block.enable_approximated_decomposition();
    c_ok()
}

/// Locks the control qubits of all gates in the given [`BasicBlock`], preventing further additions.
#[no_mangle]
pub extern "C" fn ket_block_lock_control(block: &mut Block) -> i32 {
    block.lock_control();
    c_ok()
}

/// Appends the contents of `other` block to `block` and consumes `other`.
#[no_mangle]
pub extern "C" fn ket_block_append_block(block: &mut Block, other: *mut Block) -> i32 {
    let other = unsafe { Box::from_raw(other) };
    block.append_block(*other);
    c_ok()
}

/// Adds a global phase to the [`BasicBlock`].
#[no_mangle]
pub extern "C" fn ket_block_add_global_phase(block: &mut Block, phase: f64) -> i32 {
    block.add_global_phase(phase);
    c_ok()
}

/// Serializes the properties of the [`BasicBlock`] into a JSON payload and writes it to the buffer.
#[no_mangle]
pub extern "C" fn ket_block_proprieties_json(
    block: &Block,
    proprieties: &mut *const c_char,
) -> i32 {
    *proprieties = to_json!(&block.flatten().unwrap().qubits_op).into_raw();

    c_ok()
}

/// Marks the [`BasicBlock`] as implementing a diagonal gate.
#[no_mangle]
pub extern "C" fn ket_block_set_as_diagonal(block: &mut Block) -> i32 {
    block.set_as_diagonal();

    c_ok()
}

/// Marks the [`BasicBlock`] as implementing a permutation gate.
#[no_mangle]
pub extern "C" fn ket_block_set_as_permutation(block: &mut Block) -> i32 {
    block.set_as_permutation();

    c_ok()
}
