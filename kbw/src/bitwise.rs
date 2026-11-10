// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! Bit-manipulation helpers shared by all simulator backends.
//!
//! Two representations of qubit indices are used throughout the codebase:
//!
//! * **Scalar `usize`**: used by the dense backends where the entire basis
//!   state fits in a single machine word (at most 32 qubits).
//! * **`Vec<u64>`**: used by the sparse backends where qubit indices can
//!   exceed 64, packing bits into 64-bit words (one word per 64 qubits).

use std::ops::{BitOrAssign, Shl};

use num::Integer;

/// Build a control bitmask from a list of control qubit indices.
///
/// The resulting mask has a `1` in position `c` for each control qubit `c`.
/// Used by dense simulators to check all controls with a single `&` operation.
pub(crate) fn get_ctrl_mask<T: Integer + BitOrAssign + Shl<usize, Output = T>>(
    control: &[usize],
) -> T {
    let mut control_mask = T::zero();
    for c in control {
        control_mask |= T::one() << *c;
    }
    control_mask
}

/// Flip bit `index` in `state` (scalar, dense representation).
///
/// Equivalent to applying a NOT gate on qubit `index` in the index space.
pub(crate) const fn bit_flip(state: usize, index: usize) -> usize {
    state ^ (1 << index)
}

use smallvec::SmallVec;

pub type StateKey = SmallVec<[u64; 2]>;

/// Flip bit `index` in `state` (packed `StateKey`, sparse representation).
///
/// `index` is the global qubit index; it is decomposed into an outer word
/// index and an inner bit index automatically.
pub(crate) fn bit_flip_vec(mut state: StateKey, index: usize) -> StateKey {
    let (outer_index, inner_index) = index.div_mod_floor(&64);
    state[outer_index] = bit_flip(usize::try_from(state[outer_index]).unwrap(), inner_index) as u64;
    state
}

/// Test whether bit `target` is set in `state` (scalar, dense representation).
pub(crate) const fn is_one_at(state: usize, target: usize) -> bool {
    state & (1 << target) != 0
}

/// Test whether bit `target` is set in `state` (packed `Vec<u64>`, sparse representation).
pub(crate) fn is_one_at_vec(state: &[u64], target: usize) -> bool {
    let (outer_index, inner_index) = target.div_mod_floor(&64);
    state[outer_index] & (1 << inner_index) != 0
}

/// Return `true` iff **all** qubits in `control` are `|1⟩` in `state`
/// (sparse representation).
pub(crate) fn ctrl_check_vec(state: &[u64], control: &[usize]) -> bool {
    control.iter().all(|control| is_one_at_vec(state, *control))
}
