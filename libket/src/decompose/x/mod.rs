// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Specialised decompositions for multi-controlled Pauli-X gates.
//!
//! Sub-modules:
//! - **4 or more controls:** Handled dynamically via `x_network` where the
//!   target index is the last element. Dispatched to the fallback
//!   function [`c2to4x::c1to4x`].
//! - [`single_aux`]: linear/log-depth decompositions using a single ancilla.
//! - [`v_chain`] : V-chain decompositions using a chain of ancillae.

use serde::Serialize;
pub mod c2to4x;
pub mod single_aux;
pub mod v_chain;

/// Selects the largest primitive multi-controlled-X gate available to the network
/// and V-chain decomposers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Hash)]
pub enum CXMode {
    /// Use Toffoli (C2X) as the primitive two-control gate.
    C2X,
    /// Use the three-control C3X gate as the primitive, requiring fewer
    /// recursion levels but more ancillae per level.
    C3X,
}
