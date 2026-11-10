// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Multi-controlled gate decomposition into primitive two-qubit operations.
//!
//! This module provides the data types and ancilla-count formulae that
//! orchestrate gate decomposition in the compiler.  The sub-modules implement
//! the concrete decomposition algorithms:
//!
//! - [`network`]: recursive network decomposition for general unitaries.
//! - [`su2`]    : SU(2)-based linear/log-depth decomposition.
//! - [`u2`]     : single-controlled U(2) decomposition (ZYZ basis).
//! - [`util`]   : matrix arithmetic utilities (ZYZ decomposition, eigenvalues).
//! - [`x`]      : specialised Pauli-X multi-controlled decompositions.
//!
//! The primary entry point for callers is [`crate::ir::gate::GateInstruction::decompose`].

use num::Integer;
use serde::Serialize;
use x::CXMode;

pub(crate) mod network;
pub(crate) mod su2;
pub(crate) mod u2;
pub(crate) mod util;
pub(crate) mod x;

/// Controls whether ancilla qubits used during decomposition are clean or dirty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Hash)]
pub enum AuxMode {
    /// Ancilla qubits are guaranteed to be in the |0⟩ state before use.
    Clean,
    /// Ancilla qubits may be in an arbitrary state; the algorithm must
    /// restore them to their original state before returning.
    Dirty,
}

/// Controls the circuit depth trade-off for decomposition algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Hash)]
pub enum DepthMode {
    /// Logarithmic depth (fewer layers, more CNOT gates in total).
    Log,
    /// Linear depth (more layers, fewer total CNOT gates when ancillae are available).
    Linear,
}

/// Identifies a specific multi-controlled decomposition algorithm.
///
/// Used to query the number of ancilla qubits required by each algorithm
/// via [`Algorithm::aux_needed`], enabling the compiler to choose the best
/// available strategy given the current qubit budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default, Hash)]
pub(crate) enum Algorithm {
    /// V-chain decomposition with the given CX mode and ancilla mode.
    VChain(CXMode, AuxMode),
    /// Network decomposition for general U(2) gates.
    NetworkU2(CXMode),
    /// Network decomposition specialised for Pauli-X gates.
    NetworkPauli(CXMode),
    /// Linear-depth decomposition requiring no ancilla qubits.
    #[default]
    LinearDepth,
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Returns the number of ancilla qubits required by the `NetworkU2` algorithm
/// for `n` control qubits.
///
/// The recursion groups controls into blocks of `n_cx` (2 for C2X, 3 for C3X)
/// and accumulates one ancilla per group.
fn num_aux_network_u2(cx_mode: CXMode, n: usize) -> usize {
    let n_cx = match cx_mode {
        CXMode::C2X => 2,
        CXMode::C3X => 3,
    };

    let (a, rem) = n.div_rem(&n_cx);

    if n == 1 {
        0
    } else if n == 2 && matches!(cx_mode, CXMode::C3X) {
        1
    } else {
        a + num_aux_network_u2(cx_mode, a + rem)
    }
}

/// Returns the number of ancilla qubits required by the `NetworkPauli` algorithm
/// for `n` control qubits.
///
/// Similar to [`num_aux_network_u2`] but the base case changes: no ancilla is
/// needed when `n ≤ n_cx` (since a direct `C{n_cx}X` gate is available).
fn num_aux_network_pauli(cx_mode: CXMode, n: usize) -> usize {
    let n_cx = match cx_mode {
        CXMode::C2X => 2,
        CXMode::C3X => 3,
    };

    let (a, rem) = n.div_rem(&n_cx);

    if n <= n_cx {
        0
    } else {
        a + num_aux_network_pauli(cx_mode, a + rem)
    }
}

impl Algorithm {
    /// Returns the number of ancilla qubits required for `n` control qubits.
    ///
    /// The caller should compare the returned value against the available
    /// ancilla budget before selecting this algorithm.
    pub fn aux_needed(self, n: usize) -> usize {
        match self {
            Self::VChain(cx_mode, _) => match cx_mode {
                CXMode::C2X => n - 2,
                CXMode::C3X => usize::div_ceil(n - 3, 2),
            },
            Self::NetworkU2(cx_mode) => num_aux_network_u2(cx_mode, n),
            Self::NetworkPauli(cx_mode) => num_aux_network_pauli(cx_mode, n),
            Self::LinearDepth => 0,
        }
    }
}
