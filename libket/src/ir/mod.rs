// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) modules for the Libket quantum compiler.
//!
//! This module re-exports the three primary IR sub-modules:
//! - [`block`]: basic blocks (straight-line gate sequences).
//! - [`gate`]: gate types, parameters, and gate instructions.
//! - [`hamiltonian`]: Hamiltonian operators for expectation-value tasks.

pub mod block;
pub mod gate;
pub mod hamiltonian;
pub mod macros;
