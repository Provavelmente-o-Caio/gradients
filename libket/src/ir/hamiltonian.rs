// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Hamiltonian operator representation for expectation-value computations.
//!
//! A [`Hamiltonian`] is a sum of weighted Pauli strings. Each Pauli string is a
//! tensor product of single-qubit Pauli operators ([`Pauli`]) indexed by a
//! logical qubit. The expectation value `⟨ψ|H|ψ⟩` of the Hamiltonian with
//! respect to a quantum state `|ψ⟩` is computed by the execution backend.

use std::collections::BTreeMap;

use itertools::Itertools;
use serde::{Deserialize, Serialize};

/// A single-qubit Pauli operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Pauli {
    /// The Pauli-X operator (bit-flip).
    PauliX,
    /// The Pauli-Y operator.
    PauliY,
    /// The Pauli-Z operator (phase-flip).
    PauliZ,
}

/// A single-qubit Pauli operator applied to a specific logical qubit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PauliTerm {
    /// The Pauli operator to apply.
    pub pauli: Pauli,
    /// The logical qubit index the operator acts on.
    pub qubit: usize,
}

impl PauliTerm {
    /// Returns a copy of this term with the qubit index remapped through `mapping`.
    ///
    /// `mapping` must contain an entry for `self.qubit`; the method panics
    /// otherwise (this is always satisfied after qubit allocation).
    #[must_use]
    pub fn map(&self, mapping: &BTreeMap<usize, usize>) -> Self {
        Self {
            pauli: self.pauli,
            qubit: mapping[&self.qubit],
        }
    }
}

/// An ordered list of [`PauliTerm`]s forming a tensor-product Pauli string.
pub type PauliString = Vec<PauliTerm>;

/// A Hamiltonian operator expressed as a weighted sum of Pauli strings.
///
/// ```text
/// H = Σᵢ coefficients[i] · pauli_strings[i]
/// ```
///
/// Hamiltonians are built incrementally with [`Hamiltonian::add_pauli_string`]
/// and remapped to physical qubits with [`Hamiltonian::map`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hamiltonian {
    /// The list of Pauli strings that make up this Hamiltonian.
    pub pauli_strings: Vec<PauliString>,
    /// Real-valued coefficients, one per Pauli string.
    pub coefficients: Vec<f64>,
}

impl Hamiltonian {
    /// Creates a new, empty Hamiltonian.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a weighted Pauli string to this Hamiltonian.
    pub fn add_pauli_string(&mut self, pauli_string: PauliString, coefficient: f64) {
        self.pauli_strings.push(pauli_string);
        self.coefficients.push(coefficient);
    }

    /// Returns a copy of this Hamiltonian with all qubit indices remapped
    /// through `mapping` (logical → physical qubit translation).
    #[must_use]
    pub fn map(&self, mapping: &BTreeMap<usize, usize>) -> Self {
        Self {
            pauli_strings: self
                .pauli_strings
                .iter()
                .map(|pauli_string| {
                    pauli_string
                        .iter()
                        .map(|term| term.map(mapping))
                        .collect_vec()
                })
                .collect_vec(),
            coefficients: self.coefficients.clone(),
        }
    }
}
