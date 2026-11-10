// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Qubit mapping: routing a logical circuit onto a hardware coupling graph.
//!
//! Given a [`BasicBlock`] of logical gates and a hardware coupling graph, this
//! module performs two passes:
//!
//! 1. **Initial placement** ([`allocation`]): assigns each logical qubit an
//!    initial physical qubit that minimises the expected number of SWAP
//!    operations needed later.
//! 2. **Dynamic look-ahead routing** ([`dynamic_look_ahead`]): inserts SWAP
//!    gates as needed to satisfy connectivity constraints, emitting a flat
//!    sequence of [`GateInstruction`] values in physical qubit space.
//!
//! Native-gate translation (via [`crate::execution::NativeGateSet`]) is
//! performed **after** mapping in [`crate::process::Process::execute`], so
//! this module operates purely at the IR level.
//!
//! The top-level entry point is [`map_circuit`].

use crate::{
    error::KetError,
    ir::{block::BasicBlock, gate::GateInstruction, hamiltonian::Hamiltonian},
    mapping::graph::GraphMatrix,
};

pub mod allocation;
pub mod dynamic_look_ahead;
pub mod graph;

/// Routes a logical [`BasicBlock`] to physical qubits and returns IR-level gate
/// instructions in physical qubit space.
///
/// Builds a coupling [`GraphMatrix`] from the `qpu` edge list, computes an
/// initial logical-to-physical qubit assignment via [`allocation::initial`],
/// then routes the circuit with the dynamic look-ahead SWAP inserter.  SWAP
/// gates are emitted as three consecutive CNOT instructions.
///
/// After routing, the Hamiltonians and `qubits_to_sample` indices are remapped
/// from logical to physical space using the final qubit assignment.
///
/// # Parameters
/// - `block`            : the logical gate sequence to route.
/// - `hamiltonian_list` : Hamiltonians whose qubit indices will be remapped to
///   physical space in the returned vector.
/// - `qubits_to_sample` : logical qubit indices to sample; returned remapped
///   to physical space.
/// - `qpu`              : hardware coupling graph as a list of undirected edges
///   `(i, j)`.
/// - `num_qubits`       : total number of physical qubits available.
///
/// # Returns
/// `(mapped_gates, mapped_hamiltonians, mapped_qubits_to_sample)` where all
/// qubit indices are expressed in physical space.
///
/// # Errors
/// Propagates any [`KetError`] returned by the SWAP-insertion pass.
#[allow(clippy::type_complexity)]
pub fn map_circuit(
    block: &BasicBlock,
    hamiltonian_list: &[Hamiltonian],
    qubits_to_sample: &[usize],
    qpu: &[(usize, usize)],
    num_qubits: usize,
) -> Result<(Vec<GateInstruction>, Vec<Hamiltonian>, Vec<usize>), KetError> {
    let mut coupling_graph = GraphMatrix::new(num_qubits);
    for (i, j) in qpu {
        coupling_graph.set_edge(*i, *j, 1);
    }
    coupling_graph.calculate_distance();

    let interaction_graph = allocation::build_interaction_graph(block, num_qubits);
    let (logical_to_physical, physical_to_logical) =
        allocation::initial(&interaction_graph, &coupling_graph);

    let logical_circuit = block.flat_gates(None, None);

    let (mapped_gates, final_mapping) = dynamic_look_ahead::map(
        &logical_circuit,
        logical_to_physical,
        physical_to_logical,
        &coupling_graph,
    )?;

    let mapped_gate: Vec<GateInstruction> = mapped_gates.iter().map(Into::into).collect();
    let mapped_hamiltonian: Vec<_> = hamiltonian_list
        .iter()
        .map(|h| h.map(&final_mapping))
        .collect();
    let mapped_qubits_to_sample: Vec<_> = qubits_to_sample
        .iter()
        .map(|qubit| final_mapping[qubit])
        .collect();

    Ok((mapped_gate, mapped_hamiltonian, mapped_qubits_to_sample))
}
