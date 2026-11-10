// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Initial qubit placement for the mapping pass.
//!
//! The strategy builds an *interaction graph* from the circuit's qubit
//! read/write dependency maps, places the most-connected logical qubit at the
//! physical *centre* of the coupling graph, then assigns the remaining logical
//! qubits in BFS order by choosing the physical location closest to the
//! already-placed neighbours.

use crate::ir::block::BasicBlock;
use crate::mapping::graph::GraphMatrix;
use itertools::Itertools;
use std::collections::BTreeMap;

type InitialMapping = (BTreeMap<usize, usize>, BTreeMap<usize, usize>);

/// Builds an unweighted *interaction graph* from a circuit's qubit dependencies.
///
/// An edge `(target, control)` is added for every two-qubit interaction that
/// appears at least once in `block.qubits_written`.  The distance matrix is
/// pre-computed before returning so that [`GraphMatrix::dist`] is immediately
/// available.
pub(super) fn build_interaction_graph(
    block: &BasicBlock,
    num_logical_qubits: usize,
) -> GraphMatrix {
    let mut interaction_graph = GraphMatrix::new(num_logical_qubits);

    for (target, control) in block.qubits_op.iter().flat_map(|(target, control)| {
        control
            .read_qubits
            .iter()
            .map(|control| (*target, *control))
    }) {
        interaction_graph.set_edge(target, control, 1);
    }

    interaction_graph.calculate_distance();
    interaction_graph
}

/// Computes an initial logical-to-physical qubit assignment.
///
/// The most central logical qubit (lowest graph eccentricity) is placed at
/// the most central physical qubit.  The remaining logical qubits are then
/// assigned in BFS order, choosing the physical location closest (in the
/// coupling graph) to the already-placed interaction neighbours.
///
/// Any logical qubits not connected in the interaction graph are assigned
/// to the first available unoccupied physical qubit.
///
/// Returns `(logical_to_physical, physical_to_logical)`: two maps that
/// are the inverse of each other.
pub(super) fn initial(
    interaction_graph: &GraphMatrix,
    coupling_graph: &GraphMatrix,
) -> InitialMapping {
    let center_logical_qubit = interaction_graph.get_center();
    let center_physical_qubit = coupling_graph.get_center();

    let mut logical_to_physical = BTreeMap::new();
    let mut physical_to_logical = BTreeMap::new();

    logical_to_physical.insert(center_logical_qubit, center_physical_qubit);
    physical_to_logical.insert(center_physical_qubit, center_logical_qubit);

    let mut queue = interaction_graph.breadth_first_search(center_logical_qubit);
    queue.pop_front();

    while let Some(next_logical_qubit) = queue.pop_front() {
        let ref_locations =
            reference_locations(next_logical_qubit, &logical_to_physical, interaction_graph);
        let cand_locations =
            candidate_locations(&ref_locations, &physical_to_logical, coupling_graph);

        let next_physical_qubit = next_physical_qubit(
            next_logical_qubit,
            &cand_locations,
            interaction_graph,
            coupling_graph,
        );

        logical_to_physical.insert(next_logical_qubit, next_physical_qubit);
        physical_to_logical.insert(next_physical_qubit, next_logical_qubit);
    }

    let num_logical = interaction_graph.num_nodes();
    let all_logical = 0..num_logical;

    for logical in all_logical {
        if let std::collections::btree_map::Entry::Vacant(e) = logical_to_physical.entry(logical) {
            let mut free_physical = 0;
            while physical_to_logical.contains_key(&free_physical) {
                free_physical += 1;
            }
            e.insert(free_physical);
            physical_to_logical.insert(free_physical, logical);
        }
    }

    (logical_to_physical, physical_to_logical)
}

/// Returns the physical locations of the already-placed neighbours of `qubit`
/// in the interaction graph, sorted by ascending interaction weight.
fn reference_locations(
    qubit: usize,
    logical_to_physical: &BTreeMap<usize, usize>,
    interaction_graph: &GraphMatrix,
) -> Vec<usize> {
    let mut reference_locations: Vec<_> = interaction_graph
        .neighbors(qubit)
        .into_iter()
        .filter(|(node, _)| logical_to_physical.contains_key(node))
        .collect();

    reference_locations.sort_by_key(|(_, value)| *value);
    reference_locations
        .into_iter()
        .map(|(node, _)| *logical_to_physical.get(&node).unwrap())
        .collect()
}

/// Returns the set of free physical qubits that are candidates for the next
/// placement step.
///
/// Starts from the neighbours of the first reference location and expands
/// outwards until at least one free qubit is found.  If multiple reference
/// locations are given, the candidate set is pruned to retain only those
/// within the minimum distance of each successive reference.
fn candidate_locations(
    reference_locations: &[usize],
    physical_to_logical: &BTreeMap<usize, usize>,
    coupling_graph: &GraphMatrix,
) -> Vec<usize> {
    if reference_locations.is_empty() {
        return (0..coupling_graph.num_nodes())
            .filter(|node| !physical_to_logical.contains_key(node))
            .collect();
    }

    let mut neighbors_locations: Vec<_> = coupling_graph
        .neighbors(reference_locations[0])
        .into_iter()
        .map(|(node, _)| node)
        .collect();

    let mut candidate_locations: Vec<_>;

    loop {
        if neighbors_locations.is_empty() {
            candidate_locations = (0..coupling_graph.num_nodes())
                .filter(|node| !physical_to_logical.contains_key(node))
                .collect();
            break;
        }

        candidate_locations = neighbors_locations
            .iter()
            .filter(|node| !physical_to_logical.contains_key(node))
            .copied()
            .collect();

        if !candidate_locations.is_empty() {
            break;
        }

        neighbors_locations = neighbors_locations
            .into_iter()
            .flat_map(|qubit| {
                coupling_graph
                    .neighbors(qubit)
                    .into_iter()
                    .map(|(node, _)| node)
                    .collect::<Vec<_>>()
            })
            .unique()
            .collect();
    }

    for reference in &reference_locations[1..] {
        let mut candidate_locations_dist: Vec<_> = candidate_locations
            .iter()
            .map(|candidate| (*candidate, coupling_graph.dist(*candidate, *reference)))
            .collect();

        candidate_locations_dist.sort_by_key(|(_, dist)| *dist);

        let min_dist = candidate_locations_dist[0].1;
        candidate_locations = candidate_locations_dist
            .into_iter()
            .filter_map(|(node, dist)| if dist <= min_dist { Some(node) } else { None })
            .collect();

        if candidate_locations.len() == 1 {
            break;
        }
    }

    candidate_locations
}

/// Selects the best physical qubit from the candidates for `next_logical_qubit`.
///
/// When multiple candidates remain after distance filtering, the one whose
/// degree in the coupling graph is closest to the degree of the logical qubit
/// in the interaction graph is chosen (degree-matching heuristic).
fn next_physical_qubit(
    next_logical_qubit: usize,
    candidate_locations: &[usize],
    interaction_graph: &GraphMatrix,
    coupling_graph: &GraphMatrix,
) -> usize {
    if candidate_locations.len() > 1 {
        let interaction_degree = interaction_graph.degree(next_logical_qubit);

        let mut candidate_degree: Vec<_> = candidate_locations
            .iter()
            .map(|node| {
                (
                    node,
                    coupling_graph.degree(*node).abs_diff(interaction_degree),
                )
            })
            .collect();

        candidate_degree.sort_by_key(|(_, degree)| *degree);

        *candidate_degree[0].0
    } else {
        candidate_locations[0]
    }
}
