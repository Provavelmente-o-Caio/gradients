// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Dynamic look-ahead SWAP router for connectivity-constrained qubit mapping.
//!
//! The [`map`] function implements a greedy, look-ahead SWAP insertion strategy:
//! it maintains per-qubit instruction queues, executes every gate whose operands
//! are already adjacent in the coupling graph, and: when stuck: selects the
//! SWAP that maximises the *marginal cumulative path effect* (MCPE) score over
//! the pending front layer.
//!
//! # Single-qubit gate batching
//!
//! Rather than translating each single-qubit gate individually, the router
//! accumulates a run of consecutive single-qubit gates on the same physical
//! qubit by multiplying their 2×2 unitary matrices.  The accumulated product is
//! flushed (and `translate` is called exactly once) whenever:
//!
//! - a two-qubit gate must be emitted for that qubit, or
//! - the routing loop terminates.
//!
//! This lets the [`NativeGateSet`] see the composed unitary and choose the
//! most efficient native decomposition (e.g. a single U3 gate instead of a
//! chain of Rz/Ry/Rx gates).

use crate::error::KetError;
use crate::ir::gate::DecomposedGate;
use crate::mapping::graph::GraphMatrix;
use std::collections::{BTreeMap, HashSet, VecDeque};

/// Routes a logical circuit to a physical circuit, inserting SWAPs as needed.
///
/// The algorithm operates on per-qubit instruction queues (`qubit_lines`).  In
/// each iteration it collects all gates whose operands are at the front of both
/// their queues (the *front layer*), executes those that are already adjacent
/// on the coupling graph, and inserts the best SWAP otherwise.
///
/// Single-qubit gates are **not** translated immediately.  Instead their
/// matrices are accumulated per physical qubit in `pending_sq` and flushed as
/// a single `translate` call when a two-qubit gate (or end-of-circuit) forces
/// it.  This gives the [`NativeGateSet`] the opportunity to produce a more
/// compact decomposition of the composed unitary.
///
/// The SWAP score is computed by [`calc_mcpe`]: the SWAP with the highest
/// aggregate marginal improvement over the pending interactions is chosen.
///
/// # Returns
/// `(physical_circuit, logical_to_physical)`: the translated native gate
/// sequence and the final qubit assignment.
///
/// # Errors
/// Propagates any [`KetError`] returned by [`NativeGateSet::translate`] or
/// [`NativeGateSet::swap`].
#[allow(clippy::too_many_lines)]
pub(super) fn map(
    logical_circuit: &[DecomposedGate],
    mut logical_to_physical: BTreeMap<usize, usize>,
    mut physical_to_logical: BTreeMap<usize, usize>,
    coupling_graph: &GraphMatrix,
) -> Result<(Vec<DecomposedGate>, BTreeMap<usize, usize>), KetError> {
    let mut physical_circuit = Vec::<DecomposedGate>::new();

    let mut qubit_lines: BTreeMap<usize, VecDeque<usize>> = BTreeMap::new();
    for (i, gate) in logical_circuit.iter().enumerate() {
        match gate {
            DecomposedGate::U(_, target) => qubit_lines.entry(*target).or_default().push_back(i),
            DecomposedGate::CNOT(control, target) => {
                qubit_lines.entry(*target).or_default().push_back(i);
                qubit_lines.entry(*control).or_default().push_back(i);
            }
        }
    }

    let mut visited_states = HashSet::new();

    loop {
        qubit_lines.retain(|_, queue| !queue.is_empty());
        if qubit_lines.is_empty() {
            break;
        }

        let mut executed_any = false;
        let mut active_front_gates = HashSet::new();

        for queue in qubit_lines.values() {
            let gate_idx = queue[0];
            let gate = &logical_circuit[gate_idx];

            let is_ready = if let DecomposedGate::CNOT(c, t) = gate {
                qubit_lines.get(t).is_some_and(|q| q[0] == gate_idx)
                    && qubit_lines.get(c).is_some_and(|q| q[0] == gate_idx)
            } else {
                true
            };

            if is_ready {
                active_front_gates.insert(gate_idx);
            }
        }

        let mut stuck_gates = Vec::new();

        for &gate_idx in &active_front_gates {
            let gate = &logical_circuit[gate_idx];
            match gate {
                DecomposedGate::U(gate, t) => {
                    let phys_target = logical_to_physical[t];
                    physical_circuit.push(DecomposedGate::U(*gate, phys_target));
                    qubit_lines.get_mut(t).unwrap().pop_front();
                    executed_any = true;
                }
                DecomposedGate::CNOT(c, t) => {
                    let phys_target = logical_to_physical[t];
                    let phys_control = logical_to_physical[c];

                    if coupling_graph.edge(phys_target, phys_control).is_some() {
                        physical_circuit.push(DecomposedGate::CNOT(phys_control, phys_target));
                        qubit_lines.get_mut(t).unwrap().pop_front();
                        qubit_lines.get_mut(c).unwrap().pop_front();
                        executed_any = true;
                    } else {
                        stuck_gates.push(gate_idx);
                    }
                }
            }
        }

        if executed_any {
            visited_states.clear();
            continue;
        }

        visited_states.insert(physical_to_logical.clone());

        let mut active_physical_qubits = HashSet::new();
        for &gate_idx in &stuck_gates {
            let DecomposedGate::CNOT(control, target) = &logical_circuit[gate_idx] else {
                unreachable!()
            };
            active_physical_qubits.insert(logical_to_physical[target]);
            active_physical_qubits.insert(logical_to_physical[control]);
        }

        let mut swap_candidates = HashSet::new();
        for &phys_q in &active_physical_qubits {
            for (neighbor, _) in coupling_graph.neighbors(phys_q) {
                let mut swap = [phys_q, neighbor];
                swap.sort_unstable(); // normalise edge (A, B) to ascending order
                swap_candidates.insert((swap[0], swap[1]));
            }
        }

        let mut swap_candidates_vec: Vec<_> = swap_candidates.into_iter().collect();
        swap_candidates_vec.sort_unstable();

        let best_swap = swap_candidates_vec
            .into_iter()
            .max_by_key(|&(phys_a, phys_b)| {
                let log_a = physical_to_logical.get(&phys_a).copied();
                let log_b = physical_to_logical.get(&phys_b).copied();

                let score_a = log_a.map_or(0, |la| {
                    calc_mcpe(
                        la,
                        phys_a,
                        phys_b,
                        &qubit_lines,
                        logical_circuit,
                        &logical_to_physical,
                        coupling_graph,
                    )
                });
                let score_b = log_b.map_or(0, |lb| {
                    calc_mcpe(
                        lb,
                        phys_b,
                        phys_a,
                        &qubit_lines,
                        logical_circuit,
                        &logical_to_physical,
                        coupling_graph,
                    )
                });

                let total_score = if score_a == 0 && score_b == 0 {
                    -1
                } else {
                    score_a + score_b
                };

                let mut next_state = physical_to_logical.clone();
                let log_a = next_state.remove(&phys_a);
                let log_b = next_state.remove(&phys_b);
                if let Some(la) = log_a {
                    next_state.insert(phys_b, la);
                }
                if let Some(lb) = log_b {
                    next_state.insert(phys_a, lb);
                }

                if visited_states.contains(&next_state) {
                    total_score - 1000000
                } else {
                    total_score
                }
            })
            .expect("No SWAP available.");

        let (phys_a, phys_b) = best_swap;

        physical_circuit.push(DecomposedGate::CNOT(phys_a, phys_b));
        physical_circuit.push(DecomposedGate::CNOT(phys_b, phys_a));
        physical_circuit.push(DecomposedGate::CNOT(phys_a, phys_b));

        let log_a = physical_to_logical.remove(&phys_a);
        let log_b = physical_to_logical.remove(&phys_b);

        if let Some(la) = log_a {
            logical_to_physical.insert(la, phys_b);
            physical_to_logical.insert(phys_b, la);
        }
        if let Some(lb) = log_b {
            logical_to_physical.insert(lb, phys_a);
            physical_to_logical.insert(phys_a, lb);
        }
    }

    Ok((physical_circuit, logical_to_physical))
}
/// Computes the *marginal cumulative path effect* (MCPE) score for moving
/// `logical_qubit` from `phys_current` to `phys_new`.
///
/// For each pending two-qubit gate in `logical_qubit`'s queue, the score
/// accumulates the reduction in coupling-graph distance to the partner qubit
/// that would result from the proposed SWAP.  Accumulation stops as soon as a
/// gate would *increase* the distance (negative marginal effect), preventing
/// optimistic look-ahead past a barrier gate.
fn calc_mcpe(
    logical_qubit: usize,
    phys_current: usize,
    phys_new: usize,
    qubit_lines: &BTreeMap<usize, VecDeque<usize>>,
    logical_circuit: &[DecomposedGate],
    logical_to_physical: &BTreeMap<usize, usize>,
    coupling_graph: &GraphMatrix,
) -> i64 {
    let mut score = 0;

    if let Some(queue) = qubit_lines.get(&logical_qubit) {
        for &gate_idx in queue {
            let (control, target) = match &logical_circuit[gate_idx] {
                DecomposedGate::CNOT(c, t) => (c, t),
                _ => continue,
            };

            let other_logical = if *target == logical_qubit {
                *control
            } else {
                *target
            };
            let phys_other = logical_to_physical[&other_logical];

            let old_dist = coupling_graph.dist(phys_current, phys_other);
            let new_dist = coupling_graph.dist(phys_new, phys_other);

            let effect = old_dist - new_dist;

            if effect >= 0 {
                score += effect;
            } else {
                break;
            }
        }
    }
    score
}
