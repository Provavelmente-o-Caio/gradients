// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Qubit-Wise Commutation (QWC) implementation for expectation-value estimation.
//!
//! This module groups Pauli observables into mutually commuting sets, allowing
//! simultaneous measurement of multiple observables within the same basis to
//! minimize the number of measurement circuits.

use std::collections::BTreeMap;

use itertools::Itertools;

use crate::{
    error::KetError,
    execution::{BatchExecution, LiveExecution, SampleData},
    ir::{
        gate::{GateInstruction, QuantumGate},
        hamiltonian::{Hamiltonian, Pauli, PauliString},
    },
    process::{classical_shadow::from_sample_to_exp_value, GateList, GateListOwned},
};

/// Groups a list of Pauli observables into sets of qubit-wise commuting operators
/// using a greedy approach.
fn greedy_qwc_grouping(
    pauli_strings: &[&PauliString],
) -> Vec<(BTreeMap<usize, Pauli>, Vec<usize>)> {
    let mut groups: Vec<(BTreeMap<usize, Pauli>, Vec<usize>)> = Vec::new();

    for (i, pauli_string) in pauli_strings.iter().enumerate() {
        let mut added = false;

        for (basis, members) in &mut groups {
            let is_compatible = pauli_string.iter().all(|term| {
                if let Some(&basis_pauli) = basis.get(&term.qubit) {
                    basis_pauli == term.pauli
                } else {
                    true
                }
            });

            if is_compatible {
                for term in *pauli_string {
                    basis.insert(term.qubit, term.pauli);
                }
                members.push(i);
                added = true;
                break;
            }
        }

        if !added {
            let mut new_basis = BTreeMap::new();
            for term in *pauli_string {
                new_basis.insert(term.qubit, term.pauli);
            }
            groups.push((new_basis, vec![i]));
        }
    }

    groups
}

/// Compiles measurement circuits by appending the appropriate basis-change
/// gates for each measurement basis group to the base quantum circuit.
fn qwc_circuits(
    circuit: GateList,
    bases: &[&BTreeMap<usize, Pauli>],
) -> Result<Vec<GateListOwned>, KetError> {
    match circuit {
        GateList::Ir { gates } => bases
            .iter()
            .map(|basis| {
                let mut current_circuit = gates.to_owned();

                for (&qubit, pauli) in *basis {
                    match pauli {
                        Pauli::PauliX => {
                            current_circuit
                                .push(GateInstruction::new(QuantumGate::Hadamard, qubit));
                        }
                        Pauli::PauliY => {
                            current_circuit.push(GateInstruction::new(QuantumGate::sd(), qubit));
                            current_circuit
                                .push(GateInstruction::new(QuantumGate::Hadamard, qubit));
                        }
                        Pauli::PauliZ => {}
                    }
                }
                Ok(GateListOwned::Ir {
                    gates: current_circuit,
                })
            })
            .collect(),
        GateList::Native {
            gates,
            native_gate_set,
        } => bases
            .iter()
            .map(|basis| {
                let mut current_circuit = gates.to_owned();

                for (&qubit, pauli) in *basis {
                    match pauli {
                        Pauli::PauliX => {
                            current_circuit.append(
                                &mut native_gate_set
                                    .translate(&QuantumGate::Hadamard.matrix(), qubit)?,
                            );
                        }
                        Pauli::PauliY => {
                            current_circuit.append(
                                &mut native_gate_set
                                    .translate(&QuantumGate::sd().matrix(), qubit)?,
                            );
                            current_circuit.append(
                                &mut native_gate_set
                                    .translate(&QuantumGate::Hadamard.matrix(), qubit)?,
                            );
                        }
                        Pauli::PauliZ => {}
                    }
                }
                Ok(GateListOwned::Native {
                    gates: current_circuit,
                })
            })
            .collect(),
    }
}

/// Processes the sampled measurement results to estimate the expectation value
/// of a single Pauli observable.
#[allow(clippy::cast_precision_loss)]
fn qwc_processing(obs: &PauliString, measure: &SampleData, shots: usize) -> f64 {
    let (states, counts) = measure;

    states
        .iter()
        .zip(counts.iter())
        .map(|(state, count)| {
            obs.iter()
                .map(|p| from_sample_to_exp_value(state, p.qubit))
                .product::<f64>()
                * (*count as f64)
        })
        .sum::<f64>()
        / shots as f64
}

/// Executes expectation-value computation using the Qubit-Wise Commutation (QWC) strategy
/// on a batch execution backend.
///
/// This function groups all Hamiltonian terms into commuting bases, dispatches the grouped
/// measurement circuits, and combines the resulting measurements to estimate the expectation
/// values.
pub(super) fn execute_qwc(
    qpu: &(impl BatchExecution + ?Sized),
    num_qubits: usize,
    gates: GateList,
    hamiltonian_list: &[Hamiltonian],
    shots: usize,
) -> Result<Vec<f64>, KetError> {
    let mut all_pauli_strings = Vec::new();
    let mut layout = Vec::with_capacity(hamiltonian_list.len());

    for ham in hamiltonian_list {
        layout.push(ham.pauli_strings.len());
        for pauli_string in &ham.pauli_strings {
            all_pauli_strings.push(pauli_string);
        }
    }

    let groups = greedy_qwc_grouping(&all_pauli_strings);

    let bases = groups.iter().map(|(b, _)| b).collect_vec();
    let circuits = qwc_circuits(gates, &bases)?;

    let all_qubits = (0..num_qubits).collect_vec();

    let measures: Result<Vec<_>, _> = circuits
        .iter()
        .map(|circ| match circ {
            GateListOwned::Ir { gates } => qpu.sample(gates, &all_qubits, shots),
            GateListOwned::Native { gates, .. } => qpu.sample_native(gates, &all_qubits, shots),
        })
        .collect();

    let measures = measures?;

    let mut exp_values = vec![0.0; all_pauli_strings.len()];

    for (group_idx, (_, members)) in groups.iter().enumerate() {
        for string_idx in members {
            exp_values[*string_idx] =
                qwc_processing(all_pauli_strings[*string_idx], &measures[group_idx], shots);
        }
    }

    let mut final_results = Vec::with_capacity(hamiltonian_list.len());
    let mut current_idx = 0;

    for (ham_idx, ham) in hamiltonian_list.iter().enumerate() {
        let mut ham_exp_val = 0.0;
        let count = layout[ham_idx];

        for i in 0..count {
            ham_exp_val = exp_values[current_idx].mul_add(ham.coefficients[i], ham_exp_val);
            current_idx += 1;
        }
        final_results.push(ham_exp_val);
    }

    Ok(final_results)
}

/// Executes expectation-value computation using the Qubit-Wise Commutation (QWC) strategy
/// by leveraging the backend's native expectation-value primitive.
pub(super) fn execute_qwc_exp_value(
    qpu: &(impl BatchExecution + ?Sized),
    gates: GateList,
    hamiltonian_list: &[Hamiltonian],
) -> Result<Vec<f64>, KetError> {
    let mut all_pauli_strings = Vec::new();
    let mut layout = Vec::with_capacity(hamiltonian_list.len());

    for ham in hamiltonian_list {
        layout.push(ham.pauli_strings.len());
        for pauli_string in &ham.pauli_strings {
            all_pauli_strings.push(pauli_string);
        }
    }

    let groups = greedy_qwc_grouping(&all_pauli_strings);

    let mut grouped_hamiltonians = Vec::new();
    let mut result_mapping = Vec::new();

    for (_, members) in groups {
        let mut hams = vec![Hamiltonian::new(); hamiltonian_list.len()];
        let mut has_terms = vec![false; hamiltonian_list.len()];

        for &string_idx in &members {
            let mut current_idx = 0;
            let mut ham_idx = 0;
            for (i, &count) in layout.iter().enumerate() {
                if string_idx < current_idx + count {
                    ham_idx = i;
                    break;
                }
                current_idx += count;
            }

            let coeff_idx = string_idx - current_idx;
            let coeff = hamiltonian_list[ham_idx].coefficients[coeff_idx];

            hams[ham_idx].add_pauli_string(all_pauli_strings[string_idx].clone(), coeff);
            has_terms[ham_idx] = true;
        }

        for (i, (ham, has)) in hams.into_iter().zip(has_terms).enumerate() {
            if has {
                result_mapping.push(i);
                grouped_hamiltonians.push(ham);
            }
        }
    }

    let results = match gates {
        GateList::Ir { gates } => qpu.exp_value(gates, &grouped_hamiltonians)?,
        GateList::Native { gates, .. } => qpu.exp_value_native(gates, &grouped_hamiltonians)?,
    };

    let mut final_results = vec![0.0; hamiltonian_list.len()];

    for (ham_idx, res) in result_mapping.into_iter().zip(results) {
        final_results[ham_idx] += res;
    }

    Ok(final_results)
}

pub(super) fn execute_qwc_exp_value_live(
    qpu: &mut (impl LiveExecution + ?Sized),
    hamiltonian: &Hamiltonian,
) -> Result<f64, KetError> {
    let pauli_strings_refs: Vec<_> = hamiltonian.pauli_strings.iter().collect();

    let groups = greedy_qwc_grouping(&pauli_strings_refs);

    let mut grouped_hamiltonians = Vec::with_capacity(groups.len());

    for (_, members) in groups {
        let mut sub_ham = Hamiltonian::new();

        for &string_idx in &members {
            let pauli_string = hamiltonian.pauli_strings[string_idx].clone();
            let coeff = hamiltonian.coefficients[string_idx];

            sub_ham.add_pauli_string(pauli_string, coeff);
        }

        grouped_hamiltonians.push(sub_ham);
    }

    let results = grouped_hamiltonians
        .drain(..)
        .map(|h| qpu.exp_value(h))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(results.iter().sum())
}
