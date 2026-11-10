// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Classical shadows implementation for expectation-value estimation.
//!
//! This module provides the routines necessary to perform randomized measurements
//! in the Pauli bases (X, Y, Z) and post-process the resulting bit-strings into
//! estimates of expectation values, following the classical shadows protocol.

use core::f64;

use itertools::Itertools;
use num::Integer;
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    rng,
};

use crate::{
    error::KetError,
    execution::{BatchExecution, BitString, SampleData},
    ir::{
        gate::{GateInstruction, QuantumGate},
        hamiltonian::{Hamiltonian, Pauli, PauliString},
    },
    process::{GateList, GateListOwned},
};

type CSRound = Vec<Pauli>;

/// Generates a sequence of randomized Pauli measurement bases.
///
/// # Arguments
/// * `num_qubits` - The number of qubits to measure.
/// * `samples` - The number of measurement rounds to generate.
/// * `bias` - A tuple `(X, Y, Z)` specifying the relative sampling weights for the X, Y, and Z bases.
///
/// # Errors
/// Returns [`KetError::InvalidBias`] if all three weights are zero.
fn randomized_classical_shadows(
    num_qubits: usize,
    samples: usize,
    bias: (u8, u8, u8),
) -> std::result::Result<Vec<CSRound>, KetError> {
    let mut rng = rng();
    let dist = WeightedIndex::new([bias.0, bias.1, bias.2]).map_err(|_| KetError::InvalidBias)?;
    Ok((0..samples)
        .map(|_| {
            (0..num_qubits)
                .map(|_| match dist.sample(&mut rng) {
                    0 => Pauli::PauliX,
                    1 => Pauli::PauliY,
                    2 => Pauli::PauliZ,
                    _ => unreachable!(),
                })
                .collect::<CSRound>()
        })
        .collect())
}

/// Compiles a batch of measurement circuits by appending the appropriate basis-change
/// gates to the base quantum circuit.
fn classical_shadows_circuits(
    circuit: GateList,
    rounds: &[CSRound],
) -> Result<Vec<GateListOwned>, KetError> {
    match circuit {
        GateList::Ir { gates } => rounds
            .iter()
            .map(|round| {
                let mut circuit = gates.to_owned();
                for (i, measure) in round.iter().enumerate() {
                    match measure {
                        Pauli::PauliX => {
                            circuit.push(GateInstruction::new(QuantumGate::Hadamard, i));
                        }
                        Pauli::PauliY => {
                            circuit.push(GateInstruction::new(QuantumGate::sd(), i));

                            circuit.push(GateInstruction::new(QuantumGate::Hadamard, i));
                        }
                        Pauli::PauliZ => {}
                    }
                }
                Ok(GateListOwned::Ir { gates: circuit })
            })
            .collect(),
        GateList::Native {
            gates,
            native_gate_set,
        } => rounds
            .iter()
            .map(|round| {
                let mut circuit = gates.to_owned();
                for (i, measure) in round.iter().enumerate() {
                    match measure {
                        Pauli::PauliX => {
                            circuit.append(
                                &mut native_gate_set
                                    .translate(&QuantumGate::Hadamard.matrix(), i)?,
                            );
                        }
                        Pauli::PauliY => {
                            circuit.append(
                                &mut native_gate_set.translate(&QuantumGate::sd().matrix(), i)?,
                            );

                            circuit.append(
                                &mut native_gate_set
                                    .translate(&QuantumGate::Hadamard.matrix(), i)?,
                            );
                        }
                        Pauli::PauliZ => {}
                    }
                }
                Ok(GateListOwned::Native { gates: circuit })
            })
            .collect(),
    }
}

/// Processes the sampled measurement results to estimate the expectation value
/// of a single Pauli observable.
#[allow(clippy::cast_precision_loss)]
fn classical_shadows_processing(
    obs: &PauliString,
    measures: &[SampleData],
    shots: usize,
    rounds: &[CSRound],
) -> f64 {
    let matching_measures: Vec<_> = rounds
        .iter()
        .enumerate()
        .filter_map(|(index, round)| {
            if obs.iter().all(|term| round[term.qubit] == term.pauli) {
                Some(index)
            } else {
                None
            }
        })
        .collect();

    matching_measures
        .iter()
        .map(|index| &measures[*index])
        .flat_map(|(states, counts)| {
            states.iter().zip(counts.iter()).map(|(state, count)| {
                obs.iter()
                    .map(|p| from_sample_to_exp_value(state, p.qubit))
                    .product::<f64>()
                    * *count as f64
            })
        })
        .sum::<f64>()
        / (matching_measures.len() * shots) as f64
}

/// Extracts the parity (eigenvalue `+1.0` or `-1.0`) for a single qubit from a bit-string sample.
pub(super) fn from_sample_to_exp_value(state: &BitString, qubit: usize) -> f64 {
    let (vec_index, bit_index) = qubit.div_rem(&64);

    if state[vec_index] & (1 << bit_index) == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Executes the classical shadows protocol on a quantum batch execution backend.
///
/// This function generates the randomized measurement rounds, dispatches the circuits
/// to the backend, and post-processes the samples to estimate the expectation values
/// of all provided Hamiltonians.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_classical_shadows(
    qpu: &(impl BatchExecution + ?Sized),
    num_qubits: usize,
    gates: GateList,
    hamiltonian_list: &[Hamiltonian],
    samples: usize,
    shots: usize,
    bias: (u8, u8, u8),
) -> Result<Vec<f64>, KetError> {
    let rounds = randomized_classical_shadows(num_qubits, samples, bias)?;
    let circuits = classical_shadows_circuits(gates, &rounds)?;

    let all_qubits = (0..num_qubits).collect_vec();

    let measures: Result<Vec<_>, _> = circuits
        .iter()
        .map(|gates| match gates {
            GateListOwned::Ir { gates } => qpu.sample(gates, &all_qubits, shots),
            GateListOwned::Native { gates, .. } => qpu.sample_native(gates, &all_qubits, shots),
        })
        .collect();

    let measures = measures?;

    Ok(hamiltonian_list
        .iter()
        .map(|hamiltonian| {
            hamiltonian
                .pauli_strings
                .iter()
                .zip(hamiltonian.coefficients.iter())
                .map(|(pauli_string, coefficient)| {
                    classical_shadows_processing(pauli_string, &measures, shots, &rounds)
                        * coefficient
                })
                .sum::<f64>()
        })
        .collect_vec())
}
