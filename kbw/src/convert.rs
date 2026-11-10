// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! Conversion utilities between dump data formats and shot histograms.
//!
//! `from_dump_to_prob` converts a [`DumpData`]
//! (amplitudes) into a `DumpProbability` (Born-rule probabilities), and
//! `from_prob_to_shots` draws a shot histogram from those probabilities.

use ket::execution::{DumpData, SampleData};
use num::complex::{Complex64, ComplexFloat};
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    Rng,
};
use rayon::prelude::*;
use std::collections::HashMap;

/// Intermediate representation that holds Born-rule probabilities alongside
/// the corresponding basis states after a [`dump`](crate::quantum_execution::QuantumExecution::dump).
pub(crate) struct DumpProbability {
    pub(crate) basis_states: Vec<Vec<u64>>,
    pub(crate) probabilities: Vec<f64>,
}

/// Convert [`DumpData`] (complex amplitudes) to [`DumpProbability`] (Born-rule
/// probabilities) by computing `|amplitude|^2` for each basis state.
pub(crate) fn from_dump_to_prob(data: DumpData) -> DumpProbability {
    let probabilities = data
        .amplitudes_real
        .par_iter()
        .zip(&data.amplitudes_imag)
        .map(|(real, imag)| Complex64::new(*real, *imag).abs().powi(2))
        .collect();

    DumpProbability {
        basis_states: data.basis_states,
        probabilities,
    }
}

/// Sample `shots` outcomes from `data` and return a `(basis_states, counts)` pair.
///
/// Uses a [`WeightedIndex`] distribution so that each shot independently draws
/// a basis state proportional to its probability.
///
/// # Panics
///
/// Panics if `data.probabilities` is empty or all probabilities are zero
/// (i.e. the quantum state is the zero vector, which is physically impossible
/// after a valid sequence of unitary gates and measurements).
pub(crate) fn from_prob_to_shots<R: Rng>(
    data: &DumpProbability,
    shots: usize,
    rng: &mut R,
) -> SampleData {
    let mut count_map: HashMap<&Vec<u64>, usize> = HashMap::new();
    let dist = WeightedIndex::new(&data.probabilities).unwrap();

    (0..shots).for_each(|_| {
        count_map
            .entry(&data.basis_states[dist.sample(rng)])
            .and_modify(|c| *c += 1)
            .or_insert(1);
    });

    count_map
        .drain()
        .map(|(state, count)| (state.clone(), count))
        .unzip()
}
