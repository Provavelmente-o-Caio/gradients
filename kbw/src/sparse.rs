// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! HashMap-based sparse state-vector simulator.
//!
//! # Algorithm
//!
//! Only basis states with non-zero amplitude are stored. The state is
//! represented as `HashMap<Vec<u64>, Complex<F>>` keyed by the bit-string of
//! each qubit (packed into 64-bit words).  This representation is efficient
//! when the quantum state has few non-zero amplitudes (e.g. immediately after
//! initialisation or in low-entanglement circuits).
//!
//! Amplitudes whose squared norm falls below `F::small_epsilon()^2` are
//! pruned during gate application to keep the map compact.
//!
//! The hash map uses the [xxHash64](https://docs.rs/twox-hash) algorithm for
//! fast, non-cryptographic hashing of byte-array keys.
//!
//! # Trade-offs vs [`crate::dense`]
//!
//! | | Dense | Sparse |
//! |---|---|---|
//! | Memory | O(2^n) | O(k) where k = non-zero amplitudes |
//! | Gate cost | O(2^n) parallel | O(k) sequential |
//! | Best for | genral case | one-hot encoding |

use crate::bitwise::{bit_flip_vec, ctrl_check_vec, is_one_at_vec, StateKey};
use crate::quantum_execution::QuantumExecution;
use crate::FloatOps;
use itertools::Itertools;
use ket::error::KetError;
use ket::execution::DumpData;
use num::complex::Complex;
use num::One;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

type StateMap<F> = HashMap<StateKey, Complex<F>, FxBuildHasher>;

/// HashMap-based sparse state-vector simulator generic over float type `F`.
///
/// Use the pre-defined type alias [`Sparse`](crate::Sparse) (`f32`) for
/// typical workloads.
pub struct Sparse<F: FloatOps> {
    state_0: StateMap<F>,
    state_1: StateMap<F>,
    state: bool,
    num_states: usize,
}

impl<F: FloatOps> Sparse<F> {
    const fn get_states(&mut self) -> (&mut StateMap<F>, &mut StateMap<F>) {
        self.state = !self.state;
        if self.state {
            (&mut self.state_1, &mut self.state_0)
        } else {
            (&mut self.state_0, &mut self.state_1)
        }
    }

    const fn get_current_state_mut(&mut self) -> &mut StateMap<F> {
        if self.state {
            &mut self.state_0
        } else {
            &mut self.state_1
        }
    }

    const fn get_current_state(&self) -> &StateMap<F> {
        if self.state {
            &self.state_0
        } else {
            &self.state_1
        }
    }
}

impl<F: FloatOps> QuantumExecution for Sparse<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        let num_states = num_qubits.div_ceil(64);

        let mut state_0 = StateMap::<F>::default();

        let mut zero = StateKey::new();
        zero.resize(num_states, 0);

        state_0.insert(zero, Complex::<F>::one());

        Ok(Self {
            state_0,
            state_1: StateMap::<F>::default(),
            state: true,
            num_states,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();
        let has_control = !control.is_empty();

        current_state.drain().for_each(|(state, amp)| {
            next_state.insert(
                if !has_control || ctrl_check_vec(&state, control) {
                    bit_flip_vec(state, target)
                } else {
                    state
                },
                amp,
            );
        });
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();
        let has_control = !control.is_empty();
        let i_complex = Complex::<F>::i();
        let neg_i_complex = -i_complex;

        current_state.drain().for_each(|(state, mut amp)| {
            if !has_control || ctrl_check_vec(&state, control) {
                amp *= if is_one_at_vec(&state, target) {
                    neg_i_complex
                } else {
                    i_complex
                };
                next_state.insert(bit_flip_vec(state, target), amp);
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();
        let has_control = !control.is_empty();

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if is_one_at_vec(state, target) && (!has_control || ctrl_check_vec(state, control)) {
                *amp = -*amp;
            }
        });
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();
        let has_control = !control.is_empty();

        let epsilon = F::from(F::small_epsilon()).unwrap();
        let epsilon_sqr = epsilon * epsilon;
        let inv_sqrt_2 = F::FRAC_1_SQRT_2();

        current_state.drain().for_each(|(state, mut amp)| {
            if !has_control || ctrl_check_vec(&state, control) {
                amp *= inv_sqrt_2;
                let state_flipped = bit_flip_vec(state.clone(), target);

                match next_state.entry(state_flipped) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += amp;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(amp);
                    }
                }

                amp = if is_one_at_vec(&state, target) {
                    -amp
                } else {
                    amp
                };

                match next_state.entry(state) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += amp;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(amp);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();
        let has_control = !control.is_empty();

        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if is_one_at_vec(state, target) && (!has_control || ctrl_check_vec(state, control)) {
                *amp *= phase;
            }
        });
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();
        let has_control = !control.is_empty();

        let half_theta = F::from(theta / 2.0).unwrap();
        let cos = F::cos(half_theta);
        let sin = F::sin(half_theta);

        let epsilon = F::from(F::small_epsilon()).unwrap();
        let epsilon_sqr = epsilon * epsilon;

        current_state.drain().for_each(|(state, amp)| {
            if !has_control || ctrl_check_vec(&state, control) {
                let state_flipped = bit_flip_vec(state.clone(), target);

                // Branch 1: Flip
                let flip_term = Complex::new(amp.im * sin, -amp.re * sin);
                match next_state.entry(state_flipped) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += flip_term;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(flip_term);
                    }
                }

                // Branch 2: Stay
                let stay_term = Complex::new(amp.re * cos, amp.im * cos);
                match next_state.entry(state) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += stay_term;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(stay_term);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (current_state, next_state) = self.get_states();
        let has_control = !control.is_empty();

        let half_theta = F::from(theta / 2.0).unwrap();
        let cos = F::cos(half_theta);
        let sin = F::sin(half_theta);

        let epsilon = F::from(F::small_epsilon()).unwrap();
        let epsilon_sqr = epsilon * epsilon;

        current_state.drain().for_each(|(state, amp)| {
            if !has_control || ctrl_check_vec(&state, control) {
                let state_flipped = bit_flip_vec(state.clone(), target);

                // Branch 1: Flip
                let sign = if is_one_at_vec(&state, target) {
                    -F::one()
                } else {
                    F::one()
                };
                let flip_term = Complex::new(amp.re * sign * sin, amp.im * sign * sin);
                match next_state.entry(state_flipped) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += flip_term;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(flip_term);
                    }
                }

                // Branch 2: Stay
                let stay_term = Complex::new(amp.re * cos, amp.im * cos);
                match next_state.entry(state) {
                    Entry::Occupied(mut entry) => {
                        let c_amp = entry.get_mut();
                        *c_amp += stay_term;
                        if c_amp.norm_sqr() < epsilon_sqr {
                            entry.remove();
                        }
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(stay_term);
                    }
                }
            } else {
                next_state.insert(state, amp);
            }
        });
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let current_state = self.get_current_state_mut();
        let has_control = !control.is_empty();

        let half_theta = F::from(theta / 2.0).unwrap();
        let i_complex = Complex::<F>::i();
        let phase_0 = Complex::<F>::exp(i_complex * -half_theta);
        let phase_1 = Complex::<F>::exp(i_complex * half_theta);

        current_state.par_iter_mut().for_each(|(state, amp)| {
            if !has_control || ctrl_check_vec(state, control) {
                if is_one_at_vec(state, target) {
                    *amp *= phase_1;
                } else {
                    *amp *= phase_0;
                }
            }
        });
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let current_state = self.get_current_state();

        current_state
            .iter()
            .map(|(state, amp)| {
                if is_one_at_vec(state, target) {
                    amp.norm_sqr() // Optimization: norm_sqr is faster than norm().powi(2)
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            .to_f64()
            .unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let (current_state, next_state) = self.get_states();

        let norm_factor = F::from(p).unwrap();

        current_state.drain().for_each(|(state, amp)| {
            if is_one_at_vec(&state, target) == result {
                next_state.insert(state, amp * norm_factor);
            }
        });
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let state = self.get_current_state();

        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = state
            .iter()
            .sorted_by_key(|x| x.0)
            .map(|(state, amp)| {
                let mut basis_state: Vec<u64> = qubits
                    .iter()
                    .rev()
                    .chunks(64)
                    .into_iter()
                    .map(|qubits| {
                        qubits
                            .into_iter()
                            .enumerate()
                            .map(|(index, qubit)| {
                                usize::from(is_one_at_vec(state, *qubit)) << index
                            })
                            .reduce(|a, b| a | b)
                            .unwrap_or(0) as u64
                    })
                    .collect();
                basis_state.reverse();

                (
                    basis_state,
                    amp.re.to_f64().unwrap(),
                    amp.im.to_f64().unwrap(),
                )
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn clear(&mut self) {
        self.state_0 = StateMap::<F>::default();
        self.state_1 = StateMap::<F>::default();

        let mut zero = StateKey::new();
        zero.resize(self.num_states, 0);

        self.state_0.insert(zero, Complex::<F>::one());
        self.state = true;
    }
}
