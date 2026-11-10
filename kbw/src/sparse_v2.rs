// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Vec-based sparse state-vector simulator (V2).
//!
//! # Algorithm
//!
//! Unlike the [`crate::sparse`] (`HashMap`) backend, `SparseV2` stores the
//! quantum state as a **sorted `Vec`** of `(state_key, amplitude)` pairs.
//! After gates that can increase the number of basis states (Hadamard, Rx,
//! Ry) the vector is re-sorted and a linear reduce pass merges duplicate keys
//! and prunes near-zero amplitudes.
//!
//! # Trade-offs vs [`crate::sparse`]
//!
//! | | Sparse (HashMap) | SparseV2 (Vec) |
//! |---|---|---|
//! | Per-gate lookup | O(1) amortised | O(log k) binary search (not done) |
//! | Cache efficiency | Poor (pointer-chasing) | Good (contiguous memory) |
//! | Parallelism | Diagonal gates only | Any gate |
//! | Post-Hadamard reduce | Entry-based | Sort + linear scan |

use crate::bitwise::{bit_flip_vec, ctrl_check_vec, is_one_at_vec};
use crate::quantum_execution::QuantumExecution;
use crate::FloatOps;
use itertools::Itertools;
use ket::error::KetError;
use ket::execution::DumpData;
use num::complex::Complex;
use num::One;
use rayon::prelude::*;

use crate::bitwise::StateKey;

type StateVec<F> = Vec<(StateKey, Complex<F>)>;

/// Vec-based sparse state-vector simulator generic over float type `F`.
///
/// Use the pre-defined type alias [`SparseV2`](crate::SparseV2) (`f32`) for
/// typical workloads.
pub struct SparseV2<F: FloatOps> {
    states: StateVec<F>,
    /// Number of `u64` words needed to represent a single basis state
    /// (= `ceil(num_qubits / 64)`).
    num_states: usize,
}

impl<F: FloatOps> SparseV2<F> {
    fn sort_and_reduce(&mut self) {
        if self.states.is_empty() {
            return;
        }

        self.states.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let mut write_idx = 0;

        let epsilon = F::from(F::small_epsilon()).unwrap();
        let epsilon_sqr = epsilon * epsilon;

        for read_idx in 1..self.states.len() {
            if self.states[read_idx].0 == self.states[write_idx].0 {
                let amp = self.states[read_idx].1;
                self.states[write_idx].1 += amp;
            } else {
                if self.states[write_idx].1.norm_sqr() > epsilon_sqr {
                    write_idx += 1;
                }

                if write_idx != read_idx {
                    self.states.swap(write_idx, read_idx);
                }
            }
        }

        if self.states[write_idx].1.norm_sqr() > epsilon_sqr {
            write_idx += 1;
        }

        self.states.truncate(write_idx);

        if self.states.capacity() > 4 * self.states.len() {
            self.states.shrink_to_fit();
        }
    }
}

impl<F: FloatOps> QuantumExecution for SparseV2<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError> {
        let num_states = num_qubits.div_ceil(64);
        let mut zero_state = StateKey::new();
        zero_state.resize(num_states, 0);
        let states = vec![(zero_state, Complex::<F>::one())];

        Ok(Self { states, num_states })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        if control.is_empty() {
            self.states.par_iter_mut().for_each(|(state, _)| {
                let temp_state = std::mem::take(state);
                *state = bit_flip_vec(temp_state, target);
            });
        } else {
            self.states.par_iter_mut().for_each(|(state, _)| {
                if ctrl_check_vec(state, control) {
                    let temp_state = std::mem::take(state);
                    *state = bit_flip_vec(temp_state, target);
                }
            });
        }
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let i_complex = Complex::<F>::i();

        let neg_i_complex = -i_complex;

        self.states.par_iter_mut().for_each(|(state, amp)| {
            if control.is_empty() || ctrl_check_vec(state, control) {
                *amp *= if is_one_at_vec(state, target) {
                    neg_i_complex
                } else {
                    i_complex
                };
                let temp_state = std::mem::take(state);
                *state = bit_flip_vec(temp_state, target);
            }
        });
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.states.par_iter_mut().for_each(|(state, amp)| {
            // Optimization: order checks cheapest to most expensive
            if is_one_at_vec(state, target)
                && (control.is_empty() || ctrl_check_vec(state, control))
            {
                *amp = -*amp;
            }
        });
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let inv_sqrt_2 = F::FRAC_1_SQRT_2();

        self.states = std::mem::take(&mut self.states)
            .into_par_iter()
            .flat_map_iter(|(state, mut amp)| {
                if control.is_empty() || ctrl_check_vec(&state, control) {
                    amp *= inv_sqrt_2;

                    let state_flipped = bit_flip_vec(state.clone(), target);
                    let phase_amp = if is_one_at_vec(&state, target) {
                        -amp
                    } else {
                        amp
                    };

                    itertools::Either::Left(
                        std::iter::once((state_flipped, amp))
                            .chain(std::iter::once((state, phase_amp))),
                    )
                } else {
                    itertools::Either::Right(std::iter::once((state, amp)))
                }
            })
            .collect();

        self.sort_and_reduce();
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        self.states.par_iter_mut().for_each(|(state, amp)| {
            // Optimization: check bit first (cheaper) then control
            if is_one_at_vec(state, target)
                && (control.is_empty() || ctrl_check_vec(state, control))
            {
                *amp *= phase;
            }
        });
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let half_theta = F::from(theta / 2.0).unwrap();
        let cos = F::cos(half_theta);
        let sin = F::sin(half_theta);

        self.states = std::mem::take(&mut self.states)
            .into_par_iter()
            .flat_map_iter(|(state, amp)| {
                if control.is_empty() || ctrl_check_vec(&state, control) {
                    let state_flipped = bit_flip_vec(state.clone(), target);

                    let flip_amp = Complex::new(amp.im * sin, -amp.re * sin);
                    let stay_amp = Complex::new(amp.re * cos, amp.im * cos);

                    itertools::Either::Left(
                        std::iter::once((state_flipped, flip_amp))
                            .chain(std::iter::once((state, stay_amp))),
                    )
                } else {
                    itertools::Either::Right(std::iter::once((state, amp)))
                }
            })
            .collect();

        self.sort_and_reduce();
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let half_theta = F::from(theta / 2.0).unwrap();
        let cos = F::cos(half_theta);
        let sin = F::sin(half_theta);

        self.states = std::mem::take(&mut self.states)
            .into_par_iter()
            .flat_map_iter(|(state, amp)| {
                if control.is_empty() || ctrl_check_vec(&state, control) {
                    let state_flipped = bit_flip_vec(state.clone(), target);

                    let sign = if is_one_at_vec(&state, target) {
                        -F::one()
                    } else {
                        F::one()
                    };

                    let flip_amp = Complex::new(amp.re * sign * sin, amp.im * sign * sin);
                    let stay_amp = Complex::new(amp.re * cos, amp.im * cos);

                    itertools::Either::Left(
                        std::iter::once((state_flipped, flip_amp))
                            .chain(std::iter::once((state, stay_amp))),
                    )
                } else {
                    itertools::Either::Right(std::iter::once((state, amp)))
                }
            })
            .collect();

        self.sort_and_reduce();
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let half_theta = F::from(theta / 2.0).unwrap();
        let i_complex = Complex::<F>::i();
        let phase_0 = Complex::<F>::exp(i_complex * -half_theta);
        let phase_1 = Complex::<F>::exp(i_complex * half_theta);

        self.states.par_iter_mut().for_each(|(state, amp)| {
            if control.is_empty() || ctrl_check_vec(state, control) {
                if is_one_at_vec(state, target) {
                    *amp *= phase_1;
                } else {
                    *amp *= phase_0;
                }
            }
        });
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        self.states
            .par_iter()
            .filter(|(state, _)| is_one_at_vec(state, target))
            .map(|(_, amp)| amp.norm_sqr())
            .sum::<F>()
            .to_f64()
            .unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let norm_factor = F::from(p).unwrap();

        self.states = std::mem::take(&mut self.states)
            .into_par_iter()
            .filter_map(|(state, amp)| {
                if is_one_at_vec(&state, target) == result {
                    Some((state, amp * norm_factor))
                } else {
                    None
                }
            })
            .collect();
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        self.states.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = self
            .states
            .par_iter()
            .map(|(state, amp)| {
                let mut basis_state: Vec<u64> = qubits
                    .iter()
                    .rev()
                    .chunks(64)
                    .into_iter()
                    .map(|chunk| {
                        chunk
                            .into_iter()
                            .enumerate()
                            .map(|(index, qubit)| {
                                usize::from(is_one_at_vec(state, *qubit)) << index
                            })
                            .fold(0, |a, b| a | b) as u64
                    })
                    .collect();
                basis_state.reverse();

                (
                    basis_state,
                    amp.re.to_f64().unwrap(),
                    amp.im.to_f64().unwrap(),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn clear(&mut self) {
        let mut zero_state = StateKey::new();
        zero_state.resize(self.num_states, 0);
        self.states = vec![(zero_state, Complex::<F>::one())];
    }
}
