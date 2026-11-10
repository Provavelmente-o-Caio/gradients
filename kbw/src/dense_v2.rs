// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! In-place dense state-vector simulator (V2).
//!
//! # Algorithm
//!
//! Unlike the double-buffered [`crate::dense`] backend, `DenseV2` stores only
//! a **single** state vector.  Gates operate on pairs of amplitudes
//! `(|…0…⟩, |…1…⟩)` in-place by splitting each block of `2^(target+1)`
//! elements at the midpoint.  This halves the memory footprint at the cost of
//! a slightly more complex iteration pattern.
//!
//! The inner loop is parallelised with [Rayon](https://docs.rs/rayon).
//!
//! # Limitations
//!
//! * Maximum **32 qubits**.
//! * All operations run on the CPU.

#[cfg(feature = "block")]
use std::fmt::Display;

#[cfg(feature = "block")]
use crate::block::BlockSimulator;
use crate::quantum_execution::QuantumExecution;
use crate::{
    bitwise::{get_ctrl_mask, is_one_at},
    FloatOps,
};
use itertools::Itertools;
use ket::error::KetError;
use ket::execution::DumpData;
#[cfg(feature = "block")]
use num::complex::ComplexFloat;
use num::Zero;
use num::{Complex, One};
use rayon::prelude::*;

/// In-place dense state-vector simulator generic over float type `F`.
///
/// Use the pre-defined type alias [`DenseV2`](crate::DenseV2) (`f32`) for
/// typical workloads.
///
/// The inner `Vec` stores amplitudes as `[a_0, a_1, …, a_{2^n - 1}]`.
pub struct DenseV2<F: FloatOps>(Vec<Complex<F>>);

impl<F: FloatOps> DenseV2<F> {
    fn gate<G>(&mut self, gate_impl: G, target: usize, control: &[usize])
    where
        G: Fn((&mut Complex<F>, &mut Complex<F>)) + std::marker::Sync,
    {
        let half_block_size = 1 << target;
        let full_block_size = half_block_size << 1;
        let ctrl_mask: usize = get_ctrl_mask(control);

        let ctrl_mask_low = ctrl_mask & (half_block_size - 1);
        let ctrl_mask_high = ctrl_mask & !(full_block_size - 1);

        let inner_gate =
            |(upper, lower): (&mut [Complex<F>], &mut [Complex<F>])| {
                if ctrl_mask_low == 0 {
                    if upper.len() < 2048 {
                        upper.iter_mut().zip(lower.iter_mut()).for_each(&gate_impl);
                    } else {
                        upper
                            .par_iter_mut()
                            .zip(lower.par_iter_mut())
                            .for_each(&gate_impl);
                    }
                } else {
                    if upper.len() < 2048 {
                        upper.iter_mut().zip(lower.iter_mut()).enumerate().for_each(
                            |(index, op)| {
                                if (index & ctrl_mask_low) == ctrl_mask_low {
                                    gate_impl(op);
                                }
                            },
                        );
                    } else {
                        upper
                            .par_iter_mut()
                            .zip(lower.par_iter_mut())
                            .enumerate()
                            .for_each(|(index, op)| {
                                if (index & ctrl_mask_low) == ctrl_mask_low {
                                    gate_impl(op);
                                }
                            });
                    }
                }
            };

        self.0
            .par_chunks_mut(full_block_size)
            .enumerate()
            .for_each(|(chunk_id, state)| {
                if ((chunk_id * full_block_size) & ctrl_mask_high) == ctrl_mask_high {
                    inner_gate(state.split_at_mut(half_block_size));
                }
            });
    }

    fn diagonal_gate<G>(&mut self, gate_impl: G, target: usize, control: &[usize])
    where
        G: Fn(&mut Complex<F>) + std::marker::Sync,
    {
        let half_block_size = 1 << target;
        let full_block_size = half_block_size << 1;
        let ctrl_mask: usize = get_ctrl_mask(control);

        let ctrl_mask_low = ctrl_mask & (half_block_size - 1);
        let ctrl_mask_high = ctrl_mask & !(full_block_size - 1);

        let inner_gate = |lower: &mut [Complex<F>]| {
            if ctrl_mask_low == 0 {
                if lower.len() < 2048 {
                    lower.iter_mut().for_each(&gate_impl);
                } else {
                    lower.par_iter_mut().for_each(&gate_impl);
                }
            } else {
                if lower.len() < 2048 {
                    lower.iter_mut().enumerate().for_each(|(index, op)| {
                        if (index & ctrl_mask_low) == ctrl_mask_low {
                            gate_impl(op);
                        }
                    });
                } else {
                    lower.par_iter_mut().enumerate().for_each(|(index, op)| {
                        if (index & ctrl_mask_low) == ctrl_mask_low {
                            gate_impl(op);
                        }
                    });
                }
            }
        };

        self.0
            .par_chunks_mut(full_block_size)
            .enumerate()
            .for_each(|(chunk_id, state)| {
                if ((chunk_id * full_block_size) & ctrl_mask_high) == ctrl_mask_high {
                    let (_, lower) = state.split_at_mut(half_block_size);
                    inner_gate(lower);
                }
            });
    }
}

impl<F: FloatOps> QuantumExecution for DenseV2<F> {
    fn new(num_qubits: usize) -> Result<Self, KetError>
    where
        Self: Sized,
    {
        if num_qubits > 32 {
            return Err(KetError::ExecutionFailed);
        }

        let num_states = 1 << num_qubits;
        let mut state = vec![Complex::<F>::zero(); num_states];
        state[0] = Complex::<F>::one();

        Ok(Self(state))
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                std::mem::swap(ket0, ket1);
            },
            target,
            control,
        );
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                std::mem::swap(ket0, ket1);
                *ket0 *= -Complex::<F>::i();
                *ket1 *= Complex::<F>::i();
            },
            target,
            control,
        );
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        self.diagonal_gate(
            |ket1| {
                *ket1 *= -Complex::<F>::one();
            },
            target,
            control,
        );
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        self.gate(
            |(ket0, ket1)| {
                let tmp_ket0 = *ket0;
                let tmp_ket1 = *ket1;
                *ket0 = (tmp_ket0 + tmp_ket1) * F::FRAC_1_SQRT_2();
                *ket1 = (tmp_ket0 - tmp_ket1) * F::FRAC_1_SQRT_2();
            },
            target,
            control,
        );
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let phase = Complex::<F>::exp(Complex::<F>::i() * F::from(lambda).unwrap());

        self.diagonal_gate(
            |ket1: &mut Complex<F>| {
                *ket1 *= phase;
            },
            target,
            control,
        );
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let cos = F::cos(F::from(theta / 2.0).unwrap());
        let sin = F::sin(F::from(theta / 2.0).unwrap());

        self.gate(
            |(ket0, ket1)| {
                let r0 = ket0.re;
                let i0 = ket0.im;
                let r1 = ket1.re;
                let i1 = ket1.im;

                ket0.re = cos * r0 + sin * i1;
                ket0.im = cos * i0 - sin * r1;
                ket1.re = sin * i0 + cos * r1;
                ket1.im = -sin * r0 + cos * i1;
            },
            target,
            control,
        );
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let cos = F::cos(F::from(theta / 2.0).unwrap());
        let sin = F::sin(F::from(theta / 2.0).unwrap());

        self.gate(
            |(ket0, ket1)| {
                let r0 = ket0.re;
                let i0 = ket0.im;
                let r1 = ket1.re;
                let i1 = ket1.im;

                ket0.re = cos * r0 - sin * r1;
                ket0.im = cos * i0 - sin * i1;
                ket1.re = sin * r0 + cos * r1;
                ket1.im = sin * i0 + cos * i1;
            },
            target,
            control,
        );
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let phase_0 = Complex::<F>::exp(Complex::<F>::i() * F::from(-theta / 2.0).unwrap());
        let phase_1 = Complex::<F>::exp(Complex::<F>::i() * F::from(theta / 2.0).unwrap());

        self.gate(
            |(ket0, ket1)| {
                *ket0 *= phase_0;
                *ket1 *= phase_1;
            },
            target,
            control,
        );
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let state_mask = 1 << target;
        self.0
            .par_iter()
            .enumerate()
            .map(|(state, amp)| {
                if state & state_mask == state_mask {
                    amp.norm().powi(2)
                } else {
                    F::zero()
                }
            })
            .sum::<F>()
            .to_f64()
            .unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let state_mask = 1 << target;

        self.0.par_iter_mut().enumerate().for_each(|(state, amp)| {
            *amp = if (state & state_mask == state_mask) == result {
                *amp * F::from(p).unwrap()
            } else {
                Complex::<F>::zero()
            };
        });
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = self
            .0
            .iter()
            .enumerate()
            .filter(|(_state, amp)| amp.norm() > F::from(F::small_epsilon()).unwrap())
            .map(|(state, amp)| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| usize::from(is_one_at(state, *qubit)) << index)
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                (
                    Vec::from([state as u64]),
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
        self.0.fill(Complex::<F>::zero());
        self.0[0] = Complex::<F>::one();
    }
}

#[cfg(feature = "block")]
impl<F: FloatOps + Display> BlockSimulator<Self, ()> for DenseV2<F> {
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> Result<(Vec<Self>, ()), KetError> {
        let mut simulators = vec![Self::new(num_local_qubits)?];
        let num_states = 1 << num_local_qubits;

        for _ in 1..(1 << num_global_qubits) {
            simulators.push(Self(vec![Complex::<F>::zero(); num_states]));
        }

        Ok((simulators, ()))
    }

    fn swap(
        (): &mut (),
        simulators: &mut [Self],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    ) {
        let mut new_states =
            vec![vec![Complex::<F>::zero(); 1 << num_local_qubits]; 1 << num_global_qubits];

        new_states
            .iter_mut()
            .enumerate()
            .for_each(|(global_index, global)| {
                global
                    .iter_mut()
                    .enumerate()
                    .for_each(|(local_index, state)| {
                        let g_bit_pos = global_qubit;
                        let l_bit_pos = local_qubit;

                        let g_bit = (global_index >> g_bit_pos) & 1;
                        let l_bit = (local_index >> l_bit_pos) & 1;

                        let src_global = (global_index & !(1 << g_bit_pos)) | (l_bit << g_bit_pos);

                        let src_local = (local_index & !(1 << l_bit_pos)) | (g_bit << l_bit_pos);

                        *state = simulators[src_global].0[src_local];
                    });
            });

        for (s, new_state) in simulators.iter_mut().zip(new_states.drain(..)) {
            s.0 = new_state;
        }
    }

    fn print_global_state((): &(), simulators: &[Self]) {
        println!("==============");
        for (global, s) in simulators.iter().enumerate() {
            for (local, amp) in s.0.iter().enumerate() {
                println!("{global:>2} {local:<2}: ({:.4} {:+.4})", amp.re(), amp.im());
            }
        }
        println!("==============");
    }
}
