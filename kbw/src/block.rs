// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Blocked quantum-state simulator.
//!
//! For qubit counts that exceed the local-memory capacity of a single simulator
//! instance, this module partitions the full qubit register into two sets:
//!
//! * **Local qubits**: simulated by individual [`QuantumExecution`] instances
//!   (one per global basis state). There are `2^num_global_qubits` such instances,
//!   each holding a state vector of size `2^num_local_qubits`.
//! * **Global qubits**: encoded in the *index* of the simulator instance. Gates
//!   on global qubits require a cross-block SWAP (implemented by
//!   [`BlockSimulator::swap`]) that physically moves the global qubit into the
//!   local register before the gate is applied.
//!
//! The number of local qubits is controlled by the `KBW_BLOCK_SIZE` environment
//! variable (default `20`).

use bimap::BiMap;
use ket::{error::KetError, execution::DumpData};
use smallvec::{smallvec, SmallVec};

use crate::quantum_execution::QuantumExecution;

/// Interface that a concrete simulator type must implement to participate in
/// block-partitioned simulation.
///
/// `S` is the inner simulator (e.g. [`DenseV2`](crate::DenseV2)) and `G` is an
/// arbitrary type that carries global state shared across all block instances
/// (e.g. GPU device handles, shared memory buffers).
pub trait BlockSimulator<S: QuantumExecution, G> {
    /// Allocate `2^num_global_qubits` simulator instances, each covering
    /// `num_local_qubits` qubits, together with the global shared data `G`.
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> Result<(Vec<S>, G), KetError>;

    /// Swap the amplitudes of `global_qubit` (encoded in the simulator index)
    /// with `local_qubit` (a qubit inside every block simulator).
    ///
    /// After the swap the two qubits have exchanged their roles: the formerly
    /// global qubit is now local and vice-versa.
    fn swap(
        global_data: &mut G,
        simulators: &mut [S],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    );

    /// Debug helper: print the entire joint state across all blocks.
    ///
    /// The default implementation panics: concrete backends that support
    /// printing can override this.
    fn print_global_state(_global_data: &G, _simulators: &[S]) {
        unimplemented!()
    }
}

enum Gate {
    X,
    Y,
    Z,
    H,
    P(f64),
    RX(f64),
    RY(f64),
    RZ(f64),
}

/// Top-level simulator that implements [`QuantumExecution`] over a partitioned
/// set of inner simulator instances.
///
/// Logical qubits are transparently mapped to physical qubits; when a gate
/// targets a *global* qubit the block calls [`BlockSimulator::swap`] to bring
/// that qubit into the local register first (an LRU eviction policy chooses
/// which local qubit to displace).
///
/// # Type parameters
///
/// * `G`: global shared state type (e.g. `()` for CPU backends, GPU handles
///   for the WGPU backend).
/// * `S`: inner simulator, must implement both [`QuantumExecution`] and
///   [`BlockSimulator<S, G>`].
pub struct Block<G, S: QuantumExecution + BlockSimulator<S, G>> {
    num_global_qubits: usize,
    num_local_qubits: usize,
    simulators: Vec<S>,
    global_data: G,
    /// Bidirectional map: logical qubit ↔ physical qubit.
    logical_to_physical: BiMap<usize, usize>,
    /// LRU timestamp per local physical qubit; updated on each gate access.
    qubit_last_access: Vec<u64>,
    access_counter: u64,
    #[allow(clippy::type_complexity)]
    gate_queue: Vec<(Gate, usize, SmallVec<[usize; 32]>, SmallVec<[bool; 16]>)>,
}

impl<G, S: QuantumExecution + BlockSimulator<S, G>> Block<G, S> {
    fn touch(&mut self, physical_qubit: usize) {
        if physical_qubit < self.num_local_qubits {
            self.access_counter += 1;
            self.qubit_last_access[physical_qubit] = self.access_counter;
        }
    }

    fn get_next_local_qubit(&self) -> usize {
        self.qubit_last_access
            .iter()
            .enumerate()
            .min_by_key(|(_, &timestamp)| timestamp)
            .map_or(0, |(index, _)| index) // Fallback for 0 qubits, though usually checked in new()
    }

    const fn is_local(&self, qubit: usize) -> bool {
        qubit < self.num_local_qubits
    }

    fn get_local_qubit(&mut self, logical_qubit: usize) -> usize {
        let physical_qubit = *self
            .logical_to_physical
            .get_by_left(&logical_qubit)
            .unwrap();

        if self.is_local(physical_qubit) {
            self.touch(physical_qubit);
            physical_qubit
        } else {
            self.execute_gates();
            let global_qubit = physical_qubit - self.num_local_qubits;
            let local_qubit = self.get_next_local_qubit();

            S::swap(
                &mut self.global_data,
                &mut self.simulators,
                self.num_global_qubits,
                self.num_local_qubits,
                global_qubit,
                local_qubit,
            );

            let old_logical_qubit = *self.logical_to_physical.get_by_right(&local_qubit).unwrap();

            self.logical_to_physical.insert(logical_qubit, local_qubit);
            self.logical_to_physical
                .insert(old_logical_qubit, physical_qubit);

            self.touch(local_qubit);

            local_qubit
        }
    }

    fn get_control_local_qubit(
        &self,
        control: &[usize],
    ) -> (SmallVec<[usize; 32]>, SmallVec<[bool; 16]>) {
        let mut new_control = SmallVec::new();
        let mut execute = smallvec![true; self.simulators.len()];
        for c in control {
            let c = *self.logical_to_physical.get_by_left(c).unwrap();
            if self.is_local(c) {
                new_control.push(c);
            } else {
                let c = 1usize << (c - self.num_local_qubits);
                for (i, s) in execute.iter_mut().enumerate() {
                    *s &= (i & c) == c;
                }
            }
        }

        (new_control, execute)
    }

    fn execute_gates(&mut self) {
        let queue = std::mem::take(&mut self.gate_queue);

        for (i, s) in self.simulators.iter_mut().enumerate() {
            for (gate, target, control, execute) in &queue {
                if execute[i] {
                    match gate {
                        Gate::X => {
                            s.pauli_x(*target, control.as_slice());
                        }
                        Gate::Y => {
                            s.pauli_y(*target, control.as_slice());
                        }
                        Gate::Z => {
                            s.pauli_z(*target, control.as_slice());
                        }
                        Gate::H => {
                            s.hadamard(*target, control.as_slice());
                        }
                        Gate::P(lambda) => {
                            s.phase(*lambda, *target, control.as_slice());
                        }
                        Gate::RX(theta) => s.rx(*theta, *target, control.as_slice()),
                        Gate::RY(theta) => s.ry(*theta, *target, control.as_slice()),
                        Gate::RZ(theta) => s.rz(*theta, *target, control.as_slice()),
                    }
                }
            }
        }
    }
}

impl<G, S: QuantumExecution + BlockSimulator<S, G>> QuantumExecution for Block<G, S> {
    fn new(num_qubits: usize) -> Result<Self, KetError>
    where
        Self: Sized,
    {
        // A single qubit cannot be meaningfully split into local and global
        // partitions (we need at least 1 local + 1 global qubit).
        if num_qubits <= 1 {
            return Err(KetError::ExecutionFailed);
        }

        let block_size = std::cmp::min(
            num_qubits - 1,
            std::env::var("KBW_BLOCK_SIZE")
                .unwrap_or_default()
                .parse::<usize>()
                .unwrap_or(20),
        );

        let num_local_qubits = block_size;
        let num_global_qubits = num_qubits - block_size;

        let (simulators, global_data) = S::new_blocks(num_local_qubits, num_global_qubits)?;

        let logical_to_physical = (0..num_qubits).map(|i| (i, i)).collect();
        let qubit_last_access = vec![0; num_local_qubits];

        Ok(Self {
            num_global_qubits,
            num_local_qubits,
            simulators,
            global_data,
            logical_to_physical,
            qubit_last_access,
            access_counter: 0,
            gate_queue: Vec::new(),
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::X, target, control, execute));
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::Y, target, control, execute));
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::Z, target, control, execute));
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue.push((Gate::H, target, control, execute));
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::P(lambda), target, control, execute));
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RX(theta), target, control, execute));
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RY(theta), target, control, execute));
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let target = self.get_local_qubit(target);
        let (control, execute) = self.get_control_local_qubit(control);
        self.gate_queue
            .push((Gate::RZ(theta), target, control, execute));
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        self.execute_gates();
        let target = self.get_local_qubit(target);
        self.simulators
            .iter_mut()
            .map(|s| s.measure_p1(target))
            .sum()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let target = self.get_local_qubit(target);
        for s in &mut self.simulators {
            s.measure_collapse(target, result, p);
        }
    }

    fn dump(&mut self, logical_qubits: &[usize]) -> DumpData {
        self.execute_gates();

        let physical_qubits: Vec<_> = logical_qubits
            .iter()
            .map(|q| *self.logical_to_physical.get_by_left(q).unwrap())
            .collect();

        let mut global_dump = DumpData::default();

        for (global_index, simulator) in self.simulators.iter_mut().enumerate() {
            let local_dump = simulator.dump(&physical_qubits);
            for (s, (r, i)) in local_dump.basis_states.iter().zip(
                local_dump
                    .amplitudes_real
                    .iter()
                    .zip(local_dump.amplitudes_imag.iter()),
            ) {
                let global_state = physical_qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .filter(|(_, q)| **q >= self.num_local_qubits)
                    .map(|(index, qubit)| (index, *qubit - self.num_local_qubits))
                    .map(|(index, qubit)| {
                        let bit = (global_index >> qubit) & 1;
                        bit << index
                    })
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                let state = usize::try_from(s[0]).unwrap() | global_state;
                global_dump.basis_states.push(vec![state as u64]);
                global_dump.amplitudes_real.push(*r);
                global_dump.amplitudes_imag.push(*i);
            }
        }

        #[cfg(debug_assertions)]
        {
            use itertools::Itertools;

            let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) =
                global_dump
                    .basis_states
                    .drain(..)
                    .zip(global_dump.amplitudes_real.drain(..))
                    .zip(global_dump.amplitudes_imag.drain(..))
                    .map(|((basis, real), imag)| (basis, real, imag))
                    .sorted_by(|a, b| a.0.cmp(&b.0))
                    .multiunzip();

            global_dump.basis_states = basis_states;
            global_dump.amplitudes_real = amplitudes_real;
            global_dump.amplitudes_imag = amplitudes_imag;
        }

        global_dump
    }

    fn clear(&mut self) {
        *self = Self::new(self.num_global_qubits + self.num_local_qubits).unwrap();
    }
}
