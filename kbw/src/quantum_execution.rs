// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! Core execution traits and the [`QubitManager`] adapter.
//!
//! This module defines two levels of abstraction:
//!
//! 1. **[`QuantumExecution`]**: the low-level trait that every simulator
//!    backend must implement.  It exposes one method per primitive quantum
//!    operation (gates, measurement, dump, …).
//!
//! 2. **[`QubitManager`]**: a generic adapter that wraps any
//!    `QuantumExecution` implementor and provides the higher-level
//!    [`LiveExecution`] and [`BatchExecution`] interfaces expected by the
//!    `ket` runtime.  It also owns the random-number generator (seeded from
//!    `KBW_SEED`) that is used for measurement sampling.

use itertools::Itertools;
use ket::{
    error::KetError,
    execution::{
        BatchExecution, DumpData, GradientStrategy, LiveExecution, NativeGate,
        QuantumExecution as KetQuantumExecution, SampleData,
    },
    ir::{
        gate::{GateInstruction, QuantumGate},
        hamiltonian::{Hamiltonian, Pauli},
    },
    process::QPUConfig,
};
use num::Integer;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f64::consts::FRAC_PI_2;

use crate::convert::{from_dump_to_prob, from_prob_to_shots};

/// Low-level interface that every simulator backend must satisfy.
///
/// Implementors are generic over the floating-point precision through the
/// [`FloatOps`](crate::FloatOps) trait bound used internally.  Consumers
/// interact with them exclusively through the `f64`-typed public methods
/// listed here so that gate angles and measurement probabilities are always
/// exchanged at full double precision.
pub trait QuantumExecution {
    /// Create a fresh `|0⟩^n` state for `num_qubits` qubits.
    ///
    /// Returns `Err` when the backend cannot handle the requested size
    /// (e.g. the dense backend rejects `num_qubits > 32`).
    fn new(num_qubits: usize) -> Result<Self, KetError>
    where
        Self: Sized;

    /// Apply the Pauli-X (NOT) gate, optionally controlled on `control` qubits.
    fn pauli_x(&mut self, target: usize, control: &[usize]);

    /// Apply the Pauli-Y gate, optionally controlled on `control` qubits.
    fn pauli_y(&mut self, target: usize, control: &[usize]);

    /// Apply the Pauli-Z gate, optionally controlled on `control` qubits.
    fn pauli_z(&mut self, target: usize, control: &[usize]);

    /// Apply the Hadamard gate, optionally controlled on `control` qubits.
    fn hadamard(&mut self, target: usize, control: &[usize]);

    /// Apply a phase gate `diag(1, e^{iλ})`, optionally controlled.
    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]);

    /// Apply an X-rotation `Rx(θ)`, optionally controlled.
    fn rx(&mut self, theta: f64, target: usize, control: &[usize]);

    /// Apply a Y-rotation `Ry(θ)`, optionally controlled.
    fn ry(&mut self, theta: f64, target: usize, control: &[usize]);

    /// Apply a Z-rotation `Rz(θ)`, optionally controlled.
    fn rz(&mut self, theta: f64, target: usize, control: &[usize]);

    /// Return the probability that `target` would be measured as `|1⟩`.
    ///
    /// Does **not** collapse the state.
    fn measure_p1(&mut self, target: usize) -> f64;

    /// Collapse the state after a measurement of `target`.
    ///
    /// `result` is the observed outcome and `p` is `1 / sqrt(P(result))`,
    /// the renormalisation factor.
    fn measure_collapse(&mut self, target: usize, result: bool, p: f64);

    /// Extract the (partial) state vector for the given `qubits`.
    fn dump(&mut self, qubits: &[usize]) -> DumpData;

    /// Draw samples from the distribution defined by `qubits`.
    ///
    /// The default implementation delegates to [`dump`](Self::dump) and then
    /// converts probabilities to a shot histogram.
    fn sample<R: Rng>(&mut self, qubits: &[usize], shots: usize, rng: &mut R) -> SampleData {
        let data = self.dump(qubits);
        from_prob_to_shots(&from_dump_to_prob(data), shots, rng)
    }

    /// Compute the expectation value `⟨ψ|H|ψ⟩` of a Hamiltonian.
    ///
    /// The default implementation converts each Pauli string to a Z-basis
    /// measurement via Hadamard / phase rotations, reads the resulting
    /// probability distribution, and accumulates the weighted parity sum.
    fn exp_value(&mut self, hamiltonian: &Hamiltonian) -> f64 {
        hamiltonian
            .pauli_strings
            .iter()
            .map(|pauli_terms| {
                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit, &[]),
                    Pauli::PauliY => {
                        self.phase(-FRAC_PI_2, term.qubit, &[]);
                        self.hadamard(term.qubit, &[]);
                    }
                    Pauli::PauliZ => {}
                });

                let dump_data = self.dump(&pauli_terms.iter().map(|term| term.qubit).collect_vec());
                let probabilities = from_dump_to_prob(dump_data);

                let result: f64 = probabilities
                    .basis_states
                    .iter()
                    .zip(probabilities.probabilities.iter())
                    .map(|(state, prob)| {
                        let parity = if state
                            .iter()
                            .fold(0, |acc, bit| acc + bit.count_ones())
                            .is_even()
                        {
                            1.0
                        } else {
                            -1.0
                        };
                        *prob * parity
                    })
                    .sum();

                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit, &[]),
                    Pauli::PauliY => {
                        self.hadamard(term.qubit, &[]);
                        self.phase(FRAC_PI_2, term.qubit, &[]);
                    }
                    Pauli::PauliZ => {}
                });

                result
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    /// Reset the simulator to the `|0⟩^n` state.
    fn clear(&mut self);
}

/// Wraps any [`QuantumExecution`] backend and adapts it to the
/// [`LiveExecution`] and [`BatchExecution`] interfaces expected by the
/// `ket` runtime.
///
/// A single `QubitManager` owns the RNG seeded from the `KBW_SEED`
/// environment variable so that all stochastic operations (measurement,
/// sampling) are reproducible when a seed is set.
pub struct QubitManager<S: QuantumExecution> {
    simulator: S,
    rng: std::sync::Mutex<StdRng>,
    num_qubits: usize,
}

impl<S: QuantumExecution> QubitManager<S> {
    /// Create a new `QubitManager` for `num_qubits` qubits.
    ///
    /// The RNG is seeded from the `KBW_SEED` environment variable when it
    /// contains a valid `u64`; otherwise a random seed is used.
    pub fn new(num_qubits: usize) -> Result<Self, KetError> {
        let seed = std::env::var("KBW_SEED")
            .unwrap_or_default()
            .parse::<u64>()
            .unwrap_or_else(|_| rand::random());

        Ok(Self {
            simulator: S::new(num_qubits)?,
            rng: std::sync::Mutex::new(StdRng::seed_from_u64(seed)),
            num_qubits,
        })
    }

    /// Build a [`QPUConfig`] configured for **live** (gate-by-gate) execution.
    ///
    /// Each gate is dispatched to the simulator immediately upon receipt;
    /// no circuit optimisation or decomposition is performed.
    pub fn make_live_config(num_qubits: usize, decompose: bool) -> Result<QPUConfig, KetError>
    where
        S: 'static,
    {
        Ok(QPUConfig {
            num_qubits,
            quantum_execution: Some(KetQuantumExecution::Live {
                qpu: Box::new(Self::new(num_qubits)?),
                decompose,
                native_gate_set: if decompose { Some(Box::new(())) } else { None },
            }),
        })
    }

    /// Build a [`QPUConfig`] configured for **batch** execution.
    ///
    /// The full circuit is assembled by the `ket` runtime before being handed
    /// to the simulator.  An optional `coupling_graph` restricts two-qubit
    /// connectivity, and `gradient` selects the gradient strategy.
    pub fn make_batch_config(
        num_qubits: usize,
        decompose: bool,
        coupling_graph: Option<Vec<(usize, usize)>>,
        gradient: GradientStrategy,
    ) -> Result<QPUConfig, KetError>
    where
        S: 'static,
    {
        Ok(QPUConfig {
            num_qubits,
            quantum_execution: Some(KetQuantumExecution::Batch {
                qpu: Box::new(Self::new(num_qubits)?),
                native_gate_set: if coupling_graph.is_some() {
                    Some(Box::new(()))
                } else {
                    None
                },
                gradient,
                exp_value: ket::execution::ExpValueStrategy::Native,
                coupling_graph,
                decompose,
            }),
        })
    }
}

impl<S: QuantumExecution> QubitManager<S> {
    fn apply_gate(&mut self, gate: &QuantumGate, target: usize, control: &[usize]) {
        match gate {
            QuantumGate::RotationX(theta) => {
                self.simulator.rx(theta.value(), target, control);
            }
            QuantumGate::RotationY(theta) => {
                self.simulator.ry(theta.value(), target, control);
            }
            QuantumGate::RotationZ(theta) => {
                self.simulator.rz(theta.value(), target, control);
            }
            QuantumGate::Phase(lambda) => {
                self.simulator.phase(lambda.value(), target, control);
            }
            QuantumGate::Hadamard => self.simulator.hadamard(target, control),
            QuantumGate::PauliX => self.simulator.pauli_x(target, control),
            QuantumGate::PauliY => self.simulator.pauli_y(target, control),
            QuantumGate::PauliZ => self.simulator.pauli_z(target, control),
        }
    }

    fn do_measure(&mut self, qubits: &[usize]) -> u64 {
        qubits
            .iter()
            .rev()
            .enumerate()
            .map(|(index, qubit)| {
                let p1 = self.simulator.measure_p1(*qubit);
                let p0 = (1.0 - p1).max(0.0);
                let result = WeightedIndex::new([p0, p1])
                    .unwrap()
                    .sample(&mut *self.rng.lock().unwrap());
                // Guard against division by zero when one outcome has negligible
                // probability: clamp to a small positive value before inverting.
                let prob = if result == 1 { p1 } else { p0 };
                let p = 1.0 / f64::sqrt(prob.max(f64::MIN_POSITIVE));
                self.simulator.measure_collapse(*qubit, result == 1, p);
                (result << index) as u64
            })
            .reduce(|a, b| a | b)
            .unwrap_or(0)
    }

    fn do_sample(&mut self, qubits: &[usize], shots: usize) -> SampleData {
        assert!(!qubits.is_empty());
        self.simulator
            .sample(qubits, shots, &mut *self.rng.lock().unwrap())
    }

    fn do_dump(&mut self, qubits: &[usize]) -> DumpData {
        self.simulator.dump(qubits)
    }

    fn do_exp_value(&mut self, hamiltonian: &Hamiltonian) -> f64 {
        self.simulator.exp_value(hamiltonian)
    }
}

impl<S: QuantumExecution + 'static> LiveExecution for QubitManager<S> {
    fn compute_gate(&mut self, gate: &GateInstruction) -> Result<(), ket::error::KetError> {
        let control: Vec<usize> = gate.control.iter().copied().collect();
        self.apply_gate(&gate.gate, gate.target, &control);
        Ok(())
    }

    fn compute_native_gates(&mut self, gates: &[NativeGate]) -> Result<(), ket::error::KetError> {
        self.compute_gates(gates)
    }

    fn measure(&mut self, qubits: &[usize]) -> Result<u64, ket::error::KetError> {
        Ok(self.do_measure(qubits))
    }

    fn dump(&mut self, qubits: &[usize]) -> Result<DumpData, ket::error::KetError> {
        Ok(self.do_dump(qubits))
    }

    fn sample(
        &mut self,
        qubits: &[usize],
        shots: usize,
    ) -> Result<SampleData, ket::error::KetError> {
        Ok(self.do_sample(qubits, shots))
    }

    fn exp_value(&mut self, hamiltonian: Hamiltonian) -> Result<f64, ket::error::KetError> {
        Ok(self.do_exp_value(&hamiltonian))
    }
}

impl<S: QuantumExecution + 'static> QubitManager<S> {
    fn compute_gates(&mut self, gates: &[NativeGate]) -> Result<(), KetError> {
        self.simulator.clear();
        for (name, angles, qubits) in gates {
            let control: Vec<usize> = qubits[..qubits.len().saturating_sub(1)].to_vec();
            let target = *qubits.last().unwrap();
            match name.as_str() {
                "x" => self.simulator.pauli_x(target, &[]),
                "y" => self.simulator.pauli_y(target, &[]),
                "z" => self.simulator.pauli_z(target, &[]),
                "h" => self.simulator.hadamard(target, &[]),
                "rx" => self.simulator.rx(angles[0], target, &[]),
                "ry" => self.simulator.ry(angles[0], target, &[]),
                "rz" => self.simulator.rz(angles[0], target, &[]),
                "p" => self.simulator.phase(angles[0], target, &[]),
                "cnot" => self.simulator.pauli_x(target, &control),
                "cz" => self.simulator.pauli_z(target, &control),
                _ => {
                    return Err(KetError::NativeGateUnsupported);
                }
            }
        }
        Ok(())
    }
}

impl<S: QuantumExecution + 'static> BatchExecution for QubitManager<S> {
    fn sample(
        &self,
        gates: &[GateInstruction],
        qubits_to_sample: &[usize],
        shots: usize,
    ) -> Result<SampleData, ket::error::KetError> {
        let mut sim = Self::new(self.num_qubits)?;
        for gate in gates {
            sim.compute_gate(gate)?;
        }
        Ok(sim.do_sample(qubits_to_sample, shots))
    }

    fn sample_native(
        &self,
        gates: &[NativeGate],
        qubits_to_sample: &[usize],
        shots: usize,
    ) -> Result<SampleData, ket::error::KetError> {
        let mut sim = Self::new(self.num_qubits)?;
        sim.compute_gates(gates)?;
        Ok(sim.do_sample(qubits_to_sample, shots))
    }

    fn exp_value(
        &self,
        gates: &[GateInstruction],
        hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, ket::error::KetError> {
        let mut sim = Self::new(self.num_qubits)?;
        for gate in gates {
            sim.compute_gate(gate)?;
        }
        Ok(hamiltonian_list
            .iter()
            .map(|h| sim.do_exp_value(h))
            .collect())
    }

    fn exp_value_native(
        &self,
        gates: &[NativeGate],
        hamiltonian_list: &[Hamiltonian],
    ) -> Result<Vec<f64>, ket::error::KetError> {
        let mut sim = Self::new(self.num_qubits)?;
        sim.compute_gates(gates)?;
        Ok(hamiltonian_list
            .iter()
            .map(|h| sim.do_exp_value(h))
            .collect())
    }

    fn gradient(
        &self,
        _gates: &[GateInstruction],
        _hamiltonian: &Hamiltonian,
    ) -> Result<(f64, Vec<f64>), KetError> {
        Err(KetError::NativeGateUnsupported) // KBW does not support native gradient
    }
}
