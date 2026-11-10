// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Basic block: a straight-line sequence of quantum gate instructions.
//!
//! A [`BasicBlock`] is the primary container for gate sequences in the Libket
//! IR. It maintains an index of qubit read/write dependencies that is
//! used by the qubit-mapping pass, and it applies lightweight algebraic
//! optimizations (gate merging and cancellation) whenever a new instruction is
//! appended.
//!
//! ## Gate simplification
//!
//! When a new gate is appended, the block scans backwards through the existing
//! instruction list for the first instruction that shares any qubit:
//! - If the two instructions have the same target and controls **and** are
//!   inverses of each other, both are cancelled (removed).
//! - If they have the same target and controls **and** are on the same rotation
//!   axis, their parameters are added and emitted as a single merged gate.
//! - If neither simplification applies, the gate is appended unconditionally.

use super::gate::GateInstruction;
use crate::{
    error::KetError,
    ir::gate::{DecomposedGate, GatePropriety, Param, QuantumGate},
};
use itertools::Itertools;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

/// Per-qubit dependency record maintained by a [`BasicBlock`].
///
/// One entry exists in [`BasicBlock::qubits_op`] for each qubit that appears
/// as a gate target. The record accumulates the *most general* gate propriety
/// seen for that qubit (via [`GatePropriety::restrict`]) and the set of
/// control qubits that have influenced it.
#[derive(Debug, Clone, Default, Serialize)]
pub struct QubitOp {
    /// The most general [`GatePropriety`] of any gate that has targeted this qubit.
    pub propriety: GatePropriety,
    /// The set of all control qubit indices for gates targeting this qubit.
    pub read_qubits: BTreeSet<usize>,
}

/// A straight-line sequence of quantum gate instructions.
///
/// A [`BasicBlock`] is created empty and gates are appended one at a time via
/// [`BasicBlock::append_gate`] or in bulk via [`BasicBlock::append_block`].
/// Each append operation attempts lightweight algebraic simplifications:
/// adjacent inverse gate pairs are cancelled and consecutive same-axis
/// rotation gates are merged into one.
///
/// The block also maintains a dependency index (`qubits_op`) consumed by the
/// qubit-allocation and circuit-mapping passes.
#[derive(Debug, Clone, Default)]
pub struct BasicBlock {
    /// The ordered list of gate instructions in this block.
    pub gates: Vec<GateInstruction>,
    /// Per-qubit dependency records (propriety and read-control sets).
    pub qubits_op: BTreeMap<usize, QubitOp>,
    /// Accumulated global phase (in radians), or `None` if no global phase
    /// has been added to this block.
    pub global: Option<f64>,
}

impl BasicBlock {
    /// Creates a new, empty [`BasicBlock`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a gate instruction to the internal list and updates the
    /// dependency maps and propriety counters.
    fn push_gate(&mut self, inst: GateInstruction, check_propiety: bool) {
        if check_propiety {
            self.qubits_op
                .entry(inst.target)
                .or_default()
                .propriety
                .restrict(inst.gate.propriety());

            for c in &inst.control {
                self.qubits_op
                    .entry(inst.target)
                    .or_default()
                    .read_qubits
                    .insert(*c);
            }
        }

        self.gates.push(inst);
    }

    /// Removes the gate at `index` and updates the dependency maps and
    /// propriety counters accordingly.
    fn remove_gate(&mut self, index: usize) {
        self.gates.remove(index);
    }

    /// Replaces the gate at `index` with `merged_gate` and updates the
    /// propriety counters to reflect the change.
    fn merge_gate(&mut self, index: usize, merged_gate: QuantumGate, check_propiety: bool) {
        let inst = &self.gates[index];
        if check_propiety {
            self.qubits_op
                .entry(inst.target)
                .or_default()
                .propriety
                .restrict(inst.gate.propriety());
        }
        self.gates[index].gate = merged_gate;
    }

    /// Attempts to append `new_inst` to the block with algebraic simplification.
    ///
    /// The method scans backwards through the existing instructions looking for
    /// the first instruction that shares any qubit with `new_inst`. If that
    /// instruction has the same target and controls:
    /// - and is the inverse of `new_inst`: both are cancelled (removed);
    /// - and is the same rotation axis: the two are merged into one.
    ///
    /// `epsilon` controls the threshold below which a rotation angle is
    /// considered zero (defaults to `1e-10`).
    pub(crate) fn append_instruction(
        &mut self,
        new_inst: GateInstruction,
        epsilon: Option<f64>,
        check_propiety: bool,
        min_index: usize,
    ) {
        let epsilon = epsilon.unwrap_or(1e-10);

        if new_inst.gate.is_near_zero(epsilon) {
            return;
        }

        let mut bypass_count = 0;
        for (index, inst) in self.gates.iter().enumerate().rev() {
            if index < min_index {
                break;
            }
            if bypass_count > 64 {
                break;
            }
            let shares_qubits = inst.target == new_inst.target
                || inst.control.contains(&new_inst.target)
                || new_inst.control.contains(&inst.target)
                || inst.control.iter().any(|c| new_inst.control.contains(c));

            if shares_qubits {
                let same_target = inst.target == new_inst.target;
                let same_controls = inst.control == new_inst.control;

                if same_target && same_controls {
                    if inst.gate.inverse() == new_inst.gate {
                        return self.remove_gate(index);
                    } else if let Some(merged_gate) = inst.gate.merge(&new_inst.gate) {
                        if merged_gate.is_near_zero(epsilon) {
                            return self.remove_gate(index);
                        }
                        return self.merge_gate(index, merged_gate, check_propiety);
                    }
                }

                if !inst.commutes_with(&new_inst) {
                    break;
                }
                bypass_count += 1;
            }
        }

        self.push_gate(new_inst, check_propiety);
    }

    /// Appends an uncontrolled gate to the block.
    pub fn append_gate(&mut self, gate: QuantumGate, target: usize) {
        self.append_instruction(GateInstruction::new(gate, target), None, true, 0);
    }

    /// Return a new basic block with the inverse of the circuit.
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            gates: self
                .gates
                .iter()
                .rev()
                .map(GateInstruction::inverse)
                .collect_vec(),
            global: self.global.map(|g| -g),
            qubits_op: self.qubits_op.clone(),
        }
    }

    /// Returns a copy of this block with `control_qubits` added to every gate.
    ///
    /// # Errors
    /// Propagates any [`KetError`] returned by [`GateInstruction::control`].
    pub fn control(&self, control_qubits: &[usize]) -> Result<Self, KetError> {
        let gates: Result<Vec<_>, _> = self
            .gates
            .iter()
            .map(|gate| gate.control(control_qubits))
            .collect();
        let gates = gates?;

        let qubits_op = self
            .qubits_op
            .iter()
            .map(|(target, op)| {
                let mut op = op.to_owned();
                for c in control_qubits {
                    op.read_qubits.insert(*c);
                }
                (*target, op)
            })
            .collect();

        let mut result = Self {
            gates,
            global: None,
            qubits_op,
        };

        if let Some(phase) = &self.global {
            result.append_instruction(
                GateInstruction {
                    gate: QuantumGate::Phase(Param::Value(*phase)),
                    target: control_qubits[0],
                    control: control_qubits
                        .iter()
                        .skip(1)
                        .copied()
                        .collect::<BTreeSet<_>>(),
                    control_locked: false,
                    is_approximated: false,
                    decomposed: None,
                },
                None,
                true,
                0,
            );
        }

        Ok(result)
    }

    /// Enable approximate decomposition enabled on all gate instructions.
    pub fn enable_approximated_decomposition(&mut self) {
        self.gates
            .iter_mut()
            .for_each(GateInstruction::enable_approximated_decomposition);
    }

    /// Lock control sets locked on all gates.
    pub fn lock_control(&mut self) {
        self.gates
            .iter_mut()
            .for_each(GateInstruction::lock_control);
    }

    /// Appends all instructions from `block` into `self`, applying algebraic
    /// simplifications using the given `epsilon` tolerance.
    ///
    /// After draining `block.gates`, the global phases are summed and the
    /// `qubits_op` dependency maps are merged (restricting propriety to the
    /// more-general of the two).
    pub fn append_block(&mut self, block: Self) {
        self.append_block_with_epsilon_and_min_index(block, None, 0);
    }

    pub fn append_block_with_epsilon_and_min_index(
        &mut self,
        mut block: Self,
        epsilon: Option<f64>,
        min_index: usize,
    ) {
        for inst in block.gates {
            self.append_instruction(inst, epsilon, false, min_index);
        }

        self.global = match (self.global, block.global) {
            (None, None) => None,
            (None, Some(g)) | (Some(g), None) => Some(g),
            (Some(g1), Some(g2)) => Some(g1 + g2),
        };

        for (&qubit, op) in &mut block.qubits_op {
            let self_op = self.qubits_op.entry(qubit).or_default();
            self_op.propriety.restrict(op.propriety);
            self_op.read_qubits.append(&mut op.read_qubits);
        }
    }

    /// Sum a global phase in the block.
    pub fn add_global_phase(&mut self, phase: f64) {
        self.global = Some(self.global.unwrap_or_default() + phase);
    }

    /// Returns the highest qubit index referenced by any gate in this block,
    /// or `None` if the block is empty.
    #[must_use]
    pub fn max_qubit_index(&self) -> Option<usize> {
        let max_read = self.qubits_op.values().flat_map(|op| &op.read_qubits).max();

        let max_written = self.qubits_op.keys().max();

        match (max_read, max_written) {
            (None, None) => None,
            (None, Some(&w)) => Some(w),
            (Some(&r), None) => Some(r),
            (Some(&r), Some(&w)) => Some(r.max(w)),
        }
    }

    /// Flattens decomposed multi-qubit gates into a single sequence of
    /// [`DecomposedGate`].
    ///
    /// When `parameters` is provided, any symbolic [`Param::Ref`] values are
    /// resolved to their concrete floating-point equivalents before output.
    #[must_use]
    pub fn flat_gates(
        &self,
        parameters: Option<&[f64]>,
        shift: Option<(usize, usize, f64)>, // (param_index, instance_index, shift_amount)
    ) -> Vec<DecomposedGate> {
        let mut instance_counts = std::collections::HashMap::new();

        self.gates
            .iter()
            .flat_map(|gate| {
                if let Some(gates) = &gate.decomposed {
                    gates
                        .iter()
                        .map(|g| match g {
                            DecomposedGate::U(qg, target) => DecomposedGate::U(*qg, *target),
                            DecomposedGate::CNOT(c, t) => DecomposedGate::CNOT(*c, *t),
                        })
                        .collect_vec()
                } else {
                    assert!(gate.control.is_empty());
                    let qg = &gate.gate;
                    let mut final_gate = if let Some(parameters) = parameters {
                        qg.set_parameter(parameters)
                    } else {
                        *qg
                    };
                    if let Some((shift_param, shift_inst, shift_amt)) = shift {
                        let mut p_idx = None;
                        match qg {
                            QuantumGate::RotationX(Param::Ref { index, .. })
                            | QuantumGate::RotationY(Param::Ref { index, .. })
                            | QuantumGate::RotationZ(Param::Ref { index, .. })
                            | QuantumGate::Phase(Param::Ref { index, .. }) => {
                                p_idx = Some(*index);
                            }
                            _ => {}
                        }
                        if let Some(idx) = p_idx {
                            if idx == shift_param {
                                let count = instance_counts.entry(idx).or_insert(0);
                                if *count == shift_inst {
                                    final_gate = match final_gate {
                                        QuantumGate::RotationX(Param::Value(v)) => {
                                            QuantumGate::RotationX(Param::Value(v + shift_amt))
                                        }
                                        QuantumGate::RotationY(Param::Value(v)) => {
                                            QuantumGate::RotationY(Param::Value(v + shift_amt))
                                        }
                                        QuantumGate::RotationZ(Param::Value(v)) => {
                                            QuantumGate::RotationZ(Param::Value(v + shift_amt))
                                        }
                                        QuantumGate::Phase(Param::Value(v)) => {
                                            QuantumGate::Phase(Param::Value(v + shift_amt))
                                        }
                                        _ => final_gate,
                                    };
                                }
                                *count += 1;
                            }
                        }
                    }
                    vec![(final_gate, gate.target).into()]
                }
            })
            .collect_vec()
    }

    /// Forces the propriety classification of every qubit entry to at most
    /// [`GatePropriety::Diagonal`], even if the actual gate sequence is more
    /// general. Used when an externally-verified analysis guarantees the
    /// circuit is diagonal.
    pub fn set_as_diagonal(&mut self) {
        self.qubits_op.values_mut().for_each(|op| {
            op.propriety.broaden(GatePropriety::Diagonal);
        });
    }

    /// Forces the propriety classification of every qubit entry to at most
    /// [`GatePropriety::Permutation`], even if the actual gate sequence is
    /// more general. Used when an externally-verified analysis guarantees the
    /// circuit only permutes basis states.
    pub fn set_as_permutation(&mut self) {
        self.qubits_op.values_mut().for_each(|op| {
            op.propriety.broaden(GatePropriety::Permutation);
        });
    }
}

#[derive(Debug, Clone)]
pub struct TemplateBlock {
    pub name: String,
    pub qubits: Vec<usize>,
    pub controls: Vec<usize>,
    pub control_locked: bool,
    pub enable_approximated_decomposition: bool,
    pub is_adjoint: bool,
    pub is_self_inverse: bool,
    pub is_diagonal: bool,
    pub is_permutation: bool,
}

impl TemplateBlock {
    pub fn cancels(&self, other: &TemplateBlock) -> bool {
        self.name == other.name
            && self.qubits == other.qubits
            && self.controls == other.controls
            && self.control_locked == other.control_locked
            && self.is_diagonal == other.is_diagonal
            && self.is_permutation == other.is_permutation
            && (self.is_self_inverse || self.is_adjoint != other.is_adjoint)
    }
}

#[derive(Debug, Clone)]
pub enum BlockElement {
    Basic(BasicBlock),
    Template(TemplateBlock),
}

#[derive(Debug, Clone, Default)]
pub struct Block {
    pub elements: Vec<BlockElement>,
}

impl Block {
    pub fn new() -> Self {
        Self {
            elements: vec![BlockElement::Basic(BasicBlock::new())],
        }
    }

    pub fn append_basic_block(&mut self, bb: BasicBlock) {
        if bb.gates.is_empty() && bb.global.is_none() {
            return;
        }
        if let Some(BlockElement::Basic(last_bb)) = self.elements.last_mut() {
            last_bb.append_block(bb);
        } else {
            self.elements.push(BlockElement::Basic(bb));
        }
    }

    pub fn append_template(&mut self, tb: TemplateBlock, min_index: usize) {
        // If min_index == 0, no gates have been sent to live execution from this block.
        // It is perfectly safe to cancel the last template block.
        if min_index == 0 {
            if let Some(BlockElement::Template(last_tb)) = self.elements.last() {
                if last_tb.cancels(&tb) {
                    self.elements.pop();
                    return;
                }
            }
        }
        self.elements.push(BlockElement::Template(tb));
    }

    pub fn append_basic_block_with_epsilon_and_min_index(
        &mut self,
        block: BasicBlock,
        epsilon: Option<f64>,
        min_index: usize,
    ) {
        if block.gates.is_empty() && block.global.is_none() {
            return;
        }
        if let Some(BlockElement::Basic(last_bb)) = self.elements.last_mut() {
            last_bb.append_block_with_epsilon_and_min_index(block, epsilon, min_index);
        } else {
            self.elements.push(BlockElement::Basic(block));
        }
    }

    pub fn append_block_with_epsilon_and_min_index(
        &mut self,
        block: Block,
        epsilon: Option<f64>,
        min_index: usize,
    ) {
        for element in block.elements {
            match element {
                BlockElement::Basic(bb) => {
                    self.append_basic_block_with_epsilon_and_min_index(bb, epsilon, min_index)
                }
                BlockElement::Template(tb) => self.append_template(tb, min_index),
            }
        }
    }
    pub fn append_block(&mut self, block: Block) {
        for element in block.elements {
            match element {
                BlockElement::Basic(bb) => self.append_basic_block(bb),
                BlockElement::Template(tb) => self.append_template(tb, 0),
            }
        }
    }

    pub fn append_gate(&mut self, gate: QuantumGate, target: usize) {
        let inst = GateInstruction {
            gate,
            target,
            control: std::collections::BTreeSet::new(),
            control_locked: false,
            is_approximated: false,
            decomposed: None,
        };
        if let Some(BlockElement::Basic(last_bb)) = self.elements.last_mut() {
            last_bb.append_instruction(inst, None, false, 0);
        } else {
            let mut bb = BasicBlock::new();
            bb.append_instruction(inst, None, false, 0);
            self.elements.push(BlockElement::Basic(bb));
        }
    }

    pub fn inverse(&self) -> Self {
        let elements = self
            .elements
            .iter()
            .rev()
            .map(|e| match e {
                BlockElement::Basic(bb) => BlockElement::Basic(bb.inverse()),
                BlockElement::Template(tb) => {
                    let mut new_tb = tb.clone();
                    if !new_tb.is_self_inverse {
                        new_tb.is_adjoint = !new_tb.is_adjoint;
                    }
                    BlockElement::Template(new_tb)
                }
            })
            .collect();
        Self { elements }
    }

    pub fn control(&self, control_qubits: &[usize]) -> Result<Self, KetError> {
        let elements: Result<Vec<_>, _> = self
            .elements
            .iter()
            .map(|e| match e {
                BlockElement::Basic(bb) => bb.control(control_qubits).map(BlockElement::Basic),
                BlockElement::Template(tb) => {
                    let mut new_tb = tb.clone();
                    if new_tb.control_locked {
                        return Ok(BlockElement::Template(new_tb));
                    }
                    for &c in control_qubits {
                        if new_tb.qubits.contains(&c) {
                            return Err(KetError::ControlTargetConflict);
                        }
                        if new_tb.controls.contains(&c) {
                            return Err(KetError::DuplicateControlQubit);
                        }
                        new_tb.controls.push(c);
                    }
                    new_tb.controls.sort_unstable();
                    Ok(BlockElement::Template(new_tb))
                }
            })
            .collect();
        Ok(Self {
            elements: elements?,
        })
    }

    pub fn lock_control(&mut self) {
        for e in &mut self.elements {
            match e {
                BlockElement::Basic(bb) => bb.lock_control(),
                BlockElement::Template(tb) => tb.control_locked = true,
            }
        }
    }

    pub fn enable_approximated_decomposition(&mut self) {
        for e in &mut self.elements {
            match e {
                BlockElement::Basic(bb) => bb.enable_approximated_decomposition(),
                BlockElement::Template(tb) => tb.enable_approximated_decomposition = true,
            }
        }
    }

    pub fn add_global_phase(&mut self, phase: f64) {
        if let Some(BlockElement::Basic(last_bb)) = self.elements.last_mut() {
            last_bb.add_global_phase(phase);
        } else {
            let mut bb = BasicBlock::new();
            bb.add_global_phase(phase);
            self.elements.push(BlockElement::Basic(bb));
        }
    }

    pub fn max_qubit_index(&self) -> Option<usize> {
        self.elements
            .iter()
            .filter_map(|e| match e {
                BlockElement::Basic(bb) => bb.max_qubit_index(),
                BlockElement::Template(tb) => {
                    let mq = tb.qubits.iter().max().copied();
                    let mc = tb.controls.iter().max().copied();
                    mq.into_iter().chain(mc).max()
                }
            })
            .max()
    }

    pub fn set_as_diagonal(&mut self) {
        for e in &mut self.elements {
            match e {
                BlockElement::Basic(bb) => bb.set_as_diagonal(),
                BlockElement::Template(tb) => tb.is_diagonal = true,
            }
        }
    }

    pub fn set_as_permutation(&mut self) {
        for e in &mut self.elements {
            match e {
                BlockElement::Basic(bb) => bb.set_as_permutation(),
                BlockElement::Template(tb) => tb.is_permutation = true,
            }
        }
    }

    pub fn flatten(&self) -> Result<BasicBlock, KetError> {
        self.flatten_with_min_index(0)
    }

    /// Flattens the block, never simplifying gates across `min_index`.
    ///
    /// Gates whose flattened position is below `min_index` were already sent
    /// to a live backend, so they are copied as they are and no later gate may
    /// cancel or merge with them.
    pub fn flatten_with_min_index(&self, min_index: usize) -> Result<BasicBlock, KetError> {
        let mut final_block = BasicBlock::new();
        for element in &self.elements {
            match element {
                BlockElement::Basic(bb) => {
                    final_block.append_block_with_epsilon_and_min_index(
                        bb.clone(),
                        None,
                        min_index,
                    );
                }
                BlockElement::Template(tb) => {
                    let mut instantiated =
                        crate::ir::macros::instantiate_macro(&tb.name, &tb.qubits, tb.is_adjoint)?;
                    if !tb.controls.is_empty() {
                        instantiated = instantiated.control(&tb.controls)?;
                    }
                    if tb.control_locked {
                        instantiated.lock_control();
                    }
                    if tb.enable_approximated_decomposition {
                        instantiated.enable_approximated_decomposition();
                    }
                    let inner_flat = instantiated.flatten()?;
                    final_block
                        .append_block_with_epsilon_and_min_index(inner_flat, None, min_index);
                    // Apply property overrides AFTER merging, so qubits_op entries exist.
                    if tb.is_diagonal {
                        for &q in tb.qubits.iter().chain(tb.controls.iter()) {
                            final_block
                                .qubits_op
                                .entry(q)
                                .or_default()
                                .propriety
                                .broaden(crate::ir::gate::GatePropriety::Diagonal);
                        }
                    } else if tb.is_permutation {
                        for &q in tb.qubits.iter().chain(tb.controls.iter()) {
                            final_block
                                .qubits_op
                                .entry(q)
                                .or_default()
                                .propriety
                                .broaden(crate::ir::gate::GatePropriety::Permutation);
                        }
                    }
                }
            }
        }
        Ok(final_block)
    }
}

impl From<BasicBlock> for Block {
    fn from(bb: BasicBlock) -> Self {
        let mut b = Self::new();
        b.append_basic_block(bb);
        b
    }
}
