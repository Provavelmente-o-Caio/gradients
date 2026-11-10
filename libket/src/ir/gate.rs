// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Intermediate representation (IR) for quantum gate instructions.
//!
//! This module defines the core data types used to represent individual quantum
//! gates ([`QuantumGate`]), their numeric parameters ([`Param`]), and the
//! composed gate-instruction type ([`GateInstruction`]) that carries a gate
//! together with its target and control qubits.
//!
//! Gate decomposition: the process of rewriting a multi-controlled gate into
//! a sequence of primitive one- and two-qubit gates: is handled by the
//! methods on [`GateInstruction`].

use itertools::{chain, Itertools};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_4, FRAC_PI_8, PI},
};

use crate::{
    decompose::{self, x::CXMode, Algorithm, AuxMode, DepthMode},
    error::KetError,
    ir::gate::QuantumGate::{PauliX, RotationZ},
    matrix::{Cf64, Matrix},
};

/// A numeric parameter for rotation and phase gates.
///
/// A `Param` is either a concrete floating-point value ([`Param::Value`]) or a
/// symbolic reference to an entry in a parameter vector ([`Param::Ref`]). The
/// `Ref` variant enables deferred parameter binding for variational algorithms
/// where the same circuit is evaluated at multiple parameter points.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Param {
    /// A concrete, pre-bound rotation angle (in radians).
    Value(f64),
    /// A symbolic reference to a parameter vector entry.
    ///
    /// The effective angle is `value * multiplier`. The `value` field stores
    /// the base value at binding time; `multiplier` can be adjusted (e.g.,
    /// negated for inversion) without touching the underlying vector.
    Ref {
        /// Zero-based index into the process parameter vector.
        index: usize,
        /// Scaling factor applied to the parameter value.
        multiplier: f64,
        /// The concrete value of the parameter at binding time.
        value: f64,
    },
}

impl Param {
    /// Returns the mathematical inverse of the parameter (i.e., the negation).
    ///
    /// For `Value(v)` this returns `Value(-v)`; for `Ref` this negates the
    /// `multiplier` field.
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            Self::Value(v) => Self::Value(-v),
            Self::Ref {
                index,
                multiplier,
                value,
            } => Self::Ref {
                index: *index,
                multiplier: -multiplier,
                value: *value,
            },
        }
    }

    /// Returns the current floating-point value of the parameter.
    ///
    /// For [`Param::Value`] this is a direct read. For [`Param::Ref`] the
    /// effective value is `value * multiplier`.
    #[must_use]
    pub fn value(&self) -> f64 {
        match self {
            Self::Value(value) => *value,
            Self::Ref {
                multiplier, value, ..
            } => *value * *multiplier,
        }
    }

    /// Binds a symbolic [`Param::Ref`] to a concrete value from `parameters`.
    ///
    /// If `self` is already a [`Param::Value`] it is returned unchanged.
    /// Otherwise the result is `Value(base_value * multiplier * parameters[index])`.
    #[must_use]
    pub fn set_parameter(&self, parameters: &[f64]) -> Self {
        match self {
            Self::Value(_) => *self,
            Self::Ref {
                multiplier,
                value: _,
                index,
            } => Self::Value(*multiplier * parameters[*index]),
        }
    }

    /// Attempts to add two parameters together.
    ///
    /// Returns `Some(Value(a + b))` only when **both** operands are
    /// [`Param::Value`]. Any combination involving a [`Param::Ref`] returns
    /// `None`, because symbolic parameters cannot be merged algebraically at
    /// compile time.
    #[must_use]
    pub fn add(&self, other: &Self) -> Option<Self> {
        let (Self::Value(a), Self::Value(b)) = (self, other) else {
            return None;
        };
        Some(Self::Value(a + b))
    }

    /// Returns `true` if the parameter's effective rotation angle is within
    /// `epsilon` of zero.
    ///
    /// For [`Param::Ref`] only the `multiplier` is inspected, since the actual
    /// parameter value is not known at compile time.
    #[must_use]
    pub fn is_near_zero(&self, epsilon: f64) -> bool {
        match self {
            Self::Value(v) => v.abs() <= epsilon,
            Self::Ref { multiplier, .. } => multiplier.abs() <= epsilon,
        }
    }
}

/// Classification of a gate's action on the computational basis.
///
/// Variants are ordered from least to most general so that the derived `Ord`
/// implementation makes `Unitary > Permutation > Diagonal > Identity` hold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Default, Eq, PartialOrd, Ord)]
pub enum GatePropriety {
    /// The gate is the identity: it leaves all basis states unchanged.
    #[default]
    Identity,
    /// The gate is diagonal in the computational basis (it only adds phases).
    Diagonal,
    /// The gate permutes computational-basis states without introducing superposition.
    Permutation,
    /// The gate is a general unitary that creates superposition.
    Unitary,
}

impl GatePropriety {
    pub fn restrict(&mut self, propriety: GatePropriety) {
        *self = match (*self, propriety) {
            (GatePropriety::Identity, other) | (other, GatePropriety::Identity) => other,
            (GatePropriety::Diagonal, GatePropriety::Diagonal) => GatePropriety::Diagonal,
            (_, GatePropriety::Unitary) | (GatePropriety::Unitary, _) => GatePropriety::Unitary,
            _ => GatePropriety::Permutation,
        };
    }

    pub fn broaden(&mut self, propriety: GatePropriety) {
        *self = match (*self, propriety) {
            (GatePropriety::Identity, _) | (_, GatePropriety::Identity) => GatePropriety::Identity,
            (
                GatePropriety::Diagonal | GatePropriety::Permutation | GatePropriety::Unitary,
                GatePropriety::Diagonal,
            )
            | (GatePropriety::Diagonal, GatePropriety::Permutation) => GatePropriety::Diagonal,
            (GatePropriety::Permutation | GatePropriety::Unitary, GatePropriety::Permutation) => {
                GatePropriety::Permutation
            }
            (other, GatePropriety::Unitary) => other,
        }
    }
}

/// The fundamental set of single-qubit quantum gates supported by the compiler.
///
/// Each variant corresponds to a standard gate from the universal gate set.
/// Rotation gates carry a [`Param`] that encodes the rotation angle in radians.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QuantumGate {
    /// The Pauli-X (bit-flip) gate.
    PauliX,
    /// The Pauli-Y gate.
    PauliY,
    /// The Pauli-Z (phase-flip) gate.
    PauliZ,
    /// Rotation about the X-axis by angle `θ`: `Rx(θ) = exp(-i θ/2 · X)`.
    RotationX(Param),
    /// Rotation about the Y-axis by angle `θ`: `Ry(θ) = exp(-i θ/2 · Y)`.
    RotationY(Param),
    /// Rotation about the Z-axis by angle `θ`: `Rz(θ) = exp(-i θ/2 · Z)`.
    RotationZ(Param),
    /// Phase gate: `P(λ) = diag(1, e^{iλ})`.
    Phase(Param),
    /// The Hadamard gate.
    Hadamard,
}

impl QuantumGate {
    /// Returns `true` if the gate acts as a permutation of computational-basis
    /// states (including X-rotations by `±π`).
    #[must_use]
    pub fn is_permutation(&self) -> bool {
        match self {
            Self::RotationX(param) | Self::RotationY(param) => {
                if let Param::Value(value) = param {
                    (value.abs() - PI).abs() <= 1e-10
                        || 2.0f64.mul_add(-PI, value.abs()).abs() <= 1e-10
                } else {
                    false
                }
            }
            Self::Hadamard => false,
            _ => true,
        }
    }

    /// Returns the parameter index and multiplier if this gate holds a
    /// [`Param::Ref`] (i.e., a symbolic variational parameter).
    ///
    /// Returns `Some((index, multiplier))` for `RotationX`, `RotationY`,
    /// `RotationZ`, and `Phase` variants whose parameter is a [`Param::Ref`].
    /// Returns `None` for all other gates or when the parameter is a
    /// concrete [`Param::Value`].
    ///
    /// Used by the parameter-shift rule to identify which gate instructions
    /// correspond to a given parameter index.
    #[must_use]
    pub fn param_index(&self) -> Option<(usize, f64)> {
        match self {
            Self::RotationX(Param::Ref {
                index, multiplier, ..
            })
            | Self::RotationY(Param::Ref {
                index, multiplier, ..
            })
            | Self::RotationZ(Param::Ref {
                index, multiplier, ..
            })
            | Self::Phase(Param::Ref {
                index, multiplier, ..
            }) => Some((*index, *multiplier)),
            _ => None,
        }
    }

    /// Returns `true` if the gate is diagonal in the computational basis.
    #[must_use]
    pub const fn is_diagonal(&self) -> bool {
        matches!(self, Self::PauliZ | Self::RotationZ(_) | Self::Phase(_))
    }

    /// Returns the adjoint (Hermitian conjugate / inverse) of this gate.
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            Self::PauliX => Self::PauliX,
            Self::PauliY => Self::PauliY,
            Self::PauliZ => Self::PauliZ,
            Self::RotationX(p) => Self::RotationX(p.inverse()),
            Self::RotationY(p) => Self::RotationY(p.inverse()),
            Self::RotationZ(p) => Self::RotationZ(p.inverse()),
            Self::Phase(p) => Self::Phase(p.inverse()),
            Self::Hadamard => Self::Hadamard,
        }
    }

    /// Classifies the gate according to its action on the computational basis.
    #[must_use]
    pub fn propriety(&self) -> GatePropriety {
        match self {
            Self::PauliX | Self::PauliY => GatePropriety::Permutation,
            Self::RotationX(_) | Self::RotationY(_) => {
                if self.is_permutation() {
                    GatePropriety::Permutation
                } else {
                    GatePropriety::Unitary
                }
            }
            Self::PauliZ | Self::RotationZ(_) | Self::Phase(_) => GatePropriety::Diagonal,
            Self::Hadamard => GatePropriety::Unitary,
        }
    }

    /// Constructs an S gate (`Phase(π/2)`).
    #[must_use]
    pub const fn s() -> Self {
        Self::Phase(Param::Value(FRAC_PI_2))
    }

    /// Constructs an S† gate (`Phase(-π/2)`).
    #[must_use]
    pub fn sd() -> Self {
        Self::Phase(Param::Value(-FRAC_PI_2))
    }

    /// Constructs a T gate (`Phase(π/4)`).
    #[must_use]
    pub const fn t() -> Self {
        Self::Phase(Param::Value(FRAC_PI_4))
    }

    /// Constructs a T† gate (`Phase(-π/4)`).
    #[must_use]
    pub fn td() -> Self {
        Self::Phase(Param::Value(-FRAC_PI_4))
    }

    /// Constructs a √T gate (`Phase(π/8)`).
    #[must_use]
    pub const fn sqrt_t() -> Self {
        Self::Phase(Param::Value(FRAC_PI_8))
    }

    /// Constructs a √T† gate (`Phase(-π/8)`).
    #[must_use]
    pub fn sqrt_td() -> Self {
        Self::Phase(Param::Value(-FRAC_PI_8))
    }

    /// Returns the 2×2 unitary matrix representation of the gate.
    pub(crate) fn matrix(&self) -> Matrix {
        match self {
            Self::PauliX => [[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]],
            Self::PauliY => [[0.0.into(), -Cf64::i()], [Cf64::i(), 0.0.into()]],
            Self::PauliZ => [[1.0.into(), 0.0.into()], [0.0.into(), (-1.0).into()]],
            Self::RotationX(angle) => {
                let angle = angle.value();
                [
                    [(angle / 2.0).cos().into(), -Cf64::i() * (angle / 2.0).sin()],
                    [-Cf64::i() * (angle / 2.0).sin(), (angle / 2.0).cos().into()],
                ]
            }
            Self::RotationY(angle) => {
                let angle = angle.value();
                [
                    [(angle / 2.0).cos().into(), (-(angle / 2.0).sin()).into()],
                    [(angle / 2.0).sin().into(), (angle / 2.0).cos().into()],
                ]
            }
            Self::RotationZ(angle) => {
                let angle = angle.value();
                [
                    [(-Cf64::i() * (angle / 2.0)).exp(), 0.0.into()],
                    [0.0.into(), (Cf64::i() * (angle / 2.0)).exp()],
                ]
            }
            Self::Phase(angle) => [
                [1.0.into(), 0.0.into()],
                [0.0.into(), (Cf64::i() * angle.value()).exp()],
            ],
            Self::Hadamard => [
                [(1.0 / 2.0f64.sqrt()).into(), (1.0 / 2.0f64.sqrt()).into()],
                [(1.0 / 2.0f64.sqrt()).into(), (-1.0 / 2.0f64.sqrt()).into()],
            ],
        }
    }

    /// Returns the SU(2) (determinant-1) matrix representative of the gate.
    ///
    /// Some gates (e.g. `PauliX`) have a natural matrix that is in U(2) but not
    /// SU(2). This method returns the closest SU(2) representative, which is
    /// needed for certain decomposition algorithms.
    pub(crate) fn su2_matrix(&self) -> Matrix {
        match self {
            Self::PauliX => Self::RotationX(Param::Value(PI)).matrix(),
            Self::PauliY => Self::RotationY(Param::Value(PI)).matrix(),
            Self::PauliZ => Self::RotationZ(Param::Value(PI)).matrix(),
            Self::Phase(angle) => Self::RotationZ(*angle).matrix(),
            Self::Hadamard => [
                [-Cf64::i() * FRAC_1_SQRT_2, -Cf64::i() * FRAC_1_SQRT_2],
                [-Cf64::i() * FRAC_1_SQRT_2, Cf64::i() * FRAC_1_SQRT_2],
            ],
            _ => self.matrix(),
        }
    }

    /// Attempts to merge two consecutive same-type rotation gates into one.
    ///
    /// Returns `Some(merged)` when both gates are the same rotation axis and
    /// their parameters can be added (see [`Param::add`]), or `None` otherwise.
    pub fn merge(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::RotationX(a), Self::RotationX(b)) => a.add(b).map(QuantumGate::RotationX),
            (Self::RotationY(a), Self::RotationY(b)) => a.add(b).map(QuantumGate::RotationY),
            (Self::RotationZ(a), Self::RotationZ(b)) => a.add(b).map(QuantumGate::RotationZ),
            (Self::Phase(a), Self::Phase(b)) => a.add(b).map(QuantumGate::Phase),
            _ => None,
        }
    }

    /// Returns `true` if the gate's rotation angle is within `epsilon` of zero,
    /// making it effectively the identity.
    ///
    /// Only parametric gates (`Rotation*` and `Phase`) can be near zero;
    /// non-parametric gates always return `false`.
    #[must_use]
    pub fn is_near_zero(&self, epsilon: f64) -> bool {
        match self {
            Self::RotationX(p) | Self::RotationY(p) | Self::RotationZ(p) | Self::Phase(p) => {
                p.is_near_zero(epsilon)
            }
            _ => false,
        }
    }

    /// Binds all symbolic parameters in this gate to their concrete values.
    ///
    /// For non-parametric gates (`PauliX`, `PauliY`, `PauliZ`, `Hadamard`)
    /// this is a no-op and returns a copy of `self`. For parametric gates the
    /// [`Param`] is resolved via [`Param::set_parameter`].
    #[must_use]
    pub fn set_parameter(&self, parameters: &[f64]) -> Self {
        match self {
            Self::PauliX | Self::PauliY | Self::PauliZ | Self::Hadamard => *self,
            Self::RotationX(param) => Self::RotationX(param.set_parameter(parameters)),
            Self::RotationY(param) => Self::RotationY(param.set_parameter(parameters)),
            Self::RotationZ(param) => Self::RotationZ(param.set_parameter(parameters)),
            Self::Phase(param) => Self::Phase(param.set_parameter(parameters)),
        }
    }
}

/// A primitive quantum gate resulting from the decomposition of a multi-controlled gate.
///
/// Multi-controlled gates are decomposed into a sequence of single-qubit gates
/// ([`DecomposedGate::U`]) and two-qubit CNOT gates ([`DecomposedGate::CNOT`]).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum DecomposedGate {
    /// A single-qubit gate instruction applied to a target qubit.
    U(QuantumGate, usize),
    /// A CNOT gate instruction applied to a control and a target qubit.
    CNOT(usize, usize),
}

impl DecomposedGate {
    #[must_use]
    pub fn inverse(&self) -> Self {
        match self {
            DecomposedGate::U(gate, t) => DecomposedGate::U(gate.inverse(), *t),
            DecomposedGate::CNOT(c, t) => DecomposedGate::CNOT(*c, *t),
        }
    }
}

impl From<(QuantumGate, usize)> for DecomposedGate {
    fn from(value: (QuantumGate, usize)) -> Self {
        let (gate, target) = value;
        Self::U(gate, target)
    }
}

/// A fully-specified gate instruction applied to one or more logical qubits.
///
/// A [`GateInstruction`] bundles a [`QuantumGate`] with its `target` qubit index
/// and a set of `control` qubit indices. It also tracks whether new
/// controls may still be added (`control_locked`), whether approximate
/// decomposition algorithms may be used (`is_approximated`), and the optional
/// pre-computed gate decomposition (`decomposed`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GateInstruction {
    /// The quantum gate to apply.
    pub gate: QuantumGate,
    /// The index of the target qubit.
    pub target: usize,
    /// The set of control qubit indices.
    pub control: BTreeSet<usize>,
    /// When `true`, no additional control qubits may be added to this instruction.
    pub control_locked: bool,
    /// When `true`, the gate decomposition may use relative-phase decompostion that
    /// reduce the gate count.
    pub is_approximated: bool,
    /// The gate decomposed into primitive (CNOT + single-qubit) operations.
    pub decomposed: Option<Vec<DecomposedGate>>,
}

impl From<&DecomposedGate> for GateInstruction {
    fn from(value: &DecomposedGate) -> Self {
        match value {
            DecomposedGate::U(gate, target) => Self {
                gate: *gate,
                target: *target,
                control: BTreeSet::new(),
                control_locked: false,
                is_approximated: false,
                decomposed: None,
            },
            DecomposedGate::CNOT(control, target) => Self {
                gate: PauliX,
                target: *target,
                control: BTreeSet::from([*control]),
                control_locked: false,
                is_approximated: false,
                decomposed: None,
            },
        }
    }
}

impl GateInstruction {
    /// Creates a new uncontrolled gate instruction for `gate` applied to `target`.
    #[must_use]
    pub const fn new(gate: QuantumGate, target: usize) -> Self {
        Self {
            gate,
            target,
            control: BTreeSet::new(),
            control_locked: false,
            is_approximated: false,
            decomposed: None,
        }
    }

    /// Returns the adjoint (inverse) of this gate instruction.
    ///
    /// The control set and lock status are preserved. If the instruction has
    /// already been decomposed, the decomposition is reversed and each
    /// constituent gate is individually inverted.
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self {
            gate: self.gate.inverse(),
            target: self.target,
            control: self.control.clone(),
            control_locked: self.control_locked,
            is_approximated: self.is_approximated,
            decomposed: self
                .decomposed
                .as_ref()
                .map(|gates| gates.iter().rev().map(DecomposedGate::inverse).collect()),
        }
    }

    /// Adds `control_qubits` to the control set of this instruction.
    ///
    /// # Errors
    /// - [`KetError::DuplicateControlQubit`]: one of `control_qubits` is
    ///   already present in the control set.
    /// - [`KetError::ControlTargetConflict`]: one of `control_qubits` is the
    ///   same qubit as `self.target`.
    ///
    /// If the instruction's controls are locked ([`GateInstruction::control_locked`]
    /// is `true`), the call is a no-op and `Ok(self.clone())` is returned.
    pub fn control(&self, control_qubits: &[usize]) -> Result<Self, KetError> {
        if self.control_locked {
            return Ok(self.clone());
        }

        let mut control = self.control.clone();
        control.extend(control_qubits.iter());

        if self.gate.param_index().is_some() {
            Err(KetError::ControlParameterizedGate)
        } else if control.len() != (control_qubits.len() + self.control.len()) {
            Err(KetError::DuplicateControlQubit)
        } else if control.contains(&self.target) {
            Err(KetError::ControlTargetConflict)
        } else {
            Ok(Self {
                gate: self.gate,
                target: self.target,
                control,
                control_locked: false,
                is_approximated: false,
                decomposed: None,
            })
        }
    }

    /// Enable approximate decomposition.
    ///
    /// When `is_approximated` is `true`, decomposition algorithms use a
    /// smaller circuits are permitted.
    pub fn enable_approximated_decomposition(&mut self) {
        self.is_approximated = true;
    }

    /// Lock the control set.
    ///
    /// A locked instruction will silently ignore any subsequent calls to
    /// [`GateInstruction::control`], returning a clone of itself unchanged.
    pub fn lock_control(&mut self) {
        self.control_locked = true;
    }

    /// Returns `true` if this instruction commutes with `other`.
    ///
    /// Two instructions commute when both are diagonal, or when they apply the
    /// same gate to the same target.
    #[must_use]
    pub fn commutes_with(&self, other: &Self) -> bool {
        self.gate.is_diagonal() && other.gate.is_diagonal()
            || (self.gate == other.gate && self.target == other.target)
    }

    /// Decomposes a multi-controlled gate into primitive two-qubit operations.
    ///
    /// The decomposition is written into `self.decomposed`. If the instruction
    /// has no control qubits, or if `decomposed` is already populated, this
    /// method is a no-op.
    ///
    /// `clean_auxiliary_range` is a half-open range `(start, end)` of qubit
    /// indices that are guaranteed to be in the |0⟩ state (clean auxiliary).
    pub fn decompose(&mut self, clean_auxiliary_range: (usize, usize)) {
        let control = self.control.iter().copied().collect_vec();
        if !self.control.is_empty() && self.decomposed.is_none() {
            self.decomposed = Some(match self.gate {
                QuantumGate::PauliX => Self::decompose_x(
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
                QuantumGate::PauliY => Self::decompose_y(
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
                QuantumGate::PauliZ => Self::decompose_z(
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
                QuantumGate::RotationX(_)
                | QuantumGate::RotationY(_)
                | QuantumGate::RotationZ(_) => Self::decompose_r(
                    self.gate,
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
                QuantumGate::Phase(angle) => Self::decompose_phase(
                    angle.value(),
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
                QuantumGate::Hadamard => Self::decompose_hadamard(
                    self.target,
                    &control,
                    self.is_approximated,
                    clean_auxiliary_range,
                ),
            });
        }
    }

    /// Decomposes a multi-controlled general rotation gate.
    fn decompose_r(
        gate: QuantumGate,
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        if control.len() == 1 {
            return decompose::u2::cu2(gate.matrix(), control[0], target);
        }

        let num_clean_aux = clean_axillary_range.1 - clean_axillary_range.0;
        let network_num_aux = Algorithm::NetworkU2(CXMode::C3X).aux_needed(control.len());

        if network_num_aux <= num_clean_aux {
            let aux_qubits: Vec<_> =
                (clean_axillary_range.0..clean_axillary_range.0 + network_num_aux).collect();

            decompose::network::network(
                gate,
                control,
                &aux_qubits,
                target,
                CXMode::C3X,
                approximated,
            )
        } else {
            decompose::su2::decompose(gate, control, target, DepthMode::Linear, approximated)
        }
    }

    /// Decomposes a multi-controlled Pauli-X gate.
    fn decompose_x(
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        if control.len() <= 4 {
            return decompose::x::c2to4x::c1to4x(control, target, approximated);
        }

        let num_clean_aux = clean_axillary_range.1 - clean_axillary_range.0;

        let network_c2x_num_aux = Algorithm::NetworkPauli(CXMode::C2X).aux_needed(control.len());

        if num_clean_aux >= network_c2x_num_aux {
            let aux_qubits: Vec<_> =
                (clean_axillary_range.0..clean_axillary_range.0 + network_c2x_num_aux).collect();

            return decompose::network::network(
                QuantumGate::PauliX,
                control,
                &aux_qubits,
                target,
                CXMode::C2X,
                approximated,
            );
        }

        let network_c3x_num_aux = Algorithm::NetworkPauli(CXMode::C3X).aux_needed(control.len());

        if num_clean_aux >= network_c3x_num_aux {
            let aux_qubits: Vec<_> =
                (clean_axillary_range.0..clean_axillary_range.0 + network_c3x_num_aux).collect();

            return decompose::network::network(
                QuantumGate::PauliX,
                control,
                &aux_qubits,
                target,
                CXMode::C3X,
                approximated,
            );
        }
        let v_chain_c3x_clean_num_aux =
            Algorithm::VChain(CXMode::C3X, AuxMode::Clean).aux_needed(control.len());

        if num_clean_aux >= v_chain_c3x_clean_num_aux {
            let aux_qubits: Vec<_> = (clean_axillary_range.0
                ..clean_axillary_range.0 + v_chain_c3x_clean_num_aux)
                .collect();

            return decompose::x::v_chain::v_chain(
                control,
                &aux_qubits,
                target,
                AuxMode::Clean,
                CXMode::C3X,
                approximated,
            );
        }

        let num_dirty_aux = clean_axillary_range.1 - 1 - control.len();

        let v_chain_c2x_dirty_num_aux =
            Algorithm::VChain(CXMode::C2X, AuxMode::Dirty).aux_needed(control.len());

        if num_dirty_aux >= v_chain_c2x_dirty_num_aux {
            let all_qubits = control.iter().copied().chain([target]).collect_vec();

            let aux_qubits: Vec<_> = (0..v_chain_c3x_clean_num_aux + 1 + control.len())
                .filter(|qubit| !all_qubits.contains(qubit))
                .collect();

            decompose::x::v_chain::v_chain(
                control,
                &aux_qubits,
                target,
                AuxMode::Dirty,
                CXMode::C2X,
                approximated,
            )
        } else if num_clean_aux >= 1 {
            decompose::x::single_aux::decompose(
                control,
                clean_axillary_range.0,
                target,
                AuxMode::Clean,
                DepthMode::Linear,
                approximated,
            )
        } else {
            decompose::u2::linear_depth(QuantumGate::PauliX, control, target)
        }
    }

    /// Decomposes a multi-controlled Pauli-Y gate.
    ///
    /// Uses the identity CY = (S†⊗I) · CX · (S⊗I).
    fn decompose_y(
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        chain![
            [(QuantumGate::sd(), target).into()],
            Self::decompose_x(target, control, approximated, clean_axillary_range),
            [(QuantumGate::s(), target).into()]
        ]
        .collect()
    }

    /// Decomposes a multi-controlled Pauli-Z gate.
    ///
    /// Uses the identity CZ = (H⊗I) · CX · (H⊗I).
    fn decompose_z(
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        chain![
            [(QuantumGate::Hadamard, target).into()],
            Self::decompose_x(target, control, approximated, clean_axillary_range),
            [(QuantumGate::Hadamard, target).into()]
        ]
        .collect()
    }

    /// Decomposes a multi-controlled Phase gate.
    fn decompose_phase(
        angle: f64,
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        let num_clean_aux = clean_axillary_range.1 - clean_axillary_range.0;
        let network_c3x_num_aux = Algorithm::NetworkU2(CXMode::C3X).aux_needed(control.len());

        if control.len() == 1 {
            decompose::u2::cu2(
                QuantumGate::Phase(Param::Value(angle)).matrix(),
                control[0],
                target,
            )
        } else if num_clean_aux >= network_c3x_num_aux {
            let aux_qubits: Vec<_> =
                (clean_axillary_range.0..clean_axillary_range.0 + network_c3x_num_aux).collect();

            decompose::network::network(
                QuantumGate::Phase(Param::Value(angle)),
                control,
                &aux_qubits,
                target,
                CXMode::C3X,
                approximated,
            )
        } else if num_clean_aux >= 1 {
            let control = control.iter().copied().chain([target]).collect_vec();
            decompose::su2::decompose(
                RotationZ(Param::Value(-2.0 * angle)),
                &control,
                clean_axillary_range.0,
                DepthMode::Linear,
                approximated,
            )
        } else {
            decompose::u2::linear_depth(QuantumGate::Phase(Param::Value(angle)), control, target)
        }
    }

    /// Decomposes a multi-controlled Hadamard gate.
    fn decompose_hadamard(
        target: usize,
        control: &[usize],
        approximated: bool,
        clean_axillary_range: (usize, usize),
    ) -> Vec<DecomposedGate> {
        let num_clean_aux = clean_axillary_range.1 - clean_axillary_range.0;
        let network_c3x_num_aux = Algorithm::NetworkU2(CXMode::C3X).aux_needed(control.len());

        if control.len() == 1 {
            decompose::u2::cu2(QuantumGate::Hadamard.matrix(), control[0], target)
        } else if num_clean_aux >= network_c3x_num_aux {
            let aux_qubits: Vec<_> =
                (clean_axillary_range.0..clean_axillary_range.0 + network_c3x_num_aux).collect();

            decompose::network::network(
                QuantumGate::Hadamard,
                control,
                &aux_qubits,
                target,
                CXMode::C3X,
                approximated,
            )
        } else if num_clean_aux >= 1 {
            decompose::u2::su2_rewrite_hadamard(control, clean_axillary_range.0, target)
        } else {
            decompose::u2::linear_depth(QuantumGate::Hadamard, control, target)
        }
    }
}
