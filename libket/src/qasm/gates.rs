// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

// Native `qelib1.inc` gate mappings and U3 decomposition.
//
// All standard `qelib1.inc` gates are mapped directly to their `libket`
// primitives rather than being expanded through the `qelib1.inc` file.
// This avoids unnecessary multi-level expansion and gives the compiler the
// most accurate representation.
//
// The U3 gate is decomposed using the Z-Y-Z decomposition from Section 2 of
// arXiv:1707.03429v2:
//
// ```text
//   U(θ, φ, λ) = Rz(φ) · Ry(θ) · Rz(λ)   (up to global phase)
// ```
//
// Where libket's `RotationY` and `RotationZ` gates use the convention
// `R_k(α) = exp(-i α/2 · σ_k)`, so the half-angle is applied internally.

use std::f64::consts::PI;

use crate::controlled_gate;
use crate::ir::{block::BasicBlock, gate::QuantumGate};

use crate::error::KetError;

// ---------------------------------------------------------------------------
// U3 / U gate
// ---------------------------------------------------------------------------

/// Append `U(θ, φ, λ)` to `block` on `target`, using the Z-Y-Z decomposition.
///
/// The `OpenQASM` 2.0 spec (Eq. 1 in arXiv:1707.03429v2) defines:
/// ```text
///   U(θ, φ, λ) = Rz(φ + π) · Ry(θ) · Rz(λ + π)  · (global phase)
/// ```
/// In libket conventions, `RotationZ(α)` implements `exp(-i α/2 · Z)`.
/// Matching to the spec's `u3` matrix we get:
/// ```text
///   U(θ, φ, λ) ≡ Rz(φ) · Ry(θ) · Rz(λ)
/// ```
/// up to a global phase of `exp(-i(φ+λ)/2)`, which is unobservable for a
/// single qubit and handled via `BasicBlock::add_global_phase`.
pub fn append_u3(block: &mut BasicBlock, theta: f64, phi: f64, lambda: f64, target: usize) {
    // Z-Y-Z decomposition: U(θ,φ,λ) = Rz(φ) · Ry(θ) · Rz(λ)
    // libket RotationZ(α) = exp(-i α/2 Z), matches the OpenQASM convention.
    use crate::ir::gate::Param;
    block.append_gate(QuantumGate::RotationZ(Param::Value(lambda)), target);
    block.append_gate(QuantumGate::RotationY(Param::Value(theta)), target);
    block.append_gate(QuantumGate::RotationZ(Param::Value(phi)), target);
    // Global phase exp(-i(φ+λ)/2): record it on the block.
    block.add_global_phase(-(phi + lambda) / 2.0);
}

/// Append a controlled `U(θ, φ, λ)` to `block` (controlled-U3).
///
/// Decomposes as a controlled Z-Y-Z, using `BasicBlock::control` to wrap
/// each single-qubit rotation.
pub fn append_controlled_u3(
    block: &mut BasicBlock,
    theta: f64,
    phi: f64,
    lambda: f64,
    controls: &[usize],
    target: usize,
) -> Result<(), KetError> {
    let mut u3_block = BasicBlock::new();
    append_u3(&mut u3_block, theta, phi, lambda, target);
    let controlled = u3_block.control(controls)?;
    block.append_block(controlled);
    Ok(())
}

// ---------------------------------------------------------------------------
// qelib1 gate dispatch
// ---------------------------------------------------------------------------

/// Represents a built-in `qelib1.inc` gate after parameter evaluation.
///
/// Each variant carries the concrete (evaluated) parameter values and the
/// resolved qubit indices so the caller can directly generate `BasicBlock`
/// instructions.
pub enum Qelib1Gate {
    /// U3 gate – full decomposition required.
    U3 {
        theta: f64,
        phi: f64,
        lambda: f64,
        target: usize,
    },
    /// U2 gate – U(π/2, φ, λ).
    U2 {
        phi: f64,
        lambda: f64,
        target: usize,
    },
    /// U1 gate – phase gate P(λ) = U(0, 0, λ).
    U1 { lambda: f64, target: usize },
    /// Identity gate – no-op.
    #[allow(dead_code)]
    Id { target: usize },
    /// Hadamard gate.
    H { target: usize },
    /// Pauli X gate.
    X { target: usize },
    /// Pauli Y gate.
    Y { target: usize },
    /// Pauli Z gate.
    Z { target: usize },
    /// S gate (P(π/2)).
    S { target: usize },
    /// S† gate (P(-π/2)).
    Sdg { target: usize },
    /// T gate (P(π/4)).
    T { target: usize },
    /// T† gate (P(-π/4)).
    Tdg { target: usize },
    /// √X gate = Rx(π/2).
    Sx { target: usize },
    /// √X† gate = Rx(-π/2).
    Sxdg { target: usize },
    /// Rx(θ) rotation.
    Rx { theta: f64, target: usize },
    /// Ry(θ) rotation.
    Ry { theta: f64, target: usize },
    /// Rz(λ) rotation.
    Rz { phi: f64, target: usize },
    /// CNOT gate.
    Cx { control: usize, target: usize },
    /// Controlled-Y gate.
    Cy { control: usize, target: usize },
    /// Controlled-Z gate.
    Cz { control: usize, target: usize },
    /// Controlled-H gate.
    Ch { control: usize, target: usize },
    /// Controlled-S gate.
    Cs { control: usize, target: usize },
    /// Controlled-S† gate.
    Csdg { control: usize, target: usize },
    /// Controlled-T gate.
    Ct { control: usize, target: usize },
    /// Controlled-T† gate.
    Ctdg { control: usize, target: usize },
    /// Swap gate (3 CNOTs).
    Swap { a: usize, b: usize },
    /// Controlled-Rx gate.
    Crx {
        theta: f64,
        control: usize,
        target: usize,
    },
    /// Controlled-Ry gate.
    Cry {
        theta: f64,
        control: usize,
        target: usize,
    },
    /// Controlled-Rz gate.
    Crz {
        phi: f64,
        control: usize,
        target: usize,
    },
    /// Controlled-U1 (controlled phase).
    Cu1 {
        lambda: f64,
        control: usize,
        target: usize,
    },
    /// Controlled-U3.
    Cu3 {
        theta: f64,
        phi: f64,
        lambda: f64,
        control: usize,
        target: usize,
    },
    /// Doubly-controlled-X (Toffoli).
    Ccx { c0: usize, c1: usize, target: usize },
    /// Controlled-Swap (Fredkin).
    Cswap { control: usize, a: usize, b: usize },
    /// Rxx(θ) – Ising XX coupling.
    Rxx { theta: f64, a: usize, b: usize },
    /// Rzz(θ) – Ising ZZ coupling.
    Rzz { theta: f64, a: usize, b: usize },
}

/// Append the gate described by `gate` to `block`.
pub fn append_qelib1_gate(block: &mut BasicBlock, gate: Qelib1Gate) -> Result<(), KetError> {
    use crate::ir::gate::Param;
    use crate::swap_gate;

    match gate {
        // ── Single-qubit parameterized ────────────────────────────────────
        Qelib1Gate::U3 {
            theta,
            phi,
            lambda,
            target,
        } => {
            append_u3(block, theta, phi, lambda, target);
        }
        Qelib1Gate::U2 {
            phi,
            lambda,
            target,
        } => {
            // u2(φ,λ) = U(π/2, φ, λ)
            append_u3(block, PI / 2.0, phi, lambda, target);
        }
        Qelib1Gate::U1 { lambda, target } => {
            // u1(λ) = phase gate P(λ)
            block.append_gate(QuantumGate::Phase(Param::Value(lambda)), target);
        }
        Qelib1Gate::Rx { theta, target } => {
            block.append_gate(QuantumGate::RotationX(Param::Value(theta)), target);
        }
        Qelib1Gate::Ry { theta, target } => {
            block.append_gate(QuantumGate::RotationY(Param::Value(theta)), target);
        }
        Qelib1Gate::Rz { phi, target } => {
            block.append_gate(QuantumGate::RotationZ(Param::Value(phi)), target);
        }

        // ── Single-qubit fixed ────────────────────────────────────────────
        Qelib1Gate::Id { .. } => { /* identity – no-op */ }
        Qelib1Gate::H { target } => {
            block.append_gate(QuantumGate::Hadamard, target);
        }
        Qelib1Gate::X { target } => {
            block.append_gate(QuantumGate::PauliX, target);
        }
        Qelib1Gate::Y { target } => {
            block.append_gate(QuantumGate::PauliY, target);
        }
        Qelib1Gate::Z { target } => {
            block.append_gate(QuantumGate::PauliZ, target);
        }
        Qelib1Gate::S { target } => {
            block.append_gate(QuantumGate::s(), target);
        }
        Qelib1Gate::Sdg { target } => {
            block.append_gate(QuantumGate::sd(), target);
        }
        Qelib1Gate::T { target } => {
            block.append_gate(QuantumGate::t(), target);
        }
        Qelib1Gate::Tdg { target } => {
            block.append_gate(QuantumGate::td(), target);
        }
        Qelib1Gate::Sx { target } => {
            // √X = Rx(π/2)
            block.append_gate(QuantumGate::RotationX(Param::Value(PI / 2.0)), target);
        }
        Qelib1Gate::Sxdg { target } => {
            // (√X)† = Rx(-π/2)
            block.append_gate(QuantumGate::RotationX(Param::Value(-PI / 2.0)), target);
        }

        // ── Two-qubit ─────────────────────────────────────────────────────
        Qelib1Gate::Cx { control, target } => {
            block.append_block(controlled_gate(QuantumGate::PauliX, &[control], target)?);
        }
        Qelib1Gate::Cy { control, target } => {
            block.append_block(controlled_gate(QuantumGate::PauliY, &[control], target)?);
        }
        Qelib1Gate::Cz { control, target } => {
            block.append_block(controlled_gate(QuantumGate::PauliZ, &[control], target)?);
        }
        Qelib1Gate::Ch { control, target } => {
            block.append_block(controlled_gate(QuantumGate::Hadamard, &[control], target)?);
        }
        Qelib1Gate::Cs { control, target } => {
            block.append_block(controlled_gate(QuantumGate::s(), &[control], target)?);
        }
        Qelib1Gate::Csdg { control, target } => {
            block.append_block(controlled_gate(QuantumGate::sd(), &[control], target)?);
        }
        Qelib1Gate::Ct { control, target } => {
            block.append_block(controlled_gate(QuantumGate::t(), &[control], target)?);
        }
        Qelib1Gate::Ctdg { control, target } => {
            block.append_block(controlled_gate(QuantumGate::td(), &[control], target)?);
        }
        Qelib1Gate::Swap { a, b } => {
            block.append_block(swap_gate(a, b)?);
        }
        Qelib1Gate::Crx {
            theta,
            control,
            target,
        } => {
            block.append_block(controlled_gate(
                QuantumGate::RotationX(Param::Value(theta)),
                &[control],
                target,
            )?);
        }
        Qelib1Gate::Cry {
            theta,
            control,
            target,
        } => {
            block.append_block(controlled_gate(
                QuantumGate::RotationY(Param::Value(theta)),
                &[control],
                target,
            )?);
        }
        Qelib1Gate::Crz {
            phi,
            control,
            target,
        } => {
            block.append_block(controlled_gate(
                QuantumGate::RotationZ(Param::Value(phi)),
                &[control],
                target,
            )?);
        }
        Qelib1Gate::Cu1 {
            lambda,
            control,
            target,
        } => {
            block.append_block(controlled_gate(
                QuantumGate::Phase(Param::Value(lambda)),
                &[control],
                target,
            )?);
        }
        Qelib1Gate::Cu3 {
            theta,
            phi,
            lambda,
            control,
            target,
        } => {
            append_controlled_u3(block, theta, phi, lambda, &[control], target)?;
        }

        // ── Three-qubit ───────────────────────────────────────────────────
        Qelib1Gate::Ccx { c0, c1, target } => {
            // Toffoli = CCX
            block.append_block(controlled_gate(QuantumGate::PauliX, &[c0, c1], target)?);
        }
        Qelib1Gate::Cswap { control, a, b } => {
            // Fredkin = controlled SWAP (3 CNOTs controlled)
            let mut cswap_block = BasicBlock::new();
            cswap_block.append_block(swap_gate(a, b)?);
            let controlled = cswap_block.control(&[control])?;
            block.append_block(controlled);
        }

        // ── Two-qubit parametric (Ising-type) ────────────────────────────
        Qelib1Gate::Rxx { theta, a, b } => {
            // Rxx(θ) = exp(-i θ/2 · X⊗X)
            // Decomposition: H⊗H → CX(a,b) → Rz(θ) on b → CX(a,b) → H⊗H
            block.append_gate(QuantumGate::Hadamard, a);
            block.append_gate(QuantumGate::Hadamard, b);
            block.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
            block.append_gate(QuantumGate::RotationZ(Param::Value(theta)), b);
            block.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
            block.append_gate(QuantumGate::Hadamard, a);
            block.append_gate(QuantumGate::Hadamard, b);
        }
        Qelib1Gate::Rzz { theta, a, b } => {
            // Rzz(θ) = exp(-i θ/2 · Z⊗Z)
            // Decomposition: CX(a,b) → Rz(θ) on b → CX(a,b)
            block.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
            block.append_gate(QuantumGate::RotationZ(Param::Value(theta)), b);
            block.append_block(controlled_gate(QuantumGate::PauliX, &[a], b)?);
        }
    }

    Ok(())
}
