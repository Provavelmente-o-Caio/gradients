// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Integration and unit tests for libket.
//!
//! Each `mod` targets a distinct module so test failures are easy to locate.

#[cfg(test)]
mod param_tests {
    use crate::ir::gate::Param;
    use std::f64::consts::PI;

    // ── Value variant ────────────────────────────────────────────────────────

    #[test]
    fn value_inverse_negates() {
        let p = Param::Value(PI);
        assert_eq!(p.inverse(), Param::Value(-PI));
    }

    #[test]
    fn value_inverse_of_zero_is_zero() {
        let p = Param::Value(0.0);
        let inv = p.inverse();
        assert!(inv.value().abs() < 1e-15);
    }

    #[test]
    fn value_is_near_zero_true() {
        assert!(Param::Value(1e-11).is_near_zero(1e-10));
    }

    #[test]
    fn value_is_near_zero_false() {
        assert!(!Param::Value(0.1).is_near_zero(1e-10));
    }

    #[test]
    fn value_add_sums_correctly() {
        let a = Param::Value(1.0);
        let b = Param::Value(2.0);
        assert_eq!(a.add(&b), Some(Param::Value(3.0)));
    }

    #[test]
    fn value_add_with_negative() {
        let a = Param::Value(PI);
        let b = Param::Value(-PI);
        let sum = a.add(&b).unwrap();
        assert!(sum.value().abs() < 1e-15);
    }

    #[test]
    fn value_set_parameter_is_noop() {
        let p = Param::Value(1.5);
        assert_eq!(p.set_parameter(&[99.0]), Param::Value(1.5));
    }

    // ── Ref variant ─────────────────────────────────────────────────────────

    fn ref_param(index: usize, multiplier: f64, value: f64) -> Param {
        Param::Ref {
            index,
            multiplier,
            value,
        }
    }

    #[test]
    fn ref_value_returns_product() {
        // Ref { multiplier: 2.0, value: 3.0 } => effective = 6.0
        let p = ref_param(0, 2.0, 3.0);
        assert!((p.value() - 6.0).abs() < 1e-15);
    }

    #[test]
    fn ref_inverse_negates_multiplier() {
        let p = ref_param(0, 2.0, 3.0);
        let inv = p.inverse();
        // inverse effective value = -6.0
        assert!((inv.value() - (-6.0)).abs() < 1e-15);
        // value field should be unchanged
        if let Param::Ref { value, .. } = inv {
            assert!((value - 3.0).abs() < 1e-15);
        } else {
            panic!("Expected Ref");
        }
    }

    #[test]
    fn ref_is_near_zero_when_multiplier_near_zero() {
        let p = ref_param(0, 1e-11, 100.0);
        assert!(p.is_near_zero(1e-10));
    }

    #[test]
    fn ref_is_near_zero_false_when_multiplier_nonzero() {
        let p = ref_param(0, 1.0, 0.0);
        assert!(!p.is_near_zero(1e-10));
    }

    #[test]
    fn ref_set_parameter_binds_value() {
        // Ref { index:0, multiplier:2.0, value:3.0 }
        // After set_parameter([5.0]): Value(2.0 * 5.0) = Value(10.0)
        let p = ref_param(0, 2.0, 3.0);
        let bound = p.set_parameter(&[5.0]);
        assert!((bound.value() - 10.0).abs() < 1e-15);
    }

    /// `Param::add` on any `Ref` variant (same index, different index, or
    /// mixed with `Value`) must always return `None` — symbolic parameters
    /// cannot be merged algebraically.
    #[test]
    fn ref_add_same_index_returns_none() {
        let p1 = ref_param(0, 2.0, 3.0);
        let p2 = ref_param(0, 1.0, 3.0);
        assert_eq!(p1.add(&p2), None, "Ref + Ref (same index) must return None");
    }

    #[test]
    fn ref_add_same_index_different_values_returns_none() {
        let p1 = ref_param(0, 1.0, 2.0);
        let p2 = ref_param(0, 1.0, 4.0);
        assert_eq!(
            p1.add(&p2),
            None,
            "Ref + Ref (same index, different values) must return None"
        );
    }

    #[test]
    fn ref_add_different_index_returns_none() {
        let p1 = ref_param(0, 1.0, 1.0);
        let p2 = ref_param(1, 1.0, 1.0);
        assert_eq!(
            p1.add(&p2),
            None,
            "Ref + Ref (different index) must return None"
        );
    }

    #[test]
    fn ref_add_value_ref_returns_none() {
        let p1 = Param::Value(1.0);
        let p2 = ref_param(0, 1.0, 1.0);
        assert_eq!(p1.add(&p2), None, "Value + Ref must return None");
        assert_eq!(p2.add(&p1), None, "Ref + Value must return None");
    }
}

#[cfg(test)]
mod gate_propriety_tests {
    use crate::ir::gate::GatePropriety;

    // ── restrict (upper-bound / escalate) ────────────────────────────────────

    #[test]
    fn restrict_identity_with_anything_gives_other() {
        for other in [
            GatePropriety::Identity,
            GatePropriety::Diagonal,
            GatePropriety::Permutation,
            GatePropriety::Unitary,
        ] {
            let mut p = GatePropriety::Identity;
            p.restrict(other);
            assert_eq!(p, other);

            let mut p2 = other;
            p2.restrict(GatePropriety::Identity);
            assert_eq!(p2, other);
        }
    }

    #[test]
    fn restrict_diagonal_diagonal_stays_diagonal() {
        let mut p = GatePropriety::Diagonal;
        p.restrict(GatePropriety::Diagonal);
        assert_eq!(p, GatePropriety::Diagonal);
    }

    #[test]
    fn restrict_anything_with_unitary_gives_unitary() {
        for base in [
            GatePropriety::Identity,
            GatePropriety::Diagonal,
            GatePropriety::Permutation,
        ] {
            let mut p = base;
            p.restrict(GatePropriety::Unitary);
            assert_eq!(p, GatePropriety::Unitary, "base = {base:?}");

            let mut p2 = GatePropriety::Unitary;
            p2.restrict(base);
            assert_eq!(p2, GatePropriety::Unitary, "restrict(Unitary, {base:?})");
        }
    }

    #[test]
    fn restrict_diagonal_permutation_gives_permutation() {
        let mut p = GatePropriety::Diagonal;
        p.restrict(GatePropriety::Permutation);
        assert_eq!(p, GatePropriety::Permutation);

        let mut p2 = GatePropriety::Permutation;
        p2.restrict(GatePropriety::Diagonal);
        assert_eq!(p2, GatePropriety::Permutation);
    }

    // ── broaden (lower-bound / downgrade) ────────────────────────────────────
    //
    // `broaden` is used by `set_as_diagonal` / `set_as_permutation` to
    // *downgrade* a qubit's propriety to a less-specific classification.
    // "broaden toward Identity" makes a qubit *less constrained* (identity is
    // the least specific gate action).

    #[test]
    fn broaden_any_with_identity_gives_identity() {
        for p in [
            GatePropriety::Diagonal,
            GatePropriety::Permutation,
            GatePropriety::Unitary,
        ] {
            let mut q = p;
            q.broaden(GatePropriety::Identity);
            assert_eq!(q, GatePropriety::Identity, "base={p:?}");

            let mut r = GatePropriety::Identity;
            r.broaden(p);
            assert_eq!(r, GatePropriety::Identity, "Identity.broaden({p:?})");
        }
    }

    #[test]
    fn broaden_diagonal_with_diagonal_stays_diagonal() {
        let mut p = GatePropriety::Diagonal;
        p.broaden(GatePropriety::Diagonal);
        assert_eq!(p, GatePropriety::Diagonal);
    }

    #[test]
    fn broaden_unitary_with_diagonal_gives_diagonal() {
        let mut p = GatePropriety::Unitary;
        p.broaden(GatePropriety::Diagonal);
        assert_eq!(p, GatePropriety::Diagonal);
    }

    #[test]
    fn broaden_permutation_with_diagonal_gives_diagonal() {
        let mut p = GatePropriety::Permutation;
        p.broaden(GatePropriety::Diagonal);
        assert_eq!(p, GatePropriety::Diagonal);
    }

    #[test]
    fn broaden_unitary_with_permutation_gives_permutation() {
        let mut p = GatePropriety::Unitary;
        p.broaden(GatePropriety::Permutation);
        assert_eq!(p, GatePropriety::Permutation);
    }

    /// broaden with Unitary should NOT change the propriety:
    /// Unitary is the most specific / most constrained class; broadening
    /// toward Unitary is a no-op (the gate stays as general as it already is).
    #[test]
    fn broaden_with_unitary_is_noop() {
        for base in [
            GatePropriety::Diagonal,
            GatePropriety::Permutation,
            GatePropriety::Unitary,
        ] {
            let mut p = base;
            p.broaden(GatePropriety::Unitary);
            assert_eq!(p, base, "broaden({base:?}, Unitary) should be noop");
        }
    }
}

#[cfg(test)]
mod quantum_gate_tests {
    use crate::{
        decompose::util::matrix_dot,
        ir::gate::{Param, QuantumGate},
        matrix::Cf64,
    };
    use std::f64::consts::PI;

    fn mat_close(a: &[[Cf64; 2]; 2], b: &[[Cf64; 2]; 2], eps: f64) -> bool {
        (0..2).all(|i| {
            (0..2).all(|j| {
                (a[i][j].re - b[i][j].re).abs() < eps && (a[i][j].im - b[i][j].im).abs() < eps
            })
        })
    }

    #[test]
    fn pauli_x_is_self_inverse() {
        assert_eq!(QuantumGate::PauliX.inverse(), QuantumGate::PauliX);
    }

    #[test]
    fn hadamard_is_self_inverse() {
        assert_eq!(QuantumGate::Hadamard.inverse(), QuantumGate::Hadamard);
    }

    #[test]
    fn rotation_x_inverse_negates_angle() {
        let gate = QuantumGate::RotationX(Param::Value(PI / 4.0));
        let inv = gate.inverse();
        assert_eq!(inv, QuantumGate::RotationX(Param::Value(-PI / 4.0)));
    }

    #[test]
    fn inverse_composed_is_identity() {
        let gates = [
            QuantumGate::PauliX,
            QuantumGate::PauliY,
            QuantumGate::PauliZ,
            QuantumGate::Hadamard,
            QuantumGate::RotationX(Param::Value(PI / 3.0)),
            QuantumGate::RotationY(Param::Value(PI / 5.0)),
            QuantumGate::RotationZ(Param::Value(PI / 7.0)),
            QuantumGate::Phase(Param::Value(PI / 6.0)),
        ];
        let identity = [
            [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
            [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
        ];
        for gate in &gates {
            let m = gate.matrix();
            let inv = gate.inverse().matrix();
            let product = matrix_dot(&m, &inv);
            assert!(
                mat_close(&product, &identity, 1e-10),
                "{gate:?}: U * U† ≠ I"
            );
        }
    }

    #[test]
    fn rotation_x_pi_is_permutation() {
        assert!(QuantumGate::RotationX(Param::Value(PI)).is_permutation());
    }

    #[test]
    fn rotation_x_half_pi_is_not_permutation() {
        assert!(!QuantumGate::RotationX(Param::Value(PI / 2.0)).is_permutation());
    }

    #[test]
    fn rotation_y_pi_is_permutation() {
        assert!(QuantumGate::RotationY(Param::Value(PI)).is_permutation());
    }

    #[test]
    fn hadamard_is_not_permutation() {
        assert!(!QuantumGate::Hadamard.is_permutation());
    }

    #[test]
    fn pauli_z_is_diagonal() {
        assert!(QuantumGate::PauliZ.is_diagonal());
    }

    #[test]
    fn pauli_x_is_not_diagonal() {
        assert!(!QuantumGate::PauliX.is_diagonal());
    }

    #[test]
    fn merge_same_axis_adds_angles() {
        let a = QuantumGate::RotationZ(Param::Value(1.0));
        let b = QuantumGate::RotationZ(Param::Value(2.0));
        let merged = a.merge(&b).unwrap();
        if let QuantumGate::RotationZ(p) = merged {
            assert!(
                (p.value() - 3.0).abs() < 1e-12,
                "merged angle should be 3.0"
            );
        } else {
            panic!("merged gate should be RotationZ");
        }
    }

    #[test]
    fn merge_different_axes_returns_none() {
        let a = QuantumGate::RotationX(Param::Value(1.0));
        let b = QuantumGate::RotationZ(Param::Value(1.0));
        assert_eq!(a.merge(&b), None);
    }

    #[test]
    fn merge_pauli_gates_returns_none() {
        assert_eq!(QuantumGate::PauliX.merge(&QuantumGate::PauliX), None);
    }

    #[test]
    fn s_gate_is_phase_pi_over_2() {
        let s = QuantumGate::s();
        if let QuantumGate::Phase(p) = s {
            assert!((p.value() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        } else {
            panic!("s() should be Phase");
        }
    }

    #[test]
    fn t_and_td_are_inverses() {
        let t = QuantumGate::t();
        let td = QuantumGate::td();
        assert_eq!(t.inverse(), td);
        assert_eq!(td.inverse(), t);
    }
}

#[cfg(test)]
mod gate_instruction_tests {
    use crate::{
        error::KetError,
        ir::gate::{GateInstruction, Param, QuantumGate},
    };

    #[test]
    fn new_has_empty_control_set() {
        let inst = GateInstruction::new(QuantumGate::PauliX, 0);
        assert!(inst.control.is_empty());
        assert!(!inst.control_locked);
        assert!(!inst.is_approximated);
        assert!(inst.decomposed.is_none());
    }

    #[test]
    fn control_adds_qubits() {
        let inst = GateInstruction::new(QuantumGate::PauliX, 0);
        let controlled = inst.control(&[1, 2]).unwrap();
        assert!(controlled.control.contains(&1));
        assert!(controlled.control.contains(&2));
    }

    #[test]
    fn control_duplicate_in_new_set_is_error() {
        let inst = GateInstruction::new(QuantumGate::PauliX, 0);
        let err = inst.control(&[1, 1]).unwrap_err();
        assert_eq!(err, KetError::DuplicateControlQubit);
    }

    #[test]
    fn control_duplicate_with_existing_set_is_error() {
        let inst = GateInstruction::new(QuantumGate::PauliX, 0);
        let inst2 = inst.control(&[1]).unwrap();
        let err = inst2.control(&[1]).unwrap_err();
        assert_eq!(err, KetError::DuplicateControlQubit);
    }

    #[test]
    fn control_target_conflict_is_error() {
        let inst = GateInstruction::new(QuantumGate::PauliX, 0);
        let err = inst.control(&[0]).unwrap_err();
        assert_eq!(err, KetError::ControlTargetConflict);
    }

    #[test]
    fn control_locked_is_noop() {
        let mut inst = GateInstruction::new(QuantumGate::PauliX, 0);
        inst.lock_control();
        let result = inst.control(&[1, 2]).unwrap();
        assert!(
            result.control.is_empty(),
            "locked instruction should ignore control additions"
        );
    }

    #[test]
    fn inverse_preserves_control_and_reverses_gate() {
        let inst = GateInstruction::new(QuantumGate::RotationX(Param::Value(1.0)), 0);
        let controlled = inst.control(&[1]).unwrap();
        let inv = controlled.inverse();
        assert_eq!(inv.control, controlled.control);
        if let QuantumGate::RotationX(p) = inv.gate {
            assert!((p.value() - (-1.0)).abs() < 1e-12);
        } else {
            panic!("expected RotationX");
        }
    }

    #[test]
    fn commutes_with_both_diagonal() {
        let a = GateInstruction::new(QuantumGate::PauliZ, 0);
        let b = GateInstruction::new(QuantumGate::Phase(Param::Value(1.0)), 0);
        assert!(a.commutes_with(&b));
    }

    #[test]
    fn does_not_commute_non_diagonal() {
        let a = GateInstruction::new(QuantumGate::PauliZ, 0);
        let b = GateInstruction::new(QuantumGate::PauliX, 0);
        assert!(!a.commutes_with(&b));
    }
}

#[cfg(test)]
mod basic_block_tests {
    use crate::ir::{
        block::BasicBlock,
        gate::{Param, QuantumGate},
    };
    use std::f64::consts::PI;

    #[test]
    fn new_block_is_empty() {
        let block = BasicBlock::new();
        assert!(block.gates.is_empty());
        assert!(block.qubits_op.is_empty());
        assert!(block.global.is_none());
    }

    #[test]
    fn append_gate_then_inverse_cancels() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 0);
        block.append_gate(QuantumGate::PauliX, 0);
        assert_eq!(block.gates.len(), 0, "X followed by X should cancel");
    }

    #[test]
    fn append_rotation_then_inverse_cancels() {
        let mut block = BasicBlock::new();
        let angle = PI / 4.0;
        block.append_gate(QuantumGate::RotationZ(Param::Value(angle)), 0);
        block.append_gate(QuantumGate::RotationZ(Param::Value(-angle)), 0);
        assert_eq!(
            block.gates.len(),
            0,
            "Rz(θ) followed by Rz(-θ) should cancel"
        );
    }

    #[test]
    fn append_rotations_same_axis_merge() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::RotationZ(Param::Value(1.0)), 0);
        block.append_gate(QuantumGate::RotationZ(Param::Value(2.0)), 0);
        assert_eq!(block.gates.len(), 1, "two Rz gates should merge to one");
        if let QuantumGate::RotationZ(p) = block.gates[0].gate {
            assert!(
                (p.value() - 3.0).abs() < 1e-12,
                "merged angle should be 3.0"
            );
        } else {
            panic!("merged gate should be RotationZ");
        }
    }

    #[test]
    fn append_rotations_different_axes_dont_cancel() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::RotationX(Param::Value(1.0)), 0);
        block.append_gate(QuantumGate::RotationZ(Param::Value(1.0)), 0);
        assert_eq!(block.gates.len(), 2);
    }

    #[test]
    fn near_zero_rotation_is_dropped() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::RotationZ(Param::Value(1e-11)), 0);
        assert_eq!(block.gates.len(), 0, "near-zero rotation should be dropped");
    }

    #[test]
    fn max_qubit_index_empty_block() {
        let block = BasicBlock::new();
        assert_eq!(block.max_qubit_index(), None);
    }

    #[test]
    fn max_qubit_index_counts_target_and_control() {
        let block = crate::controlled_gate(QuantumGate::PauliX, &[5], 3).unwrap();
        assert_eq!(block.max_qubit_index(), Some(5));
    }

    #[test]
    fn inverse_reverses_gate_order() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 0);
        block.append_gate(QuantumGate::PauliZ, 0);
        let inv = block.inverse();
        assert_eq!(inv.gates.len(), 2);
        assert_eq!(inv.gates[0].gate, QuantumGate::PauliZ);
        assert_eq!(inv.gates[1].gate, QuantumGate::PauliX);
    }

    #[test]
    fn inverse_global_phase_negated() {
        let mut block = BasicBlock::new();
        block.add_global_phase(1.5);
        let inv = block.inverse();
        assert!((inv.global.unwrap() - (-1.5)).abs() < 1e-12);
    }

    #[test]
    fn append_block_merges_gates() {
        let mut a = BasicBlock::new();
        a.append_gate(QuantumGate::PauliX, 0);
        let mut b = BasicBlock::new();
        b.append_gate(QuantumGate::PauliX, 0);
        a.append_block(b);
        assert_eq!(
            a.gates.len(),
            0,
            "mutual inverses across blocks should cancel"
        );
    }

    #[test]
    fn control_block_adds_controls_to_all_gates() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 1);
        block.append_gate(QuantumGate::PauliX, 2);
        let controlled = block.control(&[0]).unwrap();
        for gate in &controlled.gates {
            assert!(
                gate.control.contains(&0),
                "gate should have control qubit 0"
            );
        }
    }

    #[test]
    fn control_block_target_conflict_propagates_error() {
        use crate::error::KetError;
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 0);
        let err = block.control(&[0]).unwrap_err();
        assert_eq!(err, KetError::ControlTargetConflict);
    }

    #[test]
    fn flat_gates_single_qubit_no_decomposition() {
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 0);
        let flat = block.flat_gates(None, None);
        assert_eq!(flat.len(), 1);
    }
}

#[cfg(test)]
mod process_tests {
    use crate::{
        error::KetError,
        process::{Process, QPUConfig},
    };

    fn unconfigured(num_qubits: usize) -> Process {
        Process::new(QPUConfig {
            num_qubits,
            quantum_execution: None,
        })
    }

    #[test]
    fn alloc_within_limit() {
        let mut p = unconfigured(3);
        assert_eq!(p.alloc(), Ok(0));
        assert_eq!(p.alloc(), Ok(1));
        assert_eq!(p.alloc(), Ok(2));
    }

    #[test]
    fn alloc_beyond_limit_errors() {
        let mut p = unconfigured(1);
        p.alloc().unwrap();
        assert_eq!(p.alloc(), Err(KetError::QubitLimitExceeded));
    }

    #[test]
    fn alloc_zero_qubits_errors_immediately() {
        let mut p = unconfigured(0);
        assert_eq!(p.alloc(), Err(KetError::QubitLimitExceeded));
    }

    #[test]
    fn append_block_qubit_out_of_range() {
        use crate::ir::{block::BasicBlock, gate::QuantumGate};
        let mut p = unconfigured(2);
        p.alloc().unwrap(); // qubit 0
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 2); // qubit 2 never allocated
        assert_eq!(
            p.append_block(block.into()),
            Err(KetError::QubitIndexOutOfRange)
        );
    }

    #[test]
    fn append_block_allocated_qubit_succeeds() {
        use crate::ir::{block::BasicBlock, gate::QuantumGate};
        let mut p = unconfigured(2);
        p.alloc().unwrap();
        let mut block = BasicBlock::new();
        block.append_gate(QuantumGate::PauliX, 0);
        assert!(p.append_block(block.into()).is_ok());
    }

    #[test]
    fn sample_without_backend_errors() {
        let mut p = unconfigured(2);
        p.alloc().unwrap();
        assert_eq!(p.sample(&[0], 100), Err(KetError::SamplingUnavailable));
    }

    #[test]
    fn exp_value_without_backend_errors() {
        use crate::ir::hamiltonian::Hamiltonian;
        let mut p = unconfigured(2);
        assert_eq!(
            p.exp_value(Hamiltonian::new()),
            Err(KetError::ExpectationValueUnavailable)
        );
    }

    #[test]
    fn measure_without_backend_errors() {
        let mut p = unconfigured(2);
        p.alloc().unwrap();
        assert_eq!(
            p.measure(&[0]),
            Err(KetError::MeasurementUnavailableInBatch)
        );
    }

    #[test]
    fn dump_without_backend_errors() {
        let mut p = unconfigured(2);
        p.alloc().unwrap();
        assert_eq!(p.dump(&[0]).unwrap_err(), KetError::DumpUnavailableInBatch);
    }

    #[test]
    fn execute_no_pending_measurement_errors() {
        let mut p = unconfigured(2);
        assert_eq!(p.execute(), Err(KetError::NoPendingMeasurement));
    }
}

#[cfg(test)]
mod lib_fn_tests {
    use crate::{controlled_gate, error::KetError, ir::gate::QuantumGate, swap_gate};

    #[test]
    fn controlled_gate_basic() {
        let block = controlled_gate(QuantumGate::PauliX, &[1], 0).unwrap();
        assert_eq!(block.gates.len(), 1);
        assert!(block.gates[0].control.contains(&1));
        assert_eq!(block.gates[0].target, 0);
    }

    #[test]
    fn controlled_gate_target_conflict() {
        let err = controlled_gate(QuantumGate::PauliX, &[0], 0).unwrap_err();
        assert_eq!(err, KetError::ControlTargetConflict);
    }

    #[test]
    fn controlled_gate_duplicate_controls() {
        let err = controlled_gate(QuantumGate::PauliX, &[1, 1], 0).unwrap_err();
        assert_eq!(err, KetError::DuplicateControlQubit);
    }

    #[test]
    fn swap_gate_three_cnots() {
        let block = swap_gate(0, 1).unwrap();
        assert_eq!(
            block.gates.len(),
            3,
            "SWAP should decompose to 3 controlled-X gates"
        );
        for gate in &block.gates {
            assert_eq!(gate.gate, QuantumGate::PauliX);
            assert_eq!(gate.control.len(), 1);
        }
    }
}

#[cfg(test)]
mod error_code_tests {
    use crate::error::KetError;

    fn all_variants() -> Vec<KetError> {
        vec![
            KetError::Success,
            KetError::DuplicateControlQubit,
            KetError::ControlTargetConflict,
            KetError::QubitLimitExceeded,
            KetError::QubitIndexOutOfRange,
            KetError::ProcessTerminated,
            KetError::GateAppendForbidden,
            KetError::MeasurementUnavailableInBatch,
            KetError::SamplingUnavailable,
            KetError::ExpectationValueUnavailable,
            KetError::DumpUnavailableInBatch,
            KetError::NoPendingMeasurement,
            KetError::ExplicitExecuteInLiveMode,
            KetError::ExecutionFailed,
            KetError::ShotCountInvalid,
            KetError::NativeGateUnsupported,
            KetError::ExpValueNotSupported,
            KetError::SerdeError,
            KetError::InvalidBias,
            KetError::UnknownError,
        ]
    }

    #[test]
    fn all_variants_unique_codes() {
        let variants = all_variants();
        let mut codes: Vec<i32> = variants.iter().map(|e| e.error_code()).collect();
        let original_len = codes.len();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), original_len, "duplicate error codes found");
    }

    #[test]
    fn success_code_is_zero() {
        assert_eq!(KetError::Success.error_code(), 0);
    }

    #[test]
    fn roundtrip_all_non_sentinel_variants() {
        // UnknownError is a sentinel (no unique code), skip it
        let variants = all_variants()
            .into_iter()
            .filter(|v| v != &KetError::UnknownError)
            .collect::<Vec<_>>();
        for v in &variants {
            let code = v.error_code();
            let back = KetError::from_error_code(code);
            assert_eq!(&back, v, "roundtrip failed for {v:?}");
        }
    }

    #[test]
    fn unknown_code_maps_to_unknown_error() {
        assert_eq!(KetError::from_error_code(9999), KetError::UnknownError);
        assert_eq!(KetError::from_error_code(-1), KetError::UnknownError);
    }
}

#[cfg(test)]
mod native_gate_set_tests {
    use crate::{
        decompose::util::matrix_dot, execution::NativeGateSet, ir::gate::QuantumGate, matrix::Cf64,
    };

    fn reconstruct_matrix(gates: &[(String, Vec<f64>, Vec<usize>)]) -> [[Cf64; 2]; 2] {
        let mut m = [
            [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
            [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
        ];
        for (name, angles, _) in gates {
            let gate_m = match name.as_str() {
                "rz" => QuantumGate::RotationZ(crate::ir::gate::Param::Value(angles[0])).matrix(),
                "ry" => QuantumGate::RotationY(crate::ir::gate::Param::Value(angles[0])).matrix(),
                other => panic!("unexpected gate: {other}"),
            };
            m = matrix_dot(&gate_m, &m);
        }
        m
    }

    fn magnitudes_close(a: &[[Cf64; 2]; 2], b: &[[Cf64; 2]; 2], eps: f64) -> bool {
        (0..2).all(|i| (0..2).all(|j| (a[i][j].norm() - b[i][j].norm()).abs() < eps))
    }

    #[test]
    fn translate_pauli_x() {
        let gate_set = ();
        let matrix = QuantumGate::PauliX.matrix();
        let native = gate_set.translate(&matrix, 0).unwrap();
        let reconstructed = reconstruct_matrix(&native);
        assert!(
            magnitudes_close(&reconstructed, &matrix, 1e-8),
            "PauliX translation mismatch"
        );
    }

    #[test]
    fn translate_hadamard() {
        let gate_set = ();
        let matrix = QuantumGate::Hadamard.matrix();
        let native = gate_set.translate(&matrix, 0).unwrap();
        let reconstructed = reconstruct_matrix(&native);
        assert!(
            magnitudes_close(&reconstructed, &matrix, 1e-8),
            "Hadamard translation mismatch"
        );
    }

    #[test]
    fn translate_pauli_z() {
        let gate_set = ();
        let matrix = QuantumGate::PauliZ.matrix();
        let native = gate_set.translate(&matrix, 0).unwrap();
        let reconstructed = reconstruct_matrix(&native);
        assert!(
            magnitudes_close(&reconstructed, &matrix, 1e-8),
            "PauliZ translation mismatch"
        );
    }

    #[test]
    fn translate_identity_emits_no_gates() {
        let gate_set = ();
        let identity = [
            [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
            [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
        ];
        let native = gate_set.translate(&identity, 0).unwrap();
        assert!(native.is_empty(), "identity should produce no native gates");
    }

    #[test]
    fn cnot_produces_single_gate() {
        let gate_set = ();
        let gates = gate_set.cnot(0, 1).unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].0, "cnot");
        assert_eq!(gates[0].2, vec![0, 1]);
    }
}

#[test]
fn test_flatten_diagonal() {
    use crate::ir::block::*;
    use crate::ir::gate::*;
    let mut b1 = BasicBlock::new();
    b1.append_gate(QuantumGate::PauliX, 0);
    b1.set_as_diagonal();

    let mut block = Block::new();
    block.append_basic_block(b1);
    let flattened = block.flatten().unwrap();
    println!("Flattened: {:?}", flattened.qubits_op);
    assert_eq!(
        flattened.qubits_op.get(&0).unwrap().propriety,
        GatePropriety::Diagonal
    );
}

#[test]
fn test_flatten_diagonal_with_merge() {
    use crate::ir::block::*;
    use crate::ir::gate::*;
    let mut b1 = BasicBlock::new();
    b1.append_gate(QuantumGate::RotationX(Param::Value(1.0)), 0);
    b1.set_as_diagonal();
    let mut b2 = BasicBlock::new();
    b2.append_gate(QuantumGate::RotationX(Param::Value(1.0)), 0); // Merges with b1!
    b2.set_as_diagonal();

    let mut block = Block::new();
    block.append_basic_block(b1);
    block.append_basic_block(b2);
    let flattened = block.flatten().unwrap();
    println!("Flattened: {:?}", flattened.qubits_op);
    assert_eq!(
        flattened.qubits_op.get(&0).unwrap().propriety,
        GatePropriety::Diagonal
    );
}

#[test]
fn test_flatten_diagonal_qft() {
    use crate::ir::block::*;
    let mut block = Block::new();
    block.append_template(
        TemplateBlock {
            name: "QFT".to_string(),
            qubits: vec![0, 1, 2],
            controls: vec![],
            control_locked: false,
            enable_approximated_decomposition: false,
            is_adjoint: false,
            is_self_inverse: false,
            is_diagonal: false,
            is_permutation: false,
        },
        0,
    );
    block.set_as_diagonal();

    let flattened = block.flatten().unwrap();
    println!("Flattened: {:?}", flattened.qubits_op);
    assert_eq!(
        flattened.qubits_op.get(&0).unwrap().propriety,
        crate::ir::gate::GatePropriety::Diagonal
    );
}

#[cfg(test)]
mod live_barrier_tests {
    //! Gates already sent to a live backend must never be cancelled or merged
    //! with gates appended after a mid-circuit measurement.

    use crate::error::KetError;
    use crate::execution::{DumpData, LiveExecution, NativeGate, QuantumExecution, SampleData};
    use crate::ir::block::{Block, TemplateBlock};
    use crate::ir::gate::{GateInstruction, Param, QuantumGate};
    use crate::ir::hamiltonian::Hamiltonian;
    use crate::process::{Process, QPUConfig};
    use std::sync::{Arc, Mutex};

    /// Live backend that records every gate it receives.
    struct Recorder(Arc<Mutex<Vec<GateInstruction>>>);

    impl LiveExecution for Recorder {
        fn compute_gate(&mut self, gate: &GateInstruction) -> Result<(), KetError> {
            self.0.lock().unwrap().push(gate.clone());
            Ok(())
        }
        fn compute_native_gates(&mut self, _: &[NativeGate]) -> Result<(), KetError> {
            unimplemented!()
        }
        fn measure(&mut self, _: &[usize]) -> Result<u64, KetError> {
            Ok(0)
        }
        fn dump(&mut self, _: &[usize]) -> Result<DumpData, KetError> {
            unimplemented!()
        }
        fn sample(&mut self, _: &[usize], _: usize) -> Result<SampleData, KetError> {
            unimplemented!()
        }
        fn exp_value(&mut self, _: Hamiltonian) -> Result<f64, KetError> {
            unimplemented!()
        }
    }

    fn live_process(num_qubits: usize) -> (Process, Arc<Mutex<Vec<GateInstruction>>>) {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let process = Process::new(QPUConfig {
            num_qubits,
            quantum_execution: Some(QuantumExecution::Live {
                qpu: Box::new(Recorder(sent.clone())),
                decompose: false,
                native_gate_set: None,
            }),
        });
        (process, sent)
    }

    fn gate_block(gate: QuantumGate, target: usize) -> Block {
        let mut block = Block::new();
        block.append_gate(gate, target);
        block
    }

    fn qft_block(qubits: Vec<usize>, is_adjoint: bool) -> Block {
        let mut block = Block::new();
        block.append_template(
            TemplateBlock {
                name: "QFT".to_string(),
                qubits,
                controls: vec![],
                control_locked: false,
                enable_approximated_decomposition: false,
                is_adjoint,
                is_self_inverse: false,
                is_diagonal: false,
                is_permutation: false,
            },
            0,
        );
        block
    }

    #[test]
    fn inverse_gates_do_not_cancel_across_measurement() {
        let (mut process, sent) = live_process(1);
        let q = process.alloc().unwrap();

        process
            .append_block(gate_block(QuantumGate::Hadamard, q))
            .unwrap();
        process.measure(&[q]).unwrap();
        process
            .append_block(gate_block(QuantumGate::Hadamard, q))
            .unwrap();
        process.measure(&[q]).unwrap();

        let sent = sent.lock().unwrap();
        assert_eq!(sent.len(), 2, "both H gates must reach the backend");
        assert!(sent.iter().all(|g| g.gate == QuantumGate::Hadamard));
        assert_eq!(process.flat_main_block().unwrap().gates.len(), 2);
    }

    #[test]
    fn rotations_do_not_merge_across_measurement() {
        let (mut process, sent) = live_process(1);
        let q = process.alloc().unwrap();

        process
            .append_block(gate_block(QuantumGate::RotationY(Param::Value(0.5)), q))
            .unwrap();
        process.measure(&[q]).unwrap();
        process
            .append_block(gate_block(QuantumGate::RotationY(Param::Value(-0.5)), q))
            .unwrap();
        process.measure(&[q]).unwrap();

        assert_eq!(sent.lock().unwrap().len(), 2);
    }

    #[test]
    fn template_does_not_cancel_across_measurement() {
        let (mut process, sent) = live_process(3);
        let q: Vec<usize> = (0..3).map(|_| process.alloc().unwrap()).collect();

        process.append_block(qft_block(q.clone(), false)).unwrap();
        process.measure(&q).unwrap();
        let after_qft = sent.lock().unwrap().len();
        assert!(after_qft > 0);

        process.append_block(qft_block(q.clone(), true)).unwrap();
        process.measure(&q).unwrap();
        let sent = sent.lock().unwrap();
        assert_eq!(
            sent.len(),
            2 * after_qft,
            "inverse QFT must be executed in full"
        );
        assert_eq!(process.flat_main_block().unwrap().gates.len(), sent.len());
    }

    #[test]
    fn gates_still_simplify_after_the_barrier() {
        let (mut process, sent) = live_process(1);
        let q = process.alloc().unwrap();

        process
            .append_block(gate_block(QuantumGate::Hadamard, q))
            .unwrap();
        process.measure(&[q]).unwrap();
        process
            .append_block(gate_block(QuantumGate::PauliX, q))
            .unwrap();
        process
            .append_block(gate_block(QuantumGate::PauliX, q))
            .unwrap();
        process.measure(&[q]).unwrap();

        assert_eq!(
            sent.lock().unwrap().len(),
            1,
            "X X after the barrier should cancel"
        );
    }
}
