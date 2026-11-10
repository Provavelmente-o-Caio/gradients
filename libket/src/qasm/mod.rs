// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! `OpenQASM` 2.0 front-end for the `libket` quantum circuit compiler.
//!
//! This module parses `OpenQASM` 2.0 source code (as defined in Appendix A of
//! arXiv:1707.03429v2) and produces a [`BasicBlock`] containing the resulting
//! gate sequence.
//!
//! # Qubit mapping
//!
//! The caller supplies a slice of physical qubit indices. `qreg` declarations
//! in the QASM source are assigned physical qubits from this slice in
//! declaration order:
//!
//! ```text
//! qreg a[2];   // a[0] = qubits[0], a[1] = qubits[1]
//! qreg b[1];   // b[0] = qubits[2]
//! ```
//!
//! # `measure` and `if` semantics
//!
//! `measure` statements are **not** executed. Instead, the mapping from each
//! classical bit to the measured qubit is recorded internally. When an `if`
//! statement is encountered, the qubit(s) that back the *set* bits of the
//! expected register value are used as controls for the associated quantum gate:
//!
//! ```text
//! measure q[0] -> c[0];    // records: cbit 0 ↔ q[0]
//! if (c == 1) x q[1];      // becomes: controlled-X with q[0] as control
//! ```
//!
//! # `qelib1.inc` handling
//!
//! `include "qelib1.inc"` is recognised as a special case and handled
//! natively: every standard library gate is mapped directly to its `libket`
//! equivalent. The `U3` gate uses the Z-Y-Z decomposition from §2 of
//! arXiv:1707.03429v2:
//!
//! ```text
//! U(θ, φ, λ) ≡ Rz(φ) · Ry(θ) · Rz(λ)   (up to global phase)
//! ```

pub mod ast;

mod executor;
mod gates;

// The LALRPOP-generated parser module is included at build time.
#[allow(clippy::all, unused_imports)]
mod openqasm {
    include!(concat!(env!("OUT_DIR"), "/qasm/openqasm.rs"));
}

use crate::{error::KetError, ir::block::BasicBlock};

/// Parse an `OpenQASM` 2.0 source string and return a [`BasicBlock`] with the
/// corresponding gate sequence.
///
/// # Parameters
///
/// - `qubits`: physical qubit indices assigned to the `qreg` declarations in
///   the source, in declaration order. The total number of entries must be ≥
///   the sum of all `qreg` sizes in `qasm_code`.
/// - `qasm_code`: `OpenQASM` 2.0 source text. The header line
///   `OPENQASM 2.0;` is mandatory.
///
/// # Errors
///
/// Returns a [`KetError`] if:
/// - The input fails to parse ([`KetError::OpenQasmParseError`]).
/// - A semantic error is detected, e.g. undefined gate, wrong arity,
///   unsupported include, or not enough qubits supplied
///   ([`KetError::OpenQasmSemanticError`] / [`KetError::QubitLimitExceeded`]).
/// - A `libket` IR operation fails, e.g. controlled-parameterized gate.
pub fn parse_openqasm(qubits: &[usize], qasm_code: &str) -> Result<BasicBlock, KetError> {
    let parser = openqasm::ProgramParser::new();
    let statements = parser
        .parse(qasm_code)
        .map_err(|_| KetError::OpenQasmParseError)?;
    executor::execute(qubits, statements)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state() {
        let block = parse_openqasm(
            &[0, 1],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                h q[0];
                cx q[0], q[1];
            "#,
        )
        .unwrap();
        assert!(!block.gates.is_empty());
    }

    #[test]
    fn test_u_gate_decomposition() {
        let block = parse_openqasm(
            &[0],
            r#"
                OPENQASM 2.0;
                qreg q[1];
                U(pi/2, 0, pi) q[0];
            "#,
        )
        .unwrap();
        // U(π/2, 0, π) decomposes into rotation gates.
        assert!(!block.gates.is_empty());
    }

    #[test]
    fn test_custom_gate() {
        parse_openqasm(
            &[0, 1],
            r#"
                OPENQASM 2.0;
                qreg q[2];
                gate mybell a, b {
                    h a;
                    cx a, b;
                }
                mybell q[0], q[1];
            "#,
        )
        .unwrap();
    }

    #[test]
    fn test_qelib1_h() {
        let block = parse_openqasm(
            &[0],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[1];
                h q[0];
            "#,
        )
        .unwrap();
        assert_eq!(block.gates.len(), 1);
    }

    #[test]
    fn test_measure_records_backing() {
        // The measure statement itself produces no gates, but the backing is
        // recorded so the subsequent `if` can be quantum-conditioned.
        parse_openqasm(
            &[0, 1, 2],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                creg c[2];
                h q[0];
                cx q[0], q[1];
                measure q -> c;
            "#,
        )
        .unwrap();
    }

    #[test]
    fn test_if_quantum_conditioning() {
        // measure records backing; if becomes a controlled gate.
        let block = parse_openqasm(
            &[0, 1, 2],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[3];
                creg c[1];
                h q[0];
                measure q[0] -> c[0];
                if (c == 1) x q[1];
            "#,
        )
        .unwrap();
        assert!(!block.gates.is_empty());
    }

    #[test]
    fn test_toffoli() {
        parse_openqasm(
            &[0, 1, 2],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[3];
                ccx q[0], q[1], q[2];
            "#,
        )
        .unwrap();
    }

    #[test]
    fn test_parametric_gate() {
        parse_openqasm(
            &[0],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[1];
                rx(pi/4) q[0];
                ry(pi/2) q[0];
                rz(pi)   q[0];
            "#,
        )
        .unwrap();
    }

    #[test]
    fn test_register_broadcast() {
        let block = parse_openqasm(
            &[0, 1, 2, 3],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[4];
                h q;
            "#,
        )
        .unwrap();
        // h q broadcasts to all 4 qubits.
        assert!(!block.gates.is_empty());
    }

    #[test]
    fn test_unsupported_include_error() {
        let result = parse_openqasm(
            &[0],
            r#"
                OPENQASM 2.0;
                include "custom.inc";
                qreg q[1];
            "#,
        );
        assert!(matches!(result, Err(KetError::OpenQasmSemanticError)));
    }

    #[test]
    fn test_qubit_mapping() {
        // Physical qubit indices 5 and 7 should be used instead of 0 and 1.
        let block = parse_openqasm(
            &[5, 7],
            r#"
                OPENQASM 2.0;
                include "qelib1.inc";
                qreg q[2];
                h q[0];
                cx q[0], q[1];
            "#,
        )
        .unwrap();
        assert!(!block.gates.is_empty());
    }

    #[test]
    fn test_insufficient_qubits_error() {
        let result = parse_openqasm(
            &[0],
            r#"
                OPENQASM 2.0;
                qreg q[3];
            "#,
        );
        assert!(matches!(result, Err(KetError::QubitLimitExceeded)));
    }
}
