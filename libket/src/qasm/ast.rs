// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

// Abstract Syntax Tree types for `OpenQASM` 2.0.
//
// These types directly mirror the `OpenQASM` 2.0 grammar from Appendix A of
// arXiv:1707.03429v2. The executor module consumes this AST to drive a
// `libket` [`Process`](ket::process::Process).

use std::fmt;

// ---------------------------------------------------------------------------
// Qubit / classical-bit argument
// ---------------------------------------------------------------------------

/// A reference to a quantum or classical register, optionally with an index.
///
/// `id`    → the full register\
/// `id[n]` → single-element slice at position `n`
#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    /// Unindexed register reference (`id`).
    Id(String),
    /// Indexed register reference (`id[n]`).
    Indexed(String, usize),
}

impl fmt::Display for Argument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Argument::Id(id) => write!(f, "{id}"),
            Argument::Indexed(id, n) => write!(f, "{id}[{n}]"),
        }
    }
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// Binary arithmetic operators in parameter expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary function names allowed in parameter expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryFn {
    Sin,
    Cos,
    Tan,
    Exp,
    Ln,
    Sqrt,
}

/// A parameter expression (angle / rotation amount).
///
/// Appears inside `(…)` in gate calls and gate definitions.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Real(f64),
    Pi,
    Id(String),
    Neg(Box<Expr>),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    UnaryFn(UnaryFn, Box<Expr>),
}

impl Expr {
    /// Evaluate the expression in the context of a variable binding.
    ///
    /// `env` maps parameter names to their concrete `f64` values.
    /// Returns `None` if an unknown identifier is encountered.
    #[must_use]
    pub fn eval(&self, env: &std::collections::HashMap<String, f64>) -> Option<f64> {
        use std::f64::consts::PI;
        match self {
            Expr::Real(v) => Some(*v),
            Expr::Pi => Some(PI),
            Expr::Id(name) => env.get(name).copied(),
            Expr::Neg(e) => Some(-e.eval(env)?),
            Expr::BinOp(l, op, r) => {
                let lv = l.eval(env)?;
                let rv = r.eval(env)?;
                Some(match op {
                    BinOp::Add => lv + rv,
                    BinOp::Sub => lv - rv,
                    BinOp::Mul => lv * rv,
                    BinOp::Div => lv / rv,
                    BinOp::Pow => lv.powf(rv),
                })
            }
            Expr::UnaryFn(f, e) => {
                let v = e.eval(env)?;
                Some(match f {
                    UnaryFn::Sin => v.sin(),
                    UnaryFn::Cos => v.cos(),
                    UnaryFn::Tan => v.tan(),
                    UnaryFn::Exp => v.exp(),
                    UnaryFn::Ln => v.ln(),
                    UnaryFn::Sqrt => v.sqrt(),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Gate call
// ---------------------------------------------------------------------------

/// A gate application, either at top level or inside a gate body.
#[derive(Debug, Clone, PartialEq)]
pub struct GateCall {
    /// Gate name (e.g. `"U"`, `"CX"`, `"h"`, `"cx"`, …).
    pub name: String,
    /// Parameter expressions (rotation angles).
    pub params: Vec<Expr>,
    /// Qubit arguments.
    pub qargs: Vec<Argument>,
}

// ---------------------------------------------------------------------------
// Gate body operations
// ---------------------------------------------------------------------------

/// An operation that can appear inside a gate `{}` body.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOp {
    Gate(GateCall),
    Barrier(Vec<String>),
}

// ---------------------------------------------------------------------------
// Quantum operations (top-level)
// ---------------------------------------------------------------------------

/// A quantum operation that can appear at the top level or after `if`.
#[derive(Debug, Clone, PartialEq)]
pub enum QOp {
    /// A gate application.
    Gate(GateCall),
    /// A measurement: `measure qarg -> carg;`
    Measure(Argument, Argument),
    /// A reset: `reset qarg;`
    Reset(Argument),
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// A top-level `OpenQASM` statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    /// `qreg id[n];` – allocate `n` quantum bits.
    QReg(String, usize),
    /// `creg id[n];` – declare `n` classical bits.
    CReg(String, usize),
    /// `include "filename";`
    Include(String),
    /// `gate id(params) qargs { body }`
    GateDef(String, Vec<String>, Vec<String>, Vec<GateOp>),
    /// `opaque id(params) qargs;`
    OpaqueGate(String, Vec<String>, Vec<String>),
    /// A top-level quantum operation.
    QOp(QOp),
    /// `if (id == n) qop`
    If(String, u64, QOp),
    /// `barrier args;`
    Barrier(Vec<Argument>),
}
