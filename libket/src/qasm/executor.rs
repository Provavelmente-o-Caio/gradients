// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

// AST executor: walks an `OpenQASM` 2.0 AST and produces a [`BasicBlock`].
//
// ## Design
//
// The executor maps `qreg` declarations in the QASM source to the physical
// qubit indices supplied by the caller (in declaration order). It maintains a
// compile-time environment (`Env`) that tracks register names, gate definitions,
// and which qubit backs each classical bit so that `if` statements can be
// converted to controlled gates.
//
// ## `measure` and `if` semantics
//
// `measure` statements are never executed. Instead, the mapping from each
// classical bit to the qubit that was measured into it is recorded. When an
// `if` statement is encountered, the qubits that back the *set* bits of the
// expected register value are used as controls for the associated quantum gate.

use std::collections::{HashMap, VecDeque};

use crate::{
    controlled_gate,
    error::KetError,
    ir::{block::BasicBlock, gate::QuantumGate},
};

use super::{
    ast::{Argument, Expr, GateCall, GateOp, QOp, Statement},
    gates::{append_qelib1_gate, append_u3, Qelib1Gate},
};

// ---------------------------------------------------------------------------
// Gate definition
// ---------------------------------------------------------------------------

/// A user-defined gate as parsed from a `gate` statement.
#[derive(Debug, Clone)]
struct GateDef {
    /// Formal parameter names (rotation angles).
    params: Vec<String>,
    /// Formal qubit argument names.
    qargs: Vec<String>,
    /// The gate body operations.
    body: Vec<GateOp>,
}

// ---------------------------------------------------------------------------
// Compile-time environment
// ---------------------------------------------------------------------------

/// All state accumulated while walking the `OpenQASM` AST.
struct Env {
    /// Quantum register map: `name → Vec<physical_qubit_index>`.
    qregs: HashMap<String, Vec<usize>>,
    /// Classical register map: `name → (start_cbit_index, length)`.
    cregs: HashMap<String, (usize, usize)>,
    /// User-defined and opaque gate definitions.
    gate_defs: HashMap<String, GateDef>,
    /// Total number of declared classical bits.
    num_cbits: usize,
    /// Backing qubit for each classical bit (populated by `measure` statements).
    ///
    /// `creg_qubit_backing[cbit_index] = qubit_index` or `usize::MAX` if unknown.
    creg_qubit_backing: Vec<usize>,
    /// Iterator over the caller-supplied physical qubit indices.
    ///
    /// Each `qreg` declaration consumes the next `size` entries from this slice.
    qubit_supply: VecDeque<usize>,
}

impl Env {
    fn new(qubits: &[usize]) -> Self {
        Self {
            qregs: HashMap::new(),
            cregs: HashMap::new(),
            gate_defs: HashMap::new(),
            num_cbits: 0,
            creg_qubit_backing: Vec::new(),
            qubit_supply: qubits.iter().copied().collect(),
        }
    }

    /// Allocate the next `size` physical qubits from the caller-supplied slice.
    fn alloc_qubits(&mut self, size: usize) -> Result<Vec<usize>, KetError> {
        let mut qubits = Vec::with_capacity(size);
        for _ in 0..size {
            qubits.push(
                self.qubit_supply
                    .pop_front()
                    .ok_or(KetError::QubitLimitExceeded)?,
            );
        }
        Ok(qubits)
    }

    // ── Qubit resolution ─────────────────────────────────────────────────

    fn resolve_qarg(&self, arg: &Argument) -> Result<Vec<usize>, KetError> {
        match arg {
            Argument::Id(name) => {
                let qubits = self
                    .qregs
                    .get(name)
                    .ok_or(KetError::OpenQasmSemanticError)?;
                Ok(qubits.clone())
            }
            Argument::Indexed(name, idx) => {
                let qubits = self
                    .qregs
                    .get(name)
                    .ok_or(KetError::OpenQasmSemanticError)?;
                if *idx >= qubits.len() {
                    return Err(KetError::OpenQasmSemanticError);
                }
                Ok(vec![qubits[*idx]])
            }
        }
    }

    // ── Classical bit resolution ──────────────────────────────────────────

    fn resolve_carg(&self, arg: &Argument) -> Result<Vec<usize>, KetError> {
        match arg {
            Argument::Id(name) => {
                let &(start, len) = self
                    .cregs
                    .get(name)
                    .ok_or(KetError::OpenQasmSemanticError)?;
                Ok((start..start + len).collect())
            }
            Argument::Indexed(name, idx) => {
                let &(start, len) = self
                    .cregs
                    .get(name)
                    .ok_or(KetError::OpenQasmSemanticError)?;
                if *idx >= len {
                    return Err(KetError::OpenQasmSemanticError);
                }
                Ok(vec![start + idx])
            }
        }
    }

    // ── Backing qubit tracking ────────────────────────────────────────────

    /// Record which qubit backs each classical bit (from a `measure` statement).
    fn record_measure(&mut self, qubit_indices: &[usize], cbit_indices: &[usize]) {
        let max_cbit = cbit_indices.iter().copied().max().unwrap_or(0);
        if self.creg_qubit_backing.len() <= max_cbit {
            self.creg_qubit_backing.resize(max_cbit + 1, usize::MAX);
        }
        for (&cbit, &qubit) in cbit_indices.iter().zip(qubit_indices.iter()) {
            self.creg_qubit_backing[cbit] = qubit;
        }
    }

    /// Return the qubits that back the *set* bits of `expected` in `creg`.
    ///
    /// Used to build the control qubit list when an `if` statement is converted
    /// to controlled quantum gates.
    fn creg_backing_qubits(&self, creg: &str, expected: u64) -> Option<Vec<usize>> {
        let &(start, len) = self.cregs.get(creg)?;
        let mut controls = Vec::new();
        for i in 0..len {
            if (expected >> i) & 1 == 1 {
                let qubit = self.creg_qubit_backing.get(start + i).copied()?;
                if qubit == usize::MAX {
                    return None;
                }
                controls.push(qubit);
            }
        }
        Some(controls)
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Build a [`BasicBlock`] from a list of parsed QASM statements.
///
/// Physical qubit indices are consumed from `qubits` in `qreg` declaration
/// order. `measure` statements are silently skipped but their qubit-to-cbit
/// mapping is recorded so that subsequent `if` statements can be converted to
/// controlled gates.
pub(crate) fn execute(
    qubits: &[usize],
    statements: Vec<Statement>,
) -> Result<BasicBlock, KetError> {
    let mut env = Env::new(qubits);
    let mut block = BasicBlock::new();

    for stmt in statements {
        execute_statement(&mut block, &mut env, stmt)?;
    }
    Ok(block)
}

// ---------------------------------------------------------------------------
// Statement execution
// ---------------------------------------------------------------------------

fn execute_statement(
    block: &mut BasicBlock,
    env: &mut Env,
    stmt: Statement,
) -> Result<(), KetError> {
    match stmt {
        // ── Register declarations ─────────────────────────────────────────
        Statement::QReg(name, size) => {
            if env.qregs.contains_key(&name) {
                return Err(KetError::OpenQasmSemanticError);
            }
            let qubits = env.alloc_qubits(size)?;
            env.qregs.insert(name, qubits);
        }

        Statement::CReg(name, size) => {
            if env.cregs.contains_key(&name) {
                return Err(KetError::OpenQasmSemanticError);
            }
            let start = env.num_cbits;
            env.num_cbits += size;
            env.creg_qubit_backing.resize(env.num_cbits, usize::MAX);
            env.cregs.insert(name, (start, size));
        }

        // ── Include ───────────────────────────────────────────────────────
        Statement::Include(path) => {
            if path != "qelib1.inc" {
                return Err(KetError::OpenQasmSemanticError);
            }
            // qelib1.inc is handled natively; no action required.
        }

        // ── Gate / opaque declarations ────────────────────────────────────
        Statement::GateDef(name, params, qargs, body) => {
            env.gate_defs.insert(
                name,
                GateDef {
                    params,
                    qargs,
                    body,
                },
            );
        }

        Statement::OpaqueGate(name, params, qargs) => {
            // Record opaque gates with an empty body so that arity checks work;
            // invoking one will produce a semantic error at call time.
            env.gate_defs.insert(
                name,
                GateDef {
                    params,
                    qargs,
                    body: vec![],
                },
            );
        }

        // ── Quantum operations ────────────────────────────────────────────
        Statement::QOp(qop) => {
            execute_qop(block, env, &qop, &[])?;
        }

        // ── If statement → controlled gate ────────────────────────────────
        //
        // The qubit(s) that were measured into the classical register act as
        // controls for the gate. No classical branching occurs.
        Statement::If(creg_name, expected, qop) => {
            let controls = env
                .creg_backing_qubits(&creg_name, expected)
                .ok_or(KetError::OpenQasmSemanticError)?;
            execute_qop(block, env, &qop, &controls)?;
        }

        // ── Barrier ───────────────────────────────────────────────────────
        Statement::Barrier(_) => {
            // libket has no barrier concept; silently ignore.
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// QOp execution
// ---------------------------------------------------------------------------

/// Execute a quantum operation.
///
/// `extra_controls`: additional qubits prepended as controls to every gate
/// (used when implementing quantum conditioning of `if` statements).
fn execute_qop(
    block: &mut BasicBlock,
    env: &mut Env,
    qop: &QOp,
    extra_controls: &[usize],
) -> Result<(), KetError> {
    match qop {
        // ── Gate application ──────────────────────────────────────────────
        QOp::Gate(call) => {
            apply_gate_call(block, env, call, extra_controls, &HashMap::new())?;
        }

        // ── Measurement: record backing only, no execution ────────────────
        QOp::Measure(qarg, carg) => {
            let qubits = env.resolve_qarg(qarg)?;
            let cbits = env.resolve_carg(carg)?;
            if qubits.len() != cbits.len() {
                return Err(KetError::OpenQasmSemanticError);
            }
            env.record_measure(&qubits, &cbits);
        }

        // ── Reset: unsupported without live execution ─────────────────────
        QOp::Reset(_) => {
            // Reset requires knowledge of the qubit state and is not expressible
            // as a static gate sequence. Silently ignore.
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate call application
// ---------------------------------------------------------------------------

/// Append the gate described by `call` to `block`.
///
/// `extra_controls` are additional control qubits added to every instruction.
/// `param_env` provides values for formal parameter names (used during gate
/// body expansion).
fn apply_gate_call(
    block: &mut BasicBlock,
    env: &Env,
    call: &GateCall,
    extra_controls: &[usize],
    param_env: &HashMap<String, f64>,
) -> Result<(), KetError> {
    match call.name.as_str() {
        // ── Built-in U gate ───────────────────────────────────────────────
        "U" => {
            if call.params.len() != 3 {
                return Err(KetError::OpenQasmSemanticError);
            }
            if call.qargs.len() != 1 {
                return Err(KetError::OpenQasmSemanticError);
            }
            let vals = eval_params(&call.params, param_env)?;
            let (theta, phi, lambda) = (vals[0], vals[1], vals[2]);
            let qubits = env.resolve_qarg(&call.qargs[0])?;

            for &q in &qubits {
                if extra_controls.is_empty() {
                    append_u3(block, theta, phi, lambda, q);
                } else {
                    let mut inner = BasicBlock::new();
                    append_u3(&mut inner, theta, phi, lambda, q);
                    block.append_block(inner.control(extra_controls)?);
                }
            }
        }

        // ── Built-in CX gate ─────────────────────────────────────────────
        "CX" => {
            if call.qargs.len() != 2 {
                return Err(KetError::OpenQasmSemanticError);
            }
            let ctrl_list = env.resolve_qarg(&call.qargs[0])?;
            let tgt_list = env.resolve_qarg(&call.qargs[1])?;

            let n = broadcast_len(&ctrl_list, &tgt_list).ok_or(KetError::OpenQasmSemanticError)?;

            for i in 0..n {
                let c = if ctrl_list.len() == 1 {
                    ctrl_list[0]
                } else {
                    ctrl_list[i]
                };
                let t = if tgt_list.len() == 1 {
                    tgt_list[0]
                } else {
                    tgt_list[i]
                };
                let mut controls: Vec<usize> = extra_controls.to_vec();
                controls.push(c);
                block.append_block(controlled_gate(QuantumGate::PauliX, &controls, t)?);
            }
        }

        // ── qelib1.inc built-in gates ─────────────────────────────────────
        name if is_qelib1_gate(name) => {
            apply_qelib1_call(block, env, call, extra_controls, param_env)?;
        }

        // ── User-defined gate ─────────────────────────────────────────────
        name => {
            let def = env
                .gate_defs
                .get(name)
                .cloned()
                .ok_or(KetError::OpenQasmSemanticError)?;

            check_arity(
                def.params.len(),
                call.params.len(),
                def.qargs.len(),
                call.qargs.len(),
            )?;

            let vals = eval_params(&call.params, param_env)?;
            let new_param_env: HashMap<String, f64> =
                def.params.iter().cloned().zip(vals).collect();

            let resolved: Vec<Vec<usize>> = call
                .qargs
                .iter()
                .map(|a| env.resolve_qarg(a))
                .collect::<Result<_, _>>()?;

            let n = resolved.iter().map(std::vec::Vec::len).max().unwrap_or(1);
            for v in &resolved {
                if v.len() != 1 && v.len() != n {
                    return Err(KetError::OpenQasmSemanticError);
                }
            }

            for instance in 0..n {
                let qubit_map: HashMap<String, usize> = def
                    .qargs
                    .iter()
                    .cloned()
                    .zip(
                        resolved
                            .iter()
                            .map(|v| if v.len() == 1 { v[0] } else { v[instance] }),
                    )
                    .collect();

                expand_gate_body(
                    block,
                    env,
                    &def.body,
                    &new_param_env,
                    &qubit_map,
                    extra_controls,
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Gate body expansion
// ---------------------------------------------------------------------------

/// Expand a gate body, substituting formal qubit names with concrete indices.
fn expand_gate_body(
    block: &mut BasicBlock,
    env: &Env,
    body: &[GateOp],
    param_env: &HashMap<String, f64>,
    qubit_map: &HashMap<String, usize>,
    extra_controls: &[usize],
) -> Result<(), KetError> {
    for op in body {
        match op {
            GateOp::Gate(call) => {
                apply_gate_call_body(block, env, call, param_env, qubit_map, extra_controls)?;
            }
            GateOp::Barrier(_) => {}
        }
    }
    Ok(())
}

/// Apply a gate call that appears inside a gate body.
///
/// Qubit arguments are identifier-only (no indexing) and are resolved via
/// `qubit_map` (formal → concrete qubit index).
fn apply_gate_call_body(
    block: &mut BasicBlock,
    env: &Env,
    call: &GateCall,
    param_env: &HashMap<String, f64>,
    qubit_map: &HashMap<String, usize>,
    extra_controls: &[usize],
) -> Result<(), KetError> {
    fn resolve_body_qarg(
        arg: &Argument,
        qubit_map: &HashMap<String, usize>,
    ) -> Result<usize, KetError> {
        match arg {
            Argument::Id(name) => qubit_map
                .get(name)
                .copied()
                .ok_or(KetError::OpenQasmSemanticError),
            Argument::Indexed(_, _) => Err(KetError::OpenQasmSemanticError),
        }
    }

    let qubits: Vec<usize> = call
        .qargs
        .iter()
        .map(|a| resolve_body_qarg(a, qubit_map))
        .collect::<Result<_, _>>()?;

    match call.name.as_str() {
        "U" => {
            if call.params.len() != 3 || qubits.len() != 1 {
                return Err(KetError::OpenQasmSemanticError);
            }
            let vals = eval_params(&call.params, param_env)?;
            let (theta, phi, lambda) = (vals[0], vals[1], vals[2]);
            if extra_controls.is_empty() {
                append_u3(block, theta, phi, lambda, qubits[0]);
            } else {
                let mut inner = BasicBlock::new();
                append_u3(&mut inner, theta, phi, lambda, qubits[0]);
                block.append_block(inner.control(extra_controls)?);
            }
        }
        "CX" => {
            if qubits.len() != 2 {
                return Err(KetError::OpenQasmSemanticError);
            }
            let mut controls: Vec<usize> = extra_controls.to_vec();
            controls.push(qubits[0]);
            block.append_block(controlled_gate(QuantumGate::PauliX, &controls, qubits[1])?);
        }
        name if is_qelib1_gate(name) => {
            apply_qelib1_gate_raw(
                block,
                name,
                &call.params,
                param_env,
                &qubits,
                extra_controls,
            )?;
        }
        name => {
            let def = env
                .gate_defs
                .get(name)
                .cloned()
                .ok_or(KetError::OpenQasmSemanticError)?;
            check_arity(
                def.params.len(),
                call.params.len(),
                def.qargs.len(),
                call.qargs.len(),
            )?;

            let vals = eval_params(&call.params, param_env)?;
            let new_param_env: HashMap<String, f64> =
                def.params.iter().cloned().zip(vals).collect();

            let new_qubit_map: HashMap<String, usize> = def
                .qargs
                .iter()
                .cloned()
                .zip(qubits.iter().copied())
                .collect();

            expand_gate_body(
                block,
                env,
                &def.body,
                &new_param_env,
                &new_qubit_map,
                extra_controls,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// qelib1.inc gate dispatch
// ---------------------------------------------------------------------------

fn is_qelib1_gate(name: &str) -> bool {
    matches!(
        name,
        "u3" | "u2"
            | "u1"
            | "id"
            | "h"
            | "x"
            | "y"
            | "z"
            | "s"
            | "sdg"
            | "t"
            | "tdg"
            | "sx"
            | "sxdg"
            | "rx"
            | "ry"
            | "rz"
            | "cx"
            | "cy"
            | "cz"
            | "ch"
            | "cs"
            | "csdg"
            | "ct"
            | "ctdg"
            | "swap"
            | "crx"
            | "cry"
            | "crz"
            | "cu1"
            | "cu3"
            | "ccx"
            | "cswap"
            | "rxx"
            | "rzz"
    )
}

fn apply_qelib1_call(
    block: &mut BasicBlock,
    env: &Env,
    call: &GateCall,
    extra_controls: &[usize],
    param_env: &HashMap<String, f64>,
) -> Result<(), KetError> {
    let resolved: Vec<Vec<usize>> = call
        .qargs
        .iter()
        .map(|a| env.resolve_qarg(a))
        .collect::<Result<_, _>>()?;

    let n = resolved.iter().map(std::vec::Vec::len).max().unwrap_or(1);
    for v in &resolved {
        if v.len() != 1 && v.len() != n {
            return Err(KetError::OpenQasmSemanticError);
        }
    }

    for instance in 0..n {
        let qubits: Vec<usize> = resolved
            .iter()
            .map(|v| if v.len() == 1 { v[0] } else { v[instance] })
            .collect();
        apply_qelib1_gate_raw(
            block,
            &call.name,
            &call.params,
            param_env,
            &qubits,
            extra_controls,
        )?;
    }
    Ok(())
}

fn apply_qelib1_gate_raw(
    block: &mut BasicBlock,
    name: &str,
    raw_params: &[Expr],
    param_env: &HashMap<String, f64>,
    qubits: &[usize],
    extra_controls: &[usize],
) -> Result<(), KetError> {
    let params = eval_params(raw_params, param_env)?;

    macro_rules! q {
        ($n:expr) => {
            *qubits
                .get($n)
                .ok_or_else(|| KetError::OpenQasmSemanticError)?
        };
    }
    macro_rules! p {
        ($n:expr) => {
            *params
                .get($n)
                .ok_or_else(|| KetError::OpenQasmSemanticError)?
        };
    }

    let gate = match name {
        "u3" => Qelib1Gate::U3 {
            theta: p!(0),
            phi: p!(1),
            lambda: p!(2),
            target: q!(0),
        },
        "u2" => Qelib1Gate::U2 {
            phi: p!(0),
            lambda: p!(1),
            target: q!(0),
        },
        "u1" => Qelib1Gate::U1 {
            lambda: p!(0),
            target: q!(0),
        },
        "id" => Qelib1Gate::Id { target: q!(0) },
        "h" => Qelib1Gate::H { target: q!(0) },
        "x" => Qelib1Gate::X { target: q!(0) },
        "y" => Qelib1Gate::Y { target: q!(0) },
        "z" => Qelib1Gate::Z { target: q!(0) },
        "s" => Qelib1Gate::S { target: q!(0) },
        "sdg" => Qelib1Gate::Sdg { target: q!(0) },
        "t" => Qelib1Gate::T { target: q!(0) },
        "tdg" => Qelib1Gate::Tdg { target: q!(0) },
        "sx" => Qelib1Gate::Sx { target: q!(0) },
        "sxdg" => Qelib1Gate::Sxdg { target: q!(0) },
        "rx" => Qelib1Gate::Rx {
            theta: p!(0),
            target: q!(0),
        },
        "ry" => Qelib1Gate::Ry {
            theta: p!(0),
            target: q!(0),
        },
        "rz" => Qelib1Gate::Rz {
            phi: p!(0),
            target: q!(0),
        },
        "cx" => Qelib1Gate::Cx {
            control: q!(0),
            target: q!(1),
        },
        "cy" => Qelib1Gate::Cy {
            control: q!(0),
            target: q!(1),
        },
        "cz" => Qelib1Gate::Cz {
            control: q!(0),
            target: q!(1),
        },
        "ch" => Qelib1Gate::Ch {
            control: q!(0),
            target: q!(1),
        },
        "cs" => Qelib1Gate::Cs {
            control: q!(0),
            target: q!(1),
        },
        "csdg" => Qelib1Gate::Csdg {
            control: q!(0),
            target: q!(1),
        },
        "ct" => Qelib1Gate::Ct {
            control: q!(0),
            target: q!(1),
        },
        "ctdg" => Qelib1Gate::Ctdg {
            control: q!(0),
            target: q!(1),
        },
        "swap" => Qelib1Gate::Swap { a: q!(0), b: q!(1) },
        "crx" => Qelib1Gate::Crx {
            theta: p!(0),
            control: q!(0),
            target: q!(1),
        },
        "cry" => Qelib1Gate::Cry {
            theta: p!(0),
            control: q!(0),
            target: q!(1),
        },
        "crz" => Qelib1Gate::Crz {
            phi: p!(0),
            control: q!(0),
            target: q!(1),
        },
        "cu1" => Qelib1Gate::Cu1 {
            lambda: p!(0),
            control: q!(0),
            target: q!(1),
        },
        "cu3" => Qelib1Gate::Cu3 {
            theta: p!(0),
            phi: p!(1),
            lambda: p!(2),
            control: q!(0),
            target: q!(1),
        },
        "ccx" => Qelib1Gate::Ccx {
            c0: q!(0),
            c1: q!(1),
            target: q!(2),
        },
        "cswap" => Qelib1Gate::Cswap {
            control: q!(0),
            a: q!(1),
            b: q!(2),
        },
        "rxx" => Qelib1Gate::Rxx {
            theta: p!(0),
            a: q!(0),
            b: q!(1),
        },
        "rzz" => Qelib1Gate::Rzz {
            theta: p!(0),
            a: q!(0),
            b: q!(1),
        },
        _ => return Err(KetError::OpenQasmSemanticError),
    };

    if extra_controls.is_empty() {
        append_qelib1_gate(block, gate)?;
    } else {
        let mut inner = BasicBlock::new();
        append_qelib1_gate(&mut inner, gate)?;
        block.append_block(inner.control(extra_controls)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn eval_params(exprs: &[Expr], env: &HashMap<String, f64>) -> Result<Vec<f64>, KetError> {
    exprs
        .iter()
        .map(|e| e.eval(env).ok_or(KetError::OpenQasmSemanticError))
        .collect()
}

fn check_arity(
    expected_params: usize,
    got_params: usize,
    expected_qargs: usize,
    got_qargs: usize,
) -> Result<(), KetError> {
    if got_params != expected_params || got_qargs != expected_qargs {
        return Err(KetError::OpenQasmSemanticError);
    }
    Ok(())
}

fn broadcast_len(a: &[usize], b: &[usize]) -> Option<usize> {
    match (a.len(), b.len()) {
        (x, y) if x == y => Some(x),
        (1, y) => Some(y),
        (x, 1) => Some(x),
        _ => None,
    }
}
