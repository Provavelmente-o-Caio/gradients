// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::PI;

use ket::{
    controlled_gate,
    error::KetError,
    ir::{
        block::BasicBlock,
        gate::{Param, QuantumGate},
    },
    process::{Process, QPUConfig},
    swap_gate,
};

fn qft(qubits: &[usize], do_swap: bool) -> Result<BasicBlock, KetError> {
    let mut quantum_code = BasicBlock::new();
    if qubits.len() == 1 {
        quantum_code.append_gate(QuantumGate::Hadamard, qubits[0]);
        return Ok(quantum_code);
    }

    let init = &qubits[..qubits.len() - 1];
    let last = qubits[qubits.len() - 1];
    quantum_code.append_gate(QuantumGate::Hadamard, last);
    for (i, c) in init.iter().enumerate() {
        quantum_code.append_block(controlled_gate(
            QuantumGate::Phase(Param::Value(PI / 2.0_f64.powi(i as i32 + 1))),
            &[*c],
            last,
        )?);
    }
    quantum_code.append_block(qft(init, false)?);

    if do_swap {
        for i in 0..qubits.len() / 2 {
            quantum_code.append_block(swap_gate(qubits[i], qubits[qubits.len() - i - 1])?);
        }
    }

    Ok(quantum_code)
}

fn main() -> Result<(), KetError> {
    let num_qubits = 12;

    let mut process = Process::new(QPUConfig {
        num_qubits,
        quantum_execution: None,
    });

    let qubits: Result<Vec<_>, _> = (0..num_qubits).map(|_| process.alloc()).collect();
    let qubits = qubits?;

    process.append_block(qft(&qubits, true)?.into())?;

    println!("{:#?}", process);

    Ok(())
}
