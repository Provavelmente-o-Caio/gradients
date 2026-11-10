// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::f64::consts::FRAC_PI_4;

use ket::{
    compute_action_uncompute_gate, controlled_gate,
    error::KetError,
    ir::{block::BasicBlock, gate::QuantumGate},
    process::Process,
};

fn main() -> Result<(), KetError> {
    let num_qubits = 16;
    let simulator = kbw::DenseSimulator::make_live_config(num_qubits, false)?;

    let mut process = Process::new(simulator);

    let qubits: Result<Vec<_>, _> = (0..num_qubits).map(|_| process.alloc()).collect();
    let qubits = qubits?;

    let mut quantum_code = BasicBlock::new();

    for target in &qubits {
        quantum_code.append_gate(QuantumGate::Hadamard, *target);
    }

    process.append_block(quantum_code)?;

    let steps = ((FRAC_PI_4) * f64::sqrt((1 << num_qubits) as f64)) as i64;

    let start = std::time::Instant::now();

    for _ in 0..steps {
        process.append_block(controlled_gate(
            QuantumGate::PauliZ,
            &qubits[1..],
            qubits[0],
        )?)?;

        process.append_block(compute_action_uncompute_gate(
            |mut compute| {
                for target in &qubits {
                    compute.append_gate(QuantumGate::Hadamard, *target);
                    compute.append_gate(QuantumGate::PauliX, *target);
                }
                Ok(compute)
            },
            |mut action| {
                action.append_gate(QuantumGate::PauliZ, qubits[0]);
                Ok(action.control(&qubits[1..])?)
            },
        )?)?;
    }

    let result = process.measure(&qubits)?;
    let duration = start.elapsed();
    println!("Execution Time: {:?}", duration);
    println!("Result={}", result);

    Ok(())
}
