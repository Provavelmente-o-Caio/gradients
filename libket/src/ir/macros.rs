// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use crate::error::KetError;
use crate::ir::block::Block;

type GeneratorFn = Box<dyn Fn(&[usize]) -> Result<Block, KetError> + Send + Sync>;

pub struct MacroRegistry {
    generators: RwLock<HashMap<String, GeneratorFn>>,
}

impl Default for MacroRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MacroRegistry {
    pub fn new() -> Self {
        let registry = Self {
            generators: RwLock::new(HashMap::new()),
        };

        registry.register("QFT", |qubits| {
            let mut block = Block::new();
            block.append_basic_block(crate::qft_internal(qubits)?);
            Ok(block)
        });

        registry.register("QFT_SWAP", |qubits| {
            let mut block = Block::new();
            for i in 0..(qubits.len() / 2) {
                let mut bb = crate::ir::block::BasicBlock::new();
                bb.append_block(crate::swap_gate(qubits[i], qubits[qubits.len() - i - 1])?);
                block.append_basic_block(bb);
            }
            Ok(block)
        });

        registry
    }

    pub fn register<F>(&self, name: &str, generator: F)
    where
        F: Fn(&[usize]) -> Result<Block, KetError> + Send + Sync + 'static,
    {
        self.generators
            .write()
            .unwrap()
            .insert(name.to_string(), Box::new(generator));
    }
}

pub static MACRO_REGISTRY: OnceLock<MacroRegistry> = OnceLock::new();

pub fn instantiate_macro(
    name: &str,
    qubits: &[usize],
    is_adjoint: bool,
) -> Result<Block, KetError> {
    let registry = MACRO_REGISTRY.get_or_init(MacroRegistry::new);
    let gen_lock = registry.generators.read().unwrap();
    let generator = gen_lock.get(name).ok_or(KetError::UnknownMacro)?;

    let mapped = generator(qubits)?;

    if is_adjoint {
        Ok(mapped.inverse())
    } else {
        Ok(mapped)
    }
}
