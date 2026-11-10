// SPDX-FileCopyrightText: 2020 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
// SPDX-FileCopyrightText: 2020 Rafael de Santiago <r.santiago@ufsc.br>
//
// SPDX-License-Identifier: Apache-2.0

//! # KBW: Ket Bitwise Quantum Simulator
//!
//! KBW is a high-performance quantum circuit simulator designed to work with the
//! [Ket](https://quantumket.org) quantum programming framework. It provides multiple
//! simulation backends that trade off memory usage against speed for different
//! circuit sizes and hardware targets.
//!
//! ## Simulator Backends
//!
//! | Type alias | Description |
//! |---|---|
//! | [`DenseSimulator`] | Full state-vector (double-buffer), single process |
//! | [`DenseV2Simulator`] | Full state-vector (in-place), single process |
//! | [`DenseV2BlockSimulator`] | Blocked state-vector for >local-memory qubit counts |
//! | [`DenseGPUSimulator`] | GPU-accelerated full state-vector via [CubeCL](https://github.com/tracel-ai/cubecl) |
//! | [`DenseGPUBlockSimulator`] | Blocked GPU state-vector for large circuits |
//! | [`DenseCUDASimulator`] | NVIDIA GPU full state-vector via the CubeCL CUDA runtime (`cuda` feature) |
//! | [`DenseCUDABlockSimulator`] | Blocked CUDA state-vector for large circuits (`cuda` feature) |
//! | [`SparseSimulator`] | Sparse (HashMap-based) state-vector, single process |
//! | [`SparseV2Simulator`] | Sparse (Vec-based, cache-friendly) state-vector |
//!
//! ## Execution Modes
//!
//! Each simulator can be used in two execution modes, configured via
//! [`quantum_execution::QubitManager`]:
//!
//! - **Live execution** ([`QubitManager::make_live_config`](quantum_execution::QubitManager::make_live_config)):
//!   Executes each gate immediately as it is received. Best suited for interactive
//!   or streaming workloads.
//! - **Batch execution** ([`QubitManager::make_batch_config`](quantum_execution::QubitManager::make_batch_config)):
//!   Accumulates a full circuit description (including coupling-graph constraints
//!   and gradient strategies) and executes it in one shot. Recommended for
//!   variational algorithms and optimisation loops.
//!
//! ## C API
//!
//! A C-compatible ABI is exposed by [`c_api`] for embedding KBW in Python
//! extensions or other FFI consumers. The entry point is
//! [`c_api::kbw_make_configuration`].
//!
//! ## Environment Variables
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `KBW_SEED` | random | Unsigned 64-bit integer seed for the RNG used by measurement and sampling. Set for deterministic results. |
//! | `KBW_BLOCK_SIZE` | `20` | Number of *local* qubits per block in the blocked simulators. The remaining qubits become *global* and trigger inter-block SWAP operations. |
//! | `KBW_CUDA_DEVICE` | `0` | Index of the NVIDIA GPU used by the CUDA simulators (`cuda` feature). An invalid value makes simulator creation fail with `ExecutionFailed`. |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use kbw::DenseSimulator;
//! use ket::error::KetError;
//!
//! fn main() -> Result<(), KetError> {
//!     let num_qubits = 10;
//!     // Build a live-mode configuration for 10 qubits using the dense backend.
//!     let config = DenseSimulator::make_live_config(num_qubits, false)?;
//!     // Pass `config` to `ket::process::Process::new` to run a quantum program.
//!     Ok(())
//! }
//! ```

use std::{
    iter::Sum,
    ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign},
};

pub mod bitwise;
#[cfg(feature = "block")]
pub mod block;
pub mod c_api;
pub mod convert;
#[cfg(feature = "dense")]
pub mod dense;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod dense_gpu;
#[cfg(feature = "dense")]
pub mod dense_v2;
pub mod quantum_execution;
#[cfg(feature = "sparse")]
pub mod sparse;
#[cfg(feature = "sparse")]
pub mod sparse_v2;

/// Extension trait that adds simulator-specific helpers on top of the standard
/// [`num_traits::Float`] and arithmetic-assignment bounds.
///
/// All generic simulator implementations (`Dense`, `Sparse`, `DenseV2`, …) are
/// parameterised over `F: FloatOps` so that the same code can run in either
/// single-precision (`f32`) or double-precision (`f64`) mode.
pub trait FloatOps:
    num_traits::Float
    + num_traits::FloatConst
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + RemAssign
    + Sum
    + Send
    + Sync
{
    /// Returns the amplitude-magnitude threshold below which a basis state is
    /// considered zero and may be pruned from the state representation.
    ///
    /// The threshold is expressed in the *same* floating-point type as `Self`
    /// so that comparisons are performed without implicit precision loss.
    fn small_epsilon() -> f64;
}

impl FloatOps for f32 {
    /// Threshold for `f32`: `1e-6` (well above the `f32` machine epsilon of ~`1.2e-7`).
    fn small_epsilon() -> f64 {
        1e-6
    }
}
impl FloatOps for f64 {
    /// Threshold for `f64`: `1e-15` (just above the `f64` machine epsilon of ~`2.2e-16`).
    fn small_epsilon() -> f64 {
        1e-15
    }
}

#[cfg(feature = "dense")]
/// Double-buffered, dense state-vector simulator using `f32` amplitudes.
/// Supports up to 32 qubits.
pub type Dense = dense::Dense<f32>;

#[cfg(feature = "dense")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`Dense`] backend.
/// The recommended entry point for single-process dense simulation.
pub type DenseSimulator = quantum_execution::QubitManager<Dense>;

#[cfg(feature = "dense")]
/// In-place dense state-vector simulator using `f32` amplitudes.
/// Slightly lower memory overhead than [`Dense`] due to the single-buffer design.
pub type DenseV2 = dense_v2::DenseV2<f32>;

#[cfg(feature = "dense")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseV2`] backend.
pub type DenseV2Simulator = quantum_execution::QubitManager<DenseV2>;

#[cfg(all(feature = "block", feature = "dense"))]
/// Blocked [`DenseV2`] simulator that partitions the qubit register into
/// *local* (simulated) and *global* (index) portions, enabling circuits
/// with more qubits than fit in a single state-vector.
pub type DenseV2Block = block::Block<(), DenseV2>;

#[cfg(all(feature = "block", feature = "dense"))]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseV2Block`] backend.
pub type DenseV2BlockSimulator = quantum_execution::QubitManager<DenseV2Block>;

#[cfg(any(feature = "gpu", feature = "cuda"))]
/// Global data shared by the blocks of a blocked GPU simulator: compute client,
/// operation counter, real and imaginary state buffers, and full state size.
pub type DenseGPUBlockData = (
    std::rc::Rc<std::cell::RefCell<cubecl::client::Client>>,
    std::rc::Rc<std::cell::RefCell<usize>>,
    std::rc::Rc<std::cell::RefCell<cubecl::server::Handle>>,
    std::rc::Rc<std::cell::RefCell<cubecl::server::Handle>>,
    usize,
);

#[cfg(feature = "gpu")]
/// GPU-accelerated dense state-vector simulator using the WGPU runtime and `f32` amplitudes.
/// Requires a compatible GPU and the `gpu` feature.
pub type DenseGPU = dense_gpu::DenseGPU<dense_gpu::Wgpu, f32>;

#[cfg(feature = "gpu")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseGPU`] backend.
pub type DenseGPUSimulator = quantum_execution::QubitManager<DenseGPU>;

#[cfg(feature = "gpu")]
/// Blocked GPU dense state-vector; combines the WGPU backend with the block
/// partitioning scheme for circuits exceeding on-device memory.
pub type DenseGPUBlock = block::Block<DenseGPUBlockData, DenseGPU>;

#[cfg(feature = "gpu")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseGPUBlock`] backend.
pub type DenseGPUBlockSimulator = quantum_execution::QubitManager<DenseGPUBlock>;

#[cfg(feature = "cuda")]
/// GPU-accelerated dense state-vector simulator using the CUDA runtime and `f32` amplitudes.
/// Requires an NVIDIA GPU and the `cuda` feature.
pub type DenseCUDA = dense_gpu::DenseGPU<dense_gpu::Cuda, f32>;

#[cfg(feature = "cuda")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseCUDA`] backend.
pub type DenseCUDASimulator = quantum_execution::QubitManager<DenseCUDA>;

#[cfg(feature = "cuda")]
/// Blocked CUDA dense state-vector; combines the CUDA backend with the block
/// partitioning scheme for circuits exceeding on-device memory.
pub type DenseCUDABlock = block::Block<DenseGPUBlockData, DenseCUDA>;

#[cfg(feature = "cuda")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`DenseCUDABlock`] backend.
pub type DenseCUDABlockSimulator = quantum_execution::QubitManager<DenseCUDABlock>;

#[cfg(feature = "sparse")]
/// Sparse state-vector simulator backed by a [`HashMap`](std::collections::HashMap),
/// using `f32` amplitudes. Efficient for circuits that remain in a low-entanglement
/// regime (e.g. few non-zero amplitudes).
pub type Sparse = sparse::Sparse<f32>;

#[cfg(feature = "sparse")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`Sparse`] backend.
pub type SparseSimulator = quantum_execution::QubitManager<Sparse>;

#[cfg(feature = "sparse")]
/// Sparse state-vector simulator backed by a sorted `Vec`, providing better
/// cache locality than [`Sparse`] at the cost of periodic sort-and-reduce passes.
pub type SparseV2 = sparse_v2::SparseV2<f32>;

#[cfg(feature = "sparse")]
/// [`QubitManager`](quantum_execution::QubitManager) wrapping the [`SparseV2`] backend.
pub type SparseV2Simulator = quantum_execution::QubitManager<SparseV2>;
