// SPDX-FileCopyrightText: 2025 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

mod kernel;

use crate::{
    bitwise::{get_ctrl_mask, is_one_at},
    block::BlockSimulator,
    quantum_execution::QuantumExecution,
    FloatOps,
};
use cubecl::{client::Client, prelude::*, server::Handle};
use itertools::Itertools;
use ket::{
    error::KetError,
    execution::DumpData,
    ir::hamiltonian::{Hamiltonian, Pauli},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{cell::RefCell, fmt::Display, marker::PhantomData, rc::Rc};

#[cube]
pub trait ConstantFloat {
    const ZERO: Self;
    const ONE: Self;
    const MINUS_ONE: Self;
    const FRAC_1_SQRT_2: Self;
}

#[cube]
impl ConstantFloat for f32 {
    const ZERO: Self = 0.0f32;
    const ONE: Self = 1.0f32;
    const MINUS_ONE: Self = -1.0f32;
    const FRAC_1_SQRT_2: Self = std::f32::consts::FRAC_1_SQRT_2;
}

/// Picks the device a [`DenseGPU`] simulator runs on.
///
/// CubeCL kernels and launches are runtime-agnostic: the runtime is chosen by
/// the [`cubecl::Device`] a [`Client`] is created from. Each implementor is a
/// zero-sized marker naming one device, so the backend is still fixed at the
/// type level (e.g. `DenseGPU<Wgpu, f32>` or `DenseGPU<Cuda, f32>`).
pub trait GpuDevice: 'static {
    /// The device to simulate on.
    ///
    /// # Errors
    ///
    /// Returns [`KetError::ExecutionFailed`] when the device is not available
    /// or its configuration is invalid.
    fn device() -> Result<cubecl::Device, KetError>;

    /// Creates the compute client for [`Self::device`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::device`].
    fn client() -> Result<Client, KetError> {
        Self::device().map(|device| device.client())
    }
}

#[cfg(feature = "gpu")]
/// The best GPU available through the WGPU runtime (Vulkan, Metal, DX12, ...).
///
/// The `CUBECL_WGPU_DEFAULT_DEVICE` environment variable overrides the choice.
pub struct Wgpu;

#[cfg(feature = "gpu")]
impl GpuDevice for Wgpu {
    fn device() -> Result<cubecl::Device, KetError> {
        cubecl::Device::wgpu(cubecl::device::WgpuDeviceKind::default())
            .map_err(|_| KetError::ExecutionFailed)
    }
}

#[cfg(feature = "cuda")]
/// An NVIDIA GPU on the CUDA runtime.
///
/// The `KBW_CUDA_DEVICE` environment variable selects the device by its CUDA
/// index (default `0`).
pub struct Cuda;

#[cfg(feature = "cuda")]
impl GpuDevice for Cuda {
    fn device() -> Result<cubecl::Device, KetError> {
        let index = match std::env::var("KBW_CUDA_DEVICE") {
            Ok(index) => index
                .trim()
                .parse::<usize>()
                .map_err(|_| KetError::ExecutionFailed)?,
            Err(_) => 0,
        };
        cubecl::Device::cuda(index).map_err(|_| KetError::ExecutionFailed)
    }
}

pub struct DenseGPU<
    D: GpuDevice,
    F: Float + CubeElement + FloatOps + ConstantFloat + LaunchArg<RuntimeArg = F>,
> {
    state_real: Rc<RefCell<cubecl::server::Handle>>,
    state_imag: Rc<RefCell<cubecl::server::Handle>>,
    prob_buffer: Rc<RefCell<cubecl::server::Handle>>,
    offset: usize,
    state_size: usize,
    num_qubits: usize,

    client: Rc<RefCell<Client>>,
    op_count: Rc<RefCell<usize>>,
    _f: PhantomData<F>,
    _d: PhantomData<D>,
}

fn launch_all<F>(num_qubits: usize, state_size: usize, kernel: F)
where
    F: Fn(u32, u32, u32),
{
    let (cube_count, cube_dim) = if num_qubits <= 10 {
        (1, state_size)
    } else {
        (1 << (num_qubits - 10), 1024)
    };

    let limit = 15;

    let (cube_count_2, cube_count_per_launch) = if cube_count <= (1 << limit) {
        (1, cube_count)
    } else {
        (cube_count >> limit, 1 << limit)
    };

    let threads_per_launch = (cube_count_per_launch * cube_dim) as u32;

    for i in 0..cube_count_2 {
        kernel(
            cube_count_per_launch as u32,
            cube_dim as u32,
            i as u32 * threads_per_launch,
        );
    }
}

impl<
        D: GpuDevice,
        F: Float + CubeElement + FloatOps + ConstantFloat + LaunchArg<RuntimeArg = F>,
    > DenseGPU<D, F>
{
    fn sync(&mut self) {
        if *self.op_count.borrow() * self.state_size > 10 << 27 {
            let client = self.client.borrow();
            let fut = async { client.sync().await };
            pollster::block_on(fut).unwrap();
            *self.op_count.borrow_mut() = 0;
        } else {
            *self.op_count.borrow_mut() += 1;
        }
    }
}

impl<
        D: GpuDevice,
        F: Float + CubeElement + FloatOps + ConstantFloat + LaunchArg<RuntimeArg = F>,
    > QuantumExecution for DenseGPU<D, F>
{
    fn new(num_qubits: usize) -> Result<Self, KetError>
    where
        Self: Sized,
    {
        let client = D::client()?;

        let state_size = 1usize << num_qubits;
        let state_size_bytes = state_size * std::mem::size_of::<F>();
        let state_imag = Rc::new(RefCell::new(client.empty(state_size_bytes)));
        let state_real = Rc::new(RefCell::new(client.empty(state_size_bytes)));

        launch_all(
            num_qubits,
            state_size,
            |cube_count, cube_dim, offset| unsafe {
                kernel::init_state::launch_unchecked::<F>(
                    &client,
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(state_real.borrow().clone(), state_size),
                    BufferArg::from_raw_parts(state_imag.borrow().clone(), state_size),
                    offset,
                );
            },
        );

        let prob_buffer = Rc::new(RefCell::new(client.empty(state_size_bytes)));

        Ok(Self {
            state_real,
            state_imag,
            prob_buffer,
            op_count: Rc::new(RefCell::new(0)),
            offset: 0,
            state_size,
            num_qubits,
            client: Rc::new(RefCell::new(client)),
            _f: PhantomData,
            _d: PhantomData,
        })
    }

    fn pauli_x(&mut self, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (stride_x, stride_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_x::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        x * stride_x,
                        y * stride_y,
                    );
                }
            }
        }
    }

    fn pauli_y(&mut self, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_y::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn pauli_z(&mut self, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_z::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn hadamard(&mut self, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_h::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn phase(&mut self, lambda: f64, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_p::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        F::from(lambda.cos()).unwrap(),
                        F::from(lambda.sin()).unwrap(),
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn rx(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_rx::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        F::from((theta / 2.0).cos()).unwrap(),
                        F::from(-(theta / 2.0).sin()).unwrap(),
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn ry(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_ry::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        F::from((theta / 2.0).cos()).unwrap(),
                        F::from((theta / 2.0).sin()).unwrap(),
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn rz(&mut self, theta: f64, target: usize, control: &[usize]) {
        let (
            half_block_size,
            full_block_size,
            (cube_count_2_x, cube_count_2_y),
            (cube_size_x, cube_size_y),
            cube_count,
            cube_dim,
        ) = kernel::compute_cube_size(self.num_qubits, target);
        self.sync();

        for x in 0..cube_count_2_x {
            for y in 0..cube_count_2_y {
                unsafe {
                    kernel::gate_rz::launch_unchecked::<F>(
                        &self.client.borrow(),
                        cube_count.clone(),
                        cube_dim,
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        get_ctrl_mask(control),
                        half_block_size,
                        full_block_size,
                        F::from((theta / 2.0).cos()).unwrap(),
                        F::from((theta / 2.0).sin()).unwrap(),
                        x * cube_size_x,
                        y * cube_size_y,
                    );
                }
            }
        }
    }

    fn measure_p1(&mut self, target: usize) -> f64 {
        let prob_size = 1 << (self.num_qubits - 1);
        let prob = self.prob_buffer.borrow().clone();

        let mask = (1 << target) - 1;

        launch_all(
            self.num_qubits - 1,
            self.state_size >> 1,
            |cube_count, cube_dim, offset| unsafe {
                kernel::measure_p1::launch_unchecked::<F>(
                    &self.client.borrow(),
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(self.state_real.borrow().clone(), self.state_size),
                    BufferArg::from_raw_parts(self.state_imag.borrow().clone(), self.state_size),
                    offset as usize,
                    self.offset,
                    BufferArg::from_raw_parts(prob.clone(), prob_size),
                    target as u32,
                    mask,
                );
            },
        );

        let prob = self.client.borrow().read_one(prob).unwrap();
        let prob = F::from_bytes(&prob);
        let prob = &prob[..prob_size];

        prob.par_iter().copied().sum::<F>().to_f64().unwrap()
    }

    fn measure_collapse(&mut self, target: usize, result: bool, p: f64) {
        let mask = 1 << target;

        launch_all(
            self.num_qubits,
            self.state_size,
            |cube_count, cube_dim, offset| unsafe {
                kernel::measure_collapse::launch_unchecked::<F>(
                    &self.client.borrow(),
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(self.state_real.borrow().clone(), self.state_size),
                    BufferArg::from_raw_parts(self.state_imag.borrow().clone(), self.state_size),
                    self.offset + offset as usize,
                    mask,
                    if result { mask } else { 0 },
                    F::from(p).unwrap(),
                );
            },
        );
    }

    fn dump(&mut self, qubits: &[usize]) -> DumpData {
        let offset_bytes_begin = self.offset * std::mem::size_of::<F>();
        let offset_bytes_end = offset_bytes_begin + self.state_size * std::mem::size_of::<F>();
        let state_real = self
            .client
            .borrow()
            .read_one(self.state_real.borrow().clone())
            .unwrap();
        let state_real = F::from_bytes(&state_real[offset_bytes_begin..offset_bytes_end]);
        let state_imag = self
            .client
            .borrow()
            .read_one(self.state_imag.borrow().clone())
            .unwrap();
        let state_imag = F::from_bytes(&state_imag[offset_bytes_begin..offset_bytes_end]);

        let (basis_states, amplitudes_real, amplitudes_imag): (Vec<_>, Vec<_>, Vec<_>) = state_real
            .iter()
            .zip(state_imag)
            .enumerate()
            .filter(|(_state, (r, i))| {
                num_traits::real::Real::sqrt(**r * **r + **i * **i)
                    > F::from(F::small_epsilon()).unwrap()
            })
            .map(|(state, (r, i))| {
                let state = qubits
                    .iter()
                    .rev()
                    .enumerate()
                    .map(|(index, qubit)| usize::from(is_one_at(state, *qubit)) << index)
                    .reduce(|a, b| a | b)
                    .unwrap_or(0);

                (
                    Vec::from([state as u64]),
                    r.to_f64().unwrap(),
                    i.to_f64().unwrap(),
                )
            })
            .multiunzip();

        DumpData {
            basis_states,
            amplitudes_real,
            amplitudes_imag,
        }
    }

    fn exp_value(&mut self, hamiltonian: &Hamiltonian) -> f64 {
        let (cube_count, cube_dim) = if self.num_qubits <= 10 {
            (1, self.state_size)
        } else {
            (1 << (self.num_qubits - 10), 1024)
        };

        hamiltonian
            .pauli_strings
            .iter()
            .map(|pauli_terms| {
                for term in pauli_terms {
                    match term.pauli {
                        Pauli::PauliX => self.hadamard(term.qubit, &[]),
                        Pauli::PauliY => {
                            self.phase(-std::f64::consts::FRAC_PI_2, term.qubit, &[]);
                            self.hadamard(term.qubit, &[]);
                        }
                        Pauli::PauliZ => {}
                    }
                }

                let prob = self.prob_buffer.borrow().clone();

                let mut target_mask = 0;
                for q in pauli_terms.iter().map(|term| term.qubit) {
                    target_mask |= 1 << q;
                }

                unsafe {
                    kernel::exp_value::launch_unchecked::<F>(
                        &self.client.borrow(),
                        CubeCount::new_1d(cube_count as u32),
                        CubeDim::new_1d(cube_dim as u32),
                        BufferArg::from_raw_parts(
                            self.state_real.borrow().clone(),
                            self.state_size,
                        ),
                        BufferArg::from_raw_parts(
                            self.state_imag.borrow().clone(),
                            self.state_size,
                        ),
                        self.offset,
                        BufferArg::from_raw_parts(prob.clone(), self.state_size),
                        target_mask,
                    );
                }

                let prob = self.client.borrow().read_one(prob).unwrap();
                let prob = F::from_bytes(&prob);
                let prob = &prob[..self.state_size];

                let result: F = prob.par_iter().copied().sum();

                pauli_terms.iter().for_each(|term| match term.pauli {
                    Pauli::PauliX => self.hadamard(term.qubit, &[]),
                    Pauli::PauliY => {
                        self.hadamard(term.qubit, &[]);
                        self.phase(std::f64::consts::FRAC_PI_2, term.qubit, &[]);
                    }
                    Pauli::PauliZ => {}
                });

                result.to_f64().unwrap()
            })
            .zip(&hamiltonian.coefficients)
            .map(|(result, coefficient)| result * *coefficient)
            .sum()
    }

    fn clear(&mut self) {
        let state_size = 1usize << self.num_qubits;
        launch_all(
            self.num_qubits,
            state_size,
            |cube_count, cube_dim, offset| unsafe {
                kernel::init_state::launch_unchecked::<F>(
                    &self.client.borrow(),
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(self.state_real.borrow().clone(), state_size),
                    BufferArg::from_raw_parts(self.state_imag.borrow().clone(), state_size),
                    offset,
                );
            },
        );
    }
}

impl<
        D: GpuDevice,
        F: Float + CubeElement + FloatOps + Display + ConstantFloat + LaunchArg<RuntimeArg = F>,
    >
    BlockSimulator<
        Self,
        (
            Rc<RefCell<Client>>,
            Rc<RefCell<usize>>,
            Rc<RefCell<Handle>>,
            Rc<RefCell<Handle>>,
            usize,
        ),
    > for DenseGPU<D, F>
{
    fn new_blocks(
        num_local_qubits: usize,
        num_global_qubits: usize,
    ) -> Result<
        (
            Vec<Self>,
            (
                Rc<RefCell<Client>>,
                Rc<RefCell<usize>>,
                Rc<RefCell<Handle>>,
                Rc<RefCell<Handle>>,
                usize,
            ),
        ),
        KetError,
    > {
        let num_qubits = num_local_qubits + num_global_qubits;

        let client = Rc::new(RefCell::new(D::client()?));

        let state_size = 1usize << num_qubits;
        let state_size_bytes = state_size * std::mem::size_of::<F>();
        let state_imag = Rc::new(RefCell::new(client.borrow().empty(state_size_bytes)));
        let state_real = Rc::new(RefCell::new(client.borrow().empty(state_size_bytes)));
        let prob_buffer = Rc::new(RefCell::new(client.borrow().empty(state_size_bytes)));

        let op_count = Rc::new(RefCell::new(0));

        launch_all(
            num_qubits,
            state_size,
            |cube_count, cube_dim, offset| unsafe {
                kernel::init_state::launch_unchecked::<F>(
                    &client.borrow(),
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(state_real.borrow().clone(), state_size),
                    BufferArg::from_raw_parts(state_imag.borrow().clone(), state_size),
                    offset,
                );
            },
        );

        let mut simulators = Vec::new();

        let sub_state_size = 1 << num_local_qubits;
        for i in 0..(1 << num_global_qubits) {
            simulators.push(Self {
                state_real: Rc::clone(&state_real),
                state_imag: Rc::clone(&state_imag),
                prob_buffer: Rc::clone(&prob_buffer),
                offset: i * sub_state_size,
                state_size: sub_state_size,
                num_qubits: num_local_qubits,
                client: Rc::clone(&client),
                op_count: Rc::clone(&op_count),
                _f: PhantomData,
                _d: PhantomData,
            });
        }

        Ok((
            simulators,
            (client, op_count, state_real, state_imag, state_size),
        ))
    }

    fn swap(
        global_data: &mut (
            Rc<RefCell<Client>>,
            Rc<RefCell<usize>>,
            Rc<RefCell<Handle>>,
            Rc<RefCell<Handle>>,
            usize,
        ),
        _simulators: &mut [Self],
        num_global_qubits: usize,
        num_local_qubits: usize,
        global_qubit: usize,
        local_qubit: usize,
    ) {
        let num_qubits = num_local_qubits + num_global_qubits;
        let state_size = 1usize << num_qubits;

        let (client, _, state_real, state_imag, _) = global_data;

        let global_index = global_qubit + num_local_qubits;
        let bit1 = std::cmp::min(global_index, local_qubit);
        let bit2 = std::cmp::max(global_index, local_qubit);
        let mask1 = (1 << bit1) - 1;
        let mask2 = (1 << bit2) - 1;

        launch_all(
            num_qubits - 2,
            state_size >> 2,
            |cube_count, cube_dim, offset| unsafe {
                kernel::in_place_swap::launch_unchecked::<F>(
                    &client.borrow(),
                    CubeCount::new_1d(cube_count),
                    CubeDim::new_1d(cube_dim),
                    BufferArg::from_raw_parts(state_real.borrow().clone(), state_size),
                    BufferArg::from_raw_parts(state_imag.borrow().clone(), state_size),
                    offset,
                    global_index as u32,
                    local_qubit as u32,
                    bit1 as u32,
                    bit2 as u32,
                    mask1 as u32,
                    mask2 as u32,
                );
            },
        );
    }

    fn print_global_state(
        global_data: &(
            Rc<RefCell<Client>>,
            Rc<RefCell<usize>>,
            Rc<RefCell<Handle>>,
            Rc<RefCell<Handle>>,
            usize,
        ),
        _: &[Self],
    ) {
        let (client, _, state_real_handle, state_imag_handle, state_size) = global_data;
        let state_real = client
            .borrow()
            .read_one(state_real_handle.borrow().clone())
            .unwrap();
        let state_real = F::from_bytes(&state_real);
        let state_real = &state_real[..*state_size];

        let state_imag = client
            .borrow()
            .read_one(state_imag_handle.borrow().clone())
            .unwrap();
        let state_imag = F::from_bytes(&state_imag);
        let state_imag = &state_imag[..*state_size];

        let n = state_real.len().trailing_zeros() as usize;

        println!("==============");
        for (state, (re, im)) in state_real.iter().zip(state_imag).enumerate() {
            println!("{state:0n$b}: ({re:.4} {im:+.4})");
        }
        println!("==============");
    }
}
