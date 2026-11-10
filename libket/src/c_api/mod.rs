// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::missing_safety_doc,
    clippy::manual_let_else,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::macro_metavars_in_unsafe
)]

//! Foreign Function Interface (FFI) for the Libket library.
//!
//! This module provides a C-compatible API for the core data structures and
//! execution mechanisms in Libket, enabling it to be consumed by other languages
//! such as Python, C, and C++.
//!
//! # Design overview
//!
//! All opaque Rust types ([`crate::process::Process`],
//! [`crate::ir::block::BasicBlock`], [`crate::process::QPUConfig`],
//! [`crate::execution::QuantumExecution`]) are heap-allocated behind raw pointers.
//! Callers are responsible for pairing every `*_new` call with the corresponding
//! `*_delete` call to avoid memory leaks.
//!
//! Gate instructions and measurement results are exchanged as **JSON strings**
//! (`const char *`). The caller owns any `char *` that Libket writes into an
//! output pointer and **must** free it by calling [`ket_string_delete`] when
//! done. Static C strings (e.g., the status returned by
//! `ket_process_status`) are not heap-allocated and must **not** be freed.
//!
//! Every exported function returns a signed 32-bit integer error code. A
//! return value of `0` indicates success. Non-zero values map to the
//! [`KetError`] variants via [`KetError::from_error_code`]; call
//! [`ket_error_message`] to retrieve a human-readable description.

macro_rules! c_error {
    ($ret:expr) => {
        match $ret {
            Ok(ret) => ret,
            Err(err) => return err.error_code(),
        }
    };
}

macro_rules! to_json {
    ($data:expr) => {
        match serde_json::to_string($data) {
            Ok(data) => {
                if let Ok(data) = CString::new(data) {
                    data
                } else {
                    return KetError::SerdeError.error_code();
                }
            }
            Err(_) => return KetError::SerdeError.error_code(),
        }
    };
}

macro_rules! from_json {
    ($data:expr, $ty:ty) => {{
        let data = unsafe { std::ffi::CStr::from_ptr($data) };
        let Ok(data) = data.to_str() else {
            return KetError::SerdeError.error_code();
        };
        match serde_json::from_str::<$ty>(data) {
            Ok(data) => data,
            Err(_) => return KetError::SerdeError.error_code(),
        }
    }};
}

pub mod block;
pub mod execute;
pub mod process;

use std::ffi::{c_char, CString};

use crate::error::KetError;

/// Retrieves the human-readable error message for a given integer error code.
///
/// On success the function heap-allocates a null-terminated C string and
/// writes its address into `*error_message`. The caller must free this string
/// with [`ket_string_delete`] when done.
///
/// # Returns
/// `0` on success; a non-zero error code if the C string could not be
/// allocated (e.g., if the message contains interior null bytes).
#[no_mangle]
pub extern "C" fn ket_error_message(error_code: i32, error_message: *mut *const c_char) -> i32 {
    let Ok(msg) = CString::new(KetError::from_error_code(error_code).to_string()) else {
        return 1;
    };

    unsafe {
        *error_message = msg.into_raw();
    }

    c_ok()
}

/// Frees a heap-allocated C string that was previously written by Libket.
///
/// Every `const char **` output parameter written by Libket functions
/// (`ket_error_message`, `ket_process_gates_json`, `ket_process_dump`, etc.)
/// points to memory that must be released through this function. Passing a
/// pointer that was not produced by Libket, or freeing the same pointer twice,
/// is undefined behaviour.
///
/// # Returns
/// Always returns `0`.
#[no_mangle]
pub extern "C" fn ket_string_delete(ptr: *mut c_char) -> i32 {
    let _ = unsafe { CString::from_raw(ptr) };
    c_ok()
}

/// Returns the success error code (`0`).
#[inline(always)]
pub(super) fn c_ok() -> i32 {
    KetError::Success.error_code()
}

/// Converts a `Result<(), KetError>` to a C-compatible integer error code.
///
/// Returns `0` on `Ok(())`, or the discriminant of the `KetError` variant
/// on `Err`. Suitable for wrapping infallible-looking calls that propagate
/// errors through the return value.
pub(super) fn c_error(error: Result<(), KetError>) -> i32 {
    match error {
        Ok(()) => c_ok(),
        Err(error) => error.error_code(),
    }
}

const BUILD_INFO: &str = build_info::format!("{} v{} [{} {}]", $.crate_info.name, $.crate_info.version, $.compiler, $.target);

/// Returns a pointer and byte-length for a static build-information string.
///
/// The string is embedded at compile time and contains the crate name,
/// version, Rust compiler version, and compilation target triple. It is
/// **not** null-terminated; use `size` to bound reads. The pointer is valid
/// for the lifetime of the program and must **not** be freed.
///
/// # Parameters
/// - `msg` : Set to the address of the first byte of the build-info string.
/// - `size`: Set to the byte length of the build-info string.
///
/// # Returns
/// Always returns `0`.
#[no_mangle]
pub const extern "C" fn ket_build_info(msg: &mut *const u8, size: &mut usize) -> i32 {
    let bytes = BUILD_INFO.as_bytes();
    *size = bytes.len();
    *msg = bytes.as_ptr();
    KetError::Success.error_code()
}
