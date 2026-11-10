// SPDX-FileCopyrightText: 2026 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

use num::Complex;

/// A complex number with 64-bit floating-point components.
pub type Cf64 = Complex<f64>;

/// A 2×2 complex matrix representing a single-qubit unitary.
pub type Matrix = [[Cf64; 2]; 2];

#[inline]
pub(crate) fn identity() -> Matrix {
    [
        [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
        [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
    ]
}

/// Returns `true` when `m` is (numerically) the identity.
#[inline]
pub(crate) fn is_identity(m: &Matrix) -> bool {
    const EPS: f64 = 1e-12;
    let [[a, b], [c, d]] = m;
    (a.re - 1.0).abs() < EPS
        && a.im.abs() < EPS
        && b.norm() < EPS
        && c.norm() < EPS
        && (d.re - 1.0).abs() < EPS
        && d.im.abs() < EPS
}

/// Multiplies two 2×2 complex matrices: returns `lhs · rhs`.
///
/// `lhs` is the *later* gate (applied after `rhs`), so the product represents
/// the composed unitary `lhs ∘ rhs`.
#[inline]
pub(crate) fn mat_mul(lhs: &Matrix, rhs: &Matrix) -> Matrix {
    let [[a, b], [c, d]] = lhs;
    let [[e, f], [g, h]] = rhs;
    [
        [a * e + b * g, a * f + b * h],
        [c * e + d * g, c * f + d * h],
    ]
}
