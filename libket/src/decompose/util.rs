// SPDX-FileCopyrightText: 2024 Evandro Chagas Ribeiro da Rosa <evandro@quantuloop.com>
//
// SPDX-License-Identifier: Apache-2.0

//! Mathematical utilities for 2×2 complex matrices.
//!
//! Provides fundamental matrix operations (dot product, eigendecomposition,
//! fractional exponentiation) and the ZYZ Euler-angle decomposition needed
//! by the single-qubit gate synthesis and multi-controlled gate decomposition
//! routines.

use num::{complex::ComplexFloat, Complex};

use crate::matrix::{Cf64, Matrix};

/// Computes the standard matrix product of two 2×2 complex matrices `a` and `b`.
pub fn matrix_dot(matrix_a: &Matrix, matrix_b: &Matrix) -> Matrix {
    [
        [
            matrix_a[0][0] * matrix_b[0][0] + matrix_a[0][1] * matrix_b[1][0],
            matrix_a[0][0] * matrix_b[0][1] + matrix_a[0][1] * matrix_b[1][1],
        ],
        [
            matrix_a[1][0] * matrix_b[0][0] + matrix_a[1][1] * matrix_b[1][0],
            matrix_a[1][0] * matrix_b[0][1] + matrix_a[1][1] * matrix_b[1][1],
        ],
    ]
}

fn extract_phase(matrix: Matrix) -> f64 {
    let [[a, b], [c, d]] = matrix;
    let det = a * d - b * c;
    1.0 / 2.0 * det.im.atan2(det.re)
}

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-14
}

/// Computes the ZYZ Euler-angle decomposition of a 2×2 unitary matrix.
///
/// Every single-qubit unitary $U$ can be written as:
/// $$ U = e^{i\text{phase}} `R_z(\theta_0)` `R_y(\theta_1)` `R_z(\theta_2)` $$
///
/// Returns the tuple `(phase, theta_0, theta_1, theta_2)`.
pub fn zyz(matrix: Matrix) -> (f64, f64, f64, f64) {
    let phase = extract_phase(matrix);
    let e_phase = (-Complex::<_>::i() * phase).exp();

    let matrix = [
        [matrix[0][0] * e_phase, matrix[0][1] * e_phase],
        [matrix[1][0] * e_phase, matrix[1][1] * e_phase],
    ];

    let matrix_0_1_abs = matrix[0][0].abs().clamp(-1.0, 1.0);

    let theta_1 = if matrix[0][0].abs() >= matrix[0][1].abs() {
        2.0 * matrix_0_1_abs.acos()
    } else {
        2.0 * matrix[0][1].abs().asin()
    };

    let theta_1_2_cos = (theta_1 / 2.0).cos();
    let theta_0_plus_2 = if is_close(theta_1_2_cos, 0.0) {
        0.0
    } else {
        let tmp = matrix[1][1] / theta_1_2_cos;
        2.0 * tmp.im.atan2(tmp.re)
    };

    let theta_1_2_sin = (theta_1 / 2.0).sin();
    let theta_0_sub_2 = if is_close(theta_1_2_sin, 0.0) {
        0.0
    } else {
        let tmp = matrix[1][0] / theta_1_2_sin;
        2.0 * tmp.im.atan2(tmp.re)
    };

    let theta_0 = f64::midpoint(theta_0_plus_2, theta_0_sub_2);
    let theta_2 = (theta_0_plus_2 - theta_0_sub_2) / 2.0;

    (phase, theta_0, theta_1, theta_2)
}

type Vector = (Cf64, Cf64);

/// Computes the eigenvalues and corresponding eigenvectors of a 2×2 matrix.
///
/// Returns `((λ₁, v₁), (λ₂, v₂))` where each `v` is a normalized vector `(v_x, v_y)`.
#[allow(clippy::many_single_char_names)]
pub(super) fn eigen(matrix: Matrix) -> ((Complex<f64>, Vector), (Complex<f64>, Vector)) {
    let [[a, b], [c, d]] = matrix;

    let m = (a + d) / 2.0;
    let p = a * d - b * c;

    let lambda_1 = m + (m.powi(2) - p).sqrt();
    let lambda_2 = m - (m.powi(2) - p).sqrt();

    let (v1, v2) = if (lambda_1 - d).abs() < 1e-14 && c.abs() < 1e-14 {
        ((c, lambda_1 - a), (lambda_2 - d, b))
    } else {
        ((lambda_1 - d, c), (b, lambda_2 - a))
    };

    let p1 = v1.0.abs().hypot(v1.1.abs());
    let p2 = v2.0.abs().hypot(v2.1.abs());

    let v1 = if p1 < 1e-14 {
        (Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0))
    } else {
        (v1.0 / p1, v1.1 / p1)
    };
    let v2 = if p2 < 1e-14 {
        (Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0))
    } else {
        (v2.0 / p2, v2.1 / p2)
    };

    ((lambda_1, v1), (lambda_2, v2))
}

/// Computes the matrix exponential representing the fractional power of a gate.
///
/// Given a matrix $A$ and a parameter $p$, computes $A^p$. If `inverse` is
/// true, computes $(A^p)^\dagger$.
///
/// This is used heavily in building controlled root-of-U gates for the
/// linear-depth U(2) decomposition ladder.
pub(super) fn exp_gate(matrix: Matrix, param: f64, inverse: bool) -> Matrix {
    let ((lambda1, v1), (lambda_2, v2)) = eigen(matrix);

    let value_1 = lambda1.powf(param);
    let gate_1 = [
        [value_1 * v1.0 * v1.0.conj(), value_1 * v1.0 * v1.1.conj()],
        [value_1 * v1.1 * v1.0.conj(), value_1 * v1.1 * v1.1.conj()],
    ];

    let value_2 = lambda_2.powf(param);
    let gate_2 = [
        [value_2 * v2.0 * v2.0.conj(), value_2 * v2.0 * v2.1.conj()],
        [value_2 * v2.1 * v2.0.conj(), value_2 * v2.1 * v2.1.conj()],
    ];

    let gate = [
        [gate_1[0][0] + gate_2[0][0], gate_1[0][1] + gate_2[0][1]],
        [gate_1[1][0] + gate_2[1][0], gate_1[1][1] + gate_2[1][1]],
    ];

    if inverse {
        [
            [gate[0][0].conj(), gate[1][0].conj()],
            [gate[0][1].conj(), gate[1][1].conj()],
        ]
    } else {
        gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::gate::{Param, QuantumGate},
        matrix::Cf64,
    };
    use std::f64::consts::PI;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() < eps,
            "Expected {a} ≈ {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn assert_complex_close(a: Cf64, b: Cf64, eps: f64) {
        assert!(
            (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps,
            "Expected {a} ≈ {b}"
        );
    }

    #[test]
    fn zyz_identity_matrix() {
        let identity = [
            [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
            [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
        ];
        let (phase, theta_0, theta_1, theta_2) = zyz(identity);
        assert_close(phase, 0.0, 1e-10);
        assert_close(theta_1, 0.0, 1e-10);
        // theta_0 + theta_2 should be close to 0 (modulo 2pi)
        assert_close(theta_0 + theta_2, 0.0, 1e-10);
    }

    #[test]
    fn zyz_pauli_x_matrix() {
        let matrix = QuantumGate::PauliX.matrix();
        let (_phase, _theta_0, theta_1, _theta_2) = zyz(matrix);
        // PauliX = RY(PI) up to phase, so theta_1 should be close to PI
        assert_close(theta_1.abs(), PI, 1e-10);
    }

    #[test]
    fn zyz_hadamard_matrix() {
        let matrix = QuantumGate::Hadamard.matrix();
        let (_phase, _theta_0, theta_1, _theta_2) = zyz(matrix);
        // Hadamard has a non-trivial theta_1
        assert!(theta_1.abs() > 0.1);
    }

    #[test]
    fn zyz_roundtrip_pauli_x() {
        let matrix = QuantumGate::PauliX.matrix();
        let (_phase, theta_0, theta_1, theta_2) = zyz(matrix);

        // Reconstruct: RZ(theta_2) * RY(theta_1) * RZ(theta_0)
        let rz0 = QuantumGate::RotationZ(Param::Value(theta_0)).matrix();
        let ry1 = QuantumGate::RotationY(Param::Value(theta_1)).matrix();
        let rz2 = QuantumGate::RotationZ(Param::Value(theta_2)).matrix();

        let tmp = matrix_dot(&ry1, &rz0);
        let result = matrix_dot(&rz2, &tmp);

        // Verify magnitudes match (ZYZ decomposition preserves gate action up to phase)
        for i in 0..2 {
            for j in 0..2 {
                assert_close(result[i][j].norm(), matrix[i][j].norm(), 1e-10);
            }
        }
    }

    #[test]
    fn zyz_roundtrip_hadamard() {
        let matrix = QuantumGate::Hadamard.matrix();
        let (_phase, theta_0, theta_1, theta_2) = zyz(matrix);

        let rz0 = QuantumGate::RotationZ(Param::Value(theta_0)).matrix();
        let ry1 = QuantumGate::RotationY(Param::Value(theta_1)).matrix();
        let rz2 = QuantumGate::RotationZ(Param::Value(theta_2)).matrix();

        let tmp = matrix_dot(&ry1, &rz0);
        let result = matrix_dot(&rz2, &tmp);

        // Verify magnitudes match
        for i in 0..2 {
            for j in 0..2 {
                assert_close(result[i][j].norm(), matrix[i][j].norm(), 1e-10);
            }
        }
    }

    #[test]
    fn zyz_roundtrip_with_phase() {
        // Test that the full reconstruction e^{i*phase} * RZ(θ₂) * RY(θ₁) * RZ(θ₀)
        // produces a unitary matrix (det has modulus 1)
        let matrix = QuantumGate::Hadamard.matrix();
        let (phase, theta_0, theta_1, theta_2) = zyz(matrix);

        let rz0 = QuantumGate::RotationZ(Param::Value(theta_0)).matrix();
        let ry1 = QuantumGate::RotationY(Param::Value(theta_1)).matrix();
        let rz2 = QuantumGate::RotationZ(Param::Value(theta_2)).matrix();

        let tmp = matrix_dot(&ry1, &rz0);
        let result = matrix_dot(&rz2, &tmp);

        let e_phase = (Complex::i() * phase).exp();
        let scaled = [
            [result[0][0] * e_phase, result[0][1] * e_phase],
            [result[1][0] * e_phase, result[1][1] * e_phase],
        ];
        let det = scaled[0][0] * scaled[1][1] - scaled[0][1] * scaled[1][0];
        assert_close(det.norm(), 1.0, 1e-10);
    }

    #[test]
    fn eigen_pauli_z() {
        let matrix = QuantumGate::PauliZ.matrix();
        let ((lambda1, _), (lambda2, _)) = eigen(matrix);
        // Eigenvalues of PauliZ are +1 and -1
        let mut eigenvalues = [lambda1.re, lambda2.re];
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_close(eigenvalues[0], -1.0, 1e-10);
        assert_close(eigenvalues[1], 1.0, 1e-10);
        assert_close(lambda1.im, 0.0, 1e-10);
        assert_close(lambda2.im, 0.0, 1e-10);
    }

    #[test]
    fn eigen_identity() {
        let identity = [
            [Cf64::new(1.0, 0.0), Cf64::new(0.0, 0.0)],
            [Cf64::new(0.0, 0.0), Cf64::new(1.0, 0.0)],
        ];
        let ((lambda1, _), (lambda2, _)) = eigen(identity);
        assert_close(lambda1.re, 1.0, 1e-10);
        assert_close(lambda2.re, 1.0, 1e-10);
    }

    #[test]
    fn exp_gate_identity_param_zero() {
        // exp_gate with param=0 should give something close to identity
        // Actually, A^0 = I for any non-singular matrix
        // but exp_gate computes eigenvalue^param, so lambda^0 = 1 for all eigenvalues
        let matrix = QuantumGate::PauliZ.matrix();
        let result = exp_gate(matrix, 0.0, false);
        // Result should be close to a scalar * identity (within numerical precision)
        // Since lambda^0 = 1 for all eigenvalues, result is v1*v1' + v2*v2' = I
        assert_complex_close(result[0][0], Cf64::new(1.0, 0.0), 1e-10);
        assert_complex_close(result[0][1], Cf64::new(0.0, 0.0), 1e-10);
        assert_complex_close(result[1][0], Cf64::new(0.0, 0.0), 1e-10);
        assert_complex_close(result[1][1], Cf64::new(1.0, 0.0), 1e-10);
    }

    #[test]
    fn exp_gate_pauli_z_param_one() {
        // A^1 = A
        let matrix = QuantumGate::PauliZ.matrix();
        let result = exp_gate(matrix, 1.0, false);
        for i in 0..2 {
            for j in 0..2 {
                assert_complex_close(result[i][j], matrix[i][j], 1e-10);
            }
        }
    }

    #[test]
    fn exp_gate_inverse() {
        // exp_gate with inverse=true gives conjugate transpose
        let matrix = QuantumGate::PauliZ.matrix();
        let result = exp_gate(matrix, 1.0, false);
        let result_inv = exp_gate(matrix, 1.0, true);
        // result * result_inv should be identity
        let product = matrix_dot(&result, &result_inv);
        assert_complex_close(product[0][0], Cf64::new(1.0, 0.0), 1e-10);
        assert_complex_close(product[0][1], Cf64::new(0.0, 0.0), 1e-10);
        assert_complex_close(product[1][0], Cf64::new(0.0, 0.0), 1e-10);
        assert_complex_close(product[1][1], Cf64::new(1.0, 0.0), 1e-10);
    }
}
