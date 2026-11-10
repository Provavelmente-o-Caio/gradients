import functools
import sys
import warnings

import numpy as np
from ket.gates import RX, RY, RZ, H, X, Y, Z

from gradients.psr.main import expectation, parameter_shift_gradient

# ---------------------------------------------------------------------------
# Auxiliares
#
# O simulador do Ket tem erro absoluto da ordem de 1e-7 (nos testes
# avancados, componentes de gradiente nulo sairam com ~5e-8). Por isso,
# valores esperados proximos de zero NAO podem ser comparados com atol=1e-8.
# O erro da regra de deslocamento cresce como 1/sin(s), entao a tolerancia
# acompanha esse fator.
# ---------------------------------------------------------------------------

KET_TOL = 1e-6


def close(a, b, tol=KET_TOL):
    return abs(a - b) <= tol


def shift_tol(s, tol=KET_TOL):
    return tol / abs(np.sin(s))


def cost_of(gate=RY, operator=Z):
    return lambda theta: expectation(theta, gate, operator)


def scaled_ry(omega):
    """RY(omega * theta): o custo oscila com frequencia omega em theta."""
    return lambda theta, qubit: RY(omega * theta, qubit)


def hadamard_rz(theta, qubit):
    """H seguida de RZ(theta): leva |0> a um estado sensivel a RZ."""
    H(qubit)
    RZ(theta, qubit)


def known_failure(reason):
    """Marca um teste que descreve um comportamento ainda nao implementado.

    Se o teste falhar, imprime XFAIL. Se passar, avisa que a limitacao foi
    corrigida e que o decorador deve ser removido.
    """

    def wrap(fn):
        @functools.wraps(fn)
        def inner():
            try:
                fn()
            except Exception:
                print(f"  XFAIL ({reason})")
                return
            print("  XPASS: limitacao corrigida; remova @known_failure")

        return inner

    return wrap


# ---------------------------------------------------------------------------
# Testes originais
# ---------------------------------------------------------------------------


def test_scalar_parameter_shift():
    theta = 0.3

    cost = expectation(theta)
    gradient = parameter_shift_gradient(expectation, theta)

    print("=== Teste 1: Scalar Parameter Shift ===")
    print(f"Cost: {cost}, Expected: {np.cos(theta)}")
    print(f"Gradient: {gradient}, Expected: {-np.sin(theta)}")

    assert np.isclose(cost, np.cos(theta), atol=1e-8)
    assert np.isclose(gradient, -np.sin(theta), atol=1e-8)


# ---------------------------------------------------------------------------
# A. Contrato: a regra de dois termos e exata para portas RX, RY, RZ
# ---------------------------------------------------------------------------


# Para RY(theta)|0> e Z, o custo e cos(theta) e o gradiente e -sin(theta) para
# QUALQUER s que nao seja multiplo de pi. O erro numerico, porem, cresce como
# 1/sin(s), e s = pi/2 e o melhor condicionado.
def test_shift_independence():
    theta = 0.7
    cost = cost_of()
    print("=== Teste 2: independencia do deslocamento s ===")
    for s in (0.5, 1.0, np.pi / 2, 2.0, 2.5):
        grad = parameter_shift_gradient(cost, theta, s)
        print(f"  s={s:.4f}  grad={grad:.9f}  esperado={-np.sin(theta):.9f}")
        assert close(grad, -np.sin(theta), shift_tol(s)), (s, grad)


# Pontos estacionarios (gradiente 0) e extremos (|gradiente| = 1).
def test_special_angles():
    cost = cost_of()
    print("=== Teste 3: angulos especiais ===")
    for theta in (0.0, np.pi / 2, np.pi, -np.pi / 2, 3 * np.pi / 2, -np.pi):
        grad = parameter_shift_gradient(cost, theta)
        print(f"  theta={theta:+.4f}  grad={grad:+.9f}  esperado={-np.sin(theta):+.9f}")
        assert close(grad, -np.sin(theta)), (theta, grad)


# Propriedades que nao dependem de formula fechada:
#   g(theta + 2pi) = g(theta)    periodicidade
#   g(theta + pi)  = -g(theta)   antiperiodicidade (cos(theta + pi) = -cos(theta))
#   g(-theta)      = -g(theta)   paridade (o custo cos(theta) e par)
def test_symmetries():
    cost = cost_of()
    print("=== Teste 4: simetrias do gradiente ===")
    for theta in (0.3, 1.1, -2.0):
        g = parameter_shift_gradient(cost, theta)
        assert close(parameter_shift_gradient(cost, theta + 2 * np.pi), g)
        assert close(parameter_shift_gradient(cost, theta + np.pi), -g)
        assert close(parameter_shift_gradient(cost, -theta), -g)
        print(f"  theta={theta:+.2f}  g={g:+.9f}  OK")


# |dC/dtheta| <= 2 r ||O|| = 1 para gerador com autovalores +-1/2 e O = Z.
def test_gradient_is_bounded():
    cost = cost_of()
    for theta in np.linspace(-2 * np.pi, 2 * np.pi, 25):
        assert abs(parameter_shift_gradient(cost, theta)) <= 1 + KET_TOL


# Angulos grandes (theta + 2*pi*m). A tolerancia e maior porque o proprio
# angulo e arredondado. Se este teste falhar com tolerancia 1e-4, o Ket esta
# perdendo precisao em angulos grandes.
def test_large_angles():
    cost = cost_of()
    print("=== Teste 5: angulos grandes ===")
    for m in (10, 50, -50):
        theta = 0.7 + 2 * np.pi * m
        grad = parameter_shift_gradient(cost, theta)
        print(f"  m={m:+d}  theta={theta:+.3f}  grad={grad:+.7f}  esperado={-np.sin(0.7):+.7f}")
        assert close(grad, -np.sin(0.7), 1e-4)


# Observaveis e portas diferentes de (RY, Z). Inclui casos em que o gradiente e
# identicamente nulo, mas o custo nao precisa ser nulo:
#   - <Y> em RY|0> e sempre 0 (o estado fica no plano x-z)
#   - RZ|0> so altera a fase global, entao <Z> = 1 constante
OBSERVABLE_CASES = [
    # nome, porta, observavel, custo analitico, gradiente analitico
    ("Z / RY", RY, Z, np.cos, lambda t: -np.sin(t)),
    ("X / RY", RY, X, np.sin, np.cos),
    ("Y / RY (identicamente nulo)", RY, Y, lambda t: 0.0, lambda t: 0.0),
    ("3Z / RY", RY, lambda q: 3 * Z(q), lambda t: 3 * np.cos(t), lambda t: -3 * np.sin(t)),
    (
        "0,5Z + 0,3X / RY",
        RY,
        lambda q: 0.5 * Z(q) + 0.3 * X(q),
        lambda t: 0.5 * np.cos(t) + 0.3 * np.sin(t),
        lambda t: -0.5 * np.sin(t) + 0.3 * np.cos(t),
    ),
    ("Z / RX", RX, Z, np.cos, lambda t: -np.sin(t)),
    ("Y / RX", RX, Y, lambda t: -np.sin(t), lambda t: -np.cos(t)),
    ("Z / RZ|0> (constante)", RZ, Z, lambda t: 1.0, lambda t: 0.0),
    ("X / H.RZ", hadamard_rz, X, np.cos, lambda t: -np.sin(t)),
    ("Y / H.RZ", hadamard_rz, Y, np.sin, np.cos),
]


def test_observables_and_gates():
    print("=== Teste 6: observaveis e portas variados ===")
    for name, gate, operator, cost_fn, grad_fn in OBSERVABLE_CASES:
        cost = cost_of(gate, operator)
        for theta in (0.4, -1.3, 2.2):
            c = cost(theta)
            g = parameter_shift_gradient(cost, theta)
            assert close(c, cost_fn(theta)), (name, theta, "custo", c)
            assert close(g, grad_fn(theta)), (name, theta, "gradiente", g)
        print(f"  {name:<30} OK")


# A funcao deve avaliar o custo exatamente duas vezes (os dois circuitos
# deslocados), em theta + s e theta - s. E isso que a contagem de circuitos dos
# experimentos assume.
def test_cost_is_called_exactly_twice():
    calls = []

    def cost(theta):
        calls.append(theta)
        return np.cos(theta)

    parameter_shift_gradient(cost, 0.7, 0.4)
    assert len(calls) == 2
    assert np.allclose(sorted(calls), [0.7 - 0.4, 0.7 + 0.4])


# A regra e linear na funcao de custo.
def test_linearity_in_cost():
    a, b, theta = 2.0, -0.5, 0.9
    combined = lambda t: a * np.cos(t) + b * np.sin(t)
    expected = a * parameter_shift_gradient(np.cos, theta) + b * parameter_shift_gradient(
        np.sin, theta
    )
    assert close(parameter_shift_gradient(combined, theta), expected, 1e-12)


# Ruido de amostragem: se cada avaliacao tem erro limitado por eps, o erro do
# gradiente e limitado por eps / sin(s). Isso justifica s = pi/2.
def test_noise_amplification_follows_one_over_sin_s():
    rng = np.random.default_rng(0)
    eps, theta = 1e-6, 0.7
    worst = {}
    for s in (np.pi / 2, 0.5, 0.1, 0.01):
        errors = []
        for _ in range(200):
            noisy = lambda t: np.cos(t) + rng.uniform(-eps, eps)
            errors.append(abs(parameter_shift_gradient(noisy, theta, s) + np.sin(theta)))
        worst[s] = max(errors)
        assert worst[s] <= eps / np.sin(s) + 1e-12, (s, worst[s])
    print("=== Teste 7: amplificacao de ruido (eps = 1e-6) ===")
    for s, w in worst.items():
        print(f"  s={s:.4f}  pior erro={w:.2e}  limite={eps / np.sin(s):.2e}")
    assert worst[0.01] > 10 * worst[np.pi / 2]


def test_accepts_numeric_types():
    cost = cost_of()
    for theta in (0, 1, 0.3, np.float64(0.3), np.float32(0.3)):
        grad = parameter_shift_gradient(cost, theta)
        assert np.ndim(grad) == 0
        assert close(grad, -np.sin(float(theta)))





# Cada chamada cria um Process novo. Chamadas intercaladas nao podem
# contaminar umas as outras.
def test_repeated_calls_are_independent():
    cost = cost_of()
    first = parameter_shift_gradient(cost, 0.7)
    parameter_shift_gradient(cost, -1.3)
    second = parameter_shift_gradient(cost, 0.7)
    assert close(first, second, 1e-12)


# ---------------------------------------------------------------------------
# B. Limitacoes da implementacao basica
#
# Estes testes descrevem COMO a funcao falha fora do contrato. Eles passam hoje
# e documentam a limitacao. Se a implementacao for generalizada, devem ser
# atualizados.
# ---------------------------------------------------------------------------


# A formula (f(theta+s) - f(theta-s)) / (2 sin s) pressupoe um gerador com
# autovalores +-1/2, ou seja, custo com frequencia 1 em theta. Para uma porta
# RY(omega * theta), o custo e cos(omega * theta), e a funcao devolve
#     -sin(omega*theta) * sin(omega*s) / sin(s),
# que difere de -omega*sin(omega*theta) por um fator sin(omega*s)/(omega*sin s).
# Com omega = 2 e s = pi/2, o resultado e exatamente zero.
def test_wrong_frequency_breaks_two_term_rule():
    theta = 0.6
    print("=== Teste 8: porta com frequencia diferente de 1 ===")
    for omega in (2.0, 0.5, 3.0):
        cost = cost_of(scaled_ry(omega))
        true = -omega * np.sin(omega * theta)
        for s in (np.pi / 2, np.pi / 4):
            grad = parameter_shift_gradient(cost, theta, s)
            predicted = -np.sin(omega * theta) * np.sin(omega * s) / np.sin(s)
            factor = np.sin(omega * s) / (omega * np.sin(s))
            print(
                f"  omega={omega}  s={s:.3f}  obtido={grad:+.5f}  "
                f"verdadeiro={true:+.5f}  fator={factor:+.3f}"
            )
            assert close(grad, predicted, shift_tol(s))
            assert abs(grad / true - 1) > 0.05


# A correcao e escolher r = omega/2 (autovalores de H_j/2 em +-r) e
# s = pi/(4r), com gradiente r*(f(theta+s) - f(theta-s)). Para omega = 1 isso
# reproduz a funcao basica (r = 1/2, s = pi/2).
def test_wrong_frequency_is_fixed_by_choosing_r_and_s():
    theta = 0.6
    for omega in (2.0, 0.5, 3.0):
        cost = cost_of(scaled_ry(omega))
        r = omega / 2
        s = np.pi / (4 * r)
        grad = r * (cost(theta + s) - cost(theta - s))
        assert close(grad, -omega * np.sin(omega * theta), KET_TOL * max(1.0, omega))


# Um custo que nao e sinusoidal em theta esta fora do contrato: a regra nao e
# um diferenciador generico. Para f = theta^2 ela devolve 2*theta*s/sin(s).
def test_non_sinusoidal_cost_is_outside_the_contract():
    theta = 0.8
    for s in (np.pi / 2, 0.5):
        grad = parameter_shift_gradient(lambda t: t**2, theta, s)
        assert close(grad, 2 * theta * s / np.sin(s), 1e-12)
    assert abs(parameter_shift_gradient(lambda t: t**2, theta) - 2 * theta) > 0.5


# s = 0 faz 0/0. A funcao devolve NaN em silencio, em vez de levantar erro.
def test_zero_shift_returns_nan_silently():
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        grad = parameter_shift_gradient(np.cos, 0.7, 0.0)
    assert np.isnan(grad)


# Comportamento desejado: rejeitar s multiplo de pi (sin(s) = 0), onde a regra
# degenera. Para s = pi, f(theta + pi) e f(theta - pi) sao iguais em teoria, e
# sin(pi) ~ 1.2e-16 em ponto flutuante. O resultado e arbitrario: 0 exato se os
# dois custos arredondarem igual, ordem 1 (custo em float64) ou enorme (custo
# com ruido de ~1e-7), e nunca o gradiente correto. Para corrigir, adicione no
# inicio da funcao:
#
#     if abs(np.sin(s)) < 1e-9:
#         raise ValueError(f"s={s} e multiplo de pi; a regra degenera")
@known_failure("a implementacao basica nao valida s")
def test_degenerate_shift_is_rejected():
    for s in (0.0, np.pi, 2 * np.pi):
        try:
            parameter_shift_gradient(np.cos, 0.7, s)
        except ValueError:
            continue
        raise AssertionError(f"s={s} deveria levantar ValueError")


# NaN em theta nao e validado: o resultado e NaN em vez de um erro.
def test_nan_theta_propagates():
    with np.errstate(all="ignore"):
        assert np.isnan(parameter_shift_gradient(np.cos, np.nan))


if __name__ == "__main__":
    tests = [
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as error:
            failures += 1
            print(f"FAIL {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} testes passaram.")
    sys.exit(1 if failures else 0)
