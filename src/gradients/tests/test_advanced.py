import functools
import sys
import warnings
from contextlib import contextmanager

import numpy as np
from ket.gates import CNOT, RX, RY, RZ, X, Y, Z

import gradients.psr.advanced as advanced
from gradients.psr.advanced import (
    GateInstruction,
    expectation as psr_expectation,
    parameter_shift_gradient,
)
from gradients.utils.utils import shift_param

# ---------------------------------------------------------------------------
# Auxiliares
#
# O simulador do Ket tem erro absoluto da ordem de 1e-7 (componentes de
# gradiente nulo sairam com ~5e-8). Por isso valores esperados proximos de zero
# NAO sao comparados com atol=1e-8. O erro da regra de deslocamento cresce como
# 1/sin(s) e com a soma dos multiplicadores, e a tolerancia acompanha isso.
# ---------------------------------------------------------------------------

KET_TOL = 1e-6


def close(a, b, tol=KET_TOL):
    return abs(a - b) <= tol


def all_close(a, b, tol=KET_TOL):
    return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))


def shift_tol(s, scale=1.0, tol=KET_TOL):
    return scale * tol / abs(np.sin(s))


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


def show(title, cost=None, cost_exact=None, grad=None, grad_exact=None):
    print(f"=== {title} ===")
    if cost is not None:
        print(f"  cost Ket       = {cost:.12f}")
        print(f"  cost analitico = {cost_exact:.12f}")
    if grad is not None:
        print(f"  grad PSR       = {[float(g) for g in grad]}")
        print(f"  grad analitico = {[float(g) for g in grad_exact]}")


@contextmanager
def count_evaluations():
    """Conta quantas vezes parameter_shift_gradient avalia o custo."""
    original = advanced.expectation
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    advanced.expectation = counting
    try:
        yield calls
    finally:
        advanced.expectation = original


def z_on(index):
    return lambda qubits: Z(qubits[index])


def zz_on(first, second):
    return lambda qubits: Z(qubits[first]) * Z(qubits[second])


def cnot_gate(theta, control, target):
    CNOT(control, target)


def fixed_cnot(control, target):
    """CNOT como instrucao fixa (sem indice de parametro, nunca deslocada)."""
    return GateInstruction(gate=cnot_gate, target=(control, target), value=0.0)


def rx_pair(theta, qubits):
    """RX(theta) em dois qubits como UMA instrucao (gerador com 3 autovalores)."""
    RX(theta, qubits[0])
    RX(theta, qubits[1])


def pair_observable(qubits):
    return Z(qubits[0]) + Z(qubits[0]) * Z(qubits[1])


def two_frequency_rule():
    """Regra de deslocamento para frequencias {1, 2}: pares (coeficiente, s).

    Resolve 2 * sum_r c_r sin(k s_r) = k para k = 1, 2, com s = pi/4 e 3pi/4.
    """
    c1 = (1 + 1 / np.sqrt(2)) / 2
    c2 = (1 / np.sqrt(2) - 1) / 2
    return ((c1, np.pi / 4), (c2, 3 * np.pi / 4))


def cnot_circuit(params):
    return [
        GateInstruction(
            gate=lambda theta, control, target: (
                RY(theta, control),
                CNOT(control, target),
            ),
            target=(0, 1),
            value=params[0],
            index=0,
        ),
        GateInstruction(RY, 0, params[1], 1),
    ]


def rx_ry_circuit(params):
    return [
        GateInstruction(RX, 0, params[0], 0),
        GateInstruction(RY, 0, params[1], 1),
    ]


def three_qubit_circuit(params):
    return [
        GateInstruction(
            gate=lambda theta, q0, q1: (
                RY(theta, q0),
                CNOT(q0, q1),
            ),
            target=(0, 1),
            value=params[0],
            index=0,
        ),
        GateInstruction(
            gate=lambda theta, q1, q2: CNOT(q1, q2),
            target=(1, 2),
            value=0.0,
        ),
        GateInstruction(RY, 1, params[1], 1),
        GateInstruction(RY, 2, params[2], 2),
    ]


def two_qubit_ansatz(params):
    return [
        GateInstruction(RY, 0, params[0], 0),
        GateInstruction(RY, 1, params[1], 1),
        fixed_cnot(0, 1),
        GateInstruction(RY, 0, params[2], 2),
        GateInstruction(RY, 1, params[3], 3),
    ]


# ---------------------------------------------------------------------------
# A. Contrato: a regra e exata para portas RX, RY, RZ com parametros
#    compartilhados, portas fixas e observaveis arbitrarios
# ---------------------------------------------------------------------------


# Circuito:  RY(t0) -> q0,  CNOT(0,1),  RY(t1) -> q0
# Observavel: Z qubit 1
# Analitico:  <Z1> = cos(t0)   (CNOT leva Z1 em Z0 Z1)
# Gradientes: -sin(t0), 0
def test_multi_qubit_cnot():
    params = [0.5, 0.8]
    circuit = cnot_circuit(params)
    operator = z_on(1)

    cost = psr_expectation(2, params, circuit, operator)
    cost_exact = np.cos(params[0])
    grad = parameter_shift_gradient(2, circuit, params, operator=operator)
    grad_exact = [-np.sin(params[0]), 0.0]

    show("Teste 1: Multi-qubit CNOT", cost, cost_exact, grad, grad_exact)
    assert close(cost, cost_exact)
    assert all_close(grad, grad_exact)


# Circuito:  1 qubit,  RY(0.5*t0) + RY(1.5*t0), mesmo indice, multiplicadores
# diferentes.
# Analitico:  <Z> = cos(2 t0),  grad = -2 sin(2 t0)
def test_shared_parameters():
    theta0 = 0.7
    parameters = [theta0]
    circuit = [
        GateInstruction(gate=RY, target=0, value=0.5 * theta0, index=0, multiplier=0.5),
        GateInstruction(gate=RY, target=0, value=1.5 * theta0, index=0, multiplier=1.5),
    ]

    cost = psr_expectation(num_qubits=1, parameters=parameters, gates=circuit, operator=Z)
    cost_exact = np.cos(2 * theta0)
    grad = parameter_shift_gradient(num_qubits=1, gates=circuit, parameters=parameters)
    grad_exact = [-2 * np.sin(2 * theta0)]

    show("Teste 2: Parametros compartilhados", cost, cost_exact, grad, grad_exact)
    assert close(cost, cost_exact)
    assert all_close(grad, grad_exact)


# Circuito:  1 qubit,  RX(t0) + RY(t1);  Observavel: Z
# Analitico:  <Z> = cos(t0) cos(t1)
# Gradientes: -sin(t0) cos(t1),  -cos(t0) sin(t1)
def test_rx_ry_gates():
    params = [0.4, 1.1]
    circuit = rx_ry_circuit(params)

    cost = psr_expectation(1, params, circuit, operator=Z)
    cost_exact = np.cos(params[0]) * np.cos(params[1])
    grad = parameter_shift_gradient(1, circuit, params, operator=Z)
    grad_exact = [
        -np.sin(params[0]) * np.cos(params[1]),
        -np.cos(params[0]) * np.sin(params[1]),
    ]

    show("Teste 3: RX + RY gates", cost, cost_exact, grad, grad_exact)
    assert close(cost, cost_exact)
    assert all_close(grad, grad_exact)


# Circuito:  RY(t0)->q0, RY(t1)->q1, RY(t2)->q2, CNOT(q0,q1), CNOT(q1,q2)
# Observavel: Z qubit 0
# Analitico:  <Z0> = cos(t0)
# Gradientes: -sin(t0), 0, 0
def test_3_qubits_cnot():
    params = [0.6, 1.1, 0.3]
    circuit = three_qubit_circuit(params)
    operator = z_on(0)

    cost = psr_expectation(3, params, circuit, operator=operator)
    cost_exact = np.cos(params[0])
    grad = parameter_shift_gradient(3, circuit, params, operator=operator)
    grad_exact = [-np.sin(params[0]), 0.0, 0.0]

    show("Teste 4: 3 qubits CNOT", cost, cost_exact, grad, grad_exact)
    assert close(cost, cost_exact)
    assert all_close(grad, grad_exact)


# O parametro 1 nao aparece em nenhuma instrucao. O resultado deve ter 3
# componentes, com 0 na posicao 1 (alinhamento da lista de gradientes).
def test_unused_parameter():
    params = [0.5, 0.9, 0.3]
    circuit = [
        GateInstruction(gate=RY, target=0, value=params[0], index=0),
        GateInstruction(gate=RY, target=0, value=params[2], index=2),
    ]

    cost = psr_expectation(1, params, circuit, operator=Z)
    cost_exact = np.cos(params[0] + params[2])
    grad = parameter_shift_gradient(1, circuit, params)
    expected = -np.sin(params[0] + params[2])

    show("Teste 5: Parametro nao utilizado", cost, cost_exact, grad, [expected, 0.0, expected])
    assert close(cost, cost_exact)
    assert len(grad) == 3
    assert all_close(grad, [expected, 0.0, expected])


# Circuito: RY(t) em q0 e q1;  Observavel: Z0 Z1
# Analitico: <Z0 Z1> = cos^2(t),  gradiente = -sin(2t)
# O custo e quadratico no parametro: a soma sobre as portas reproduz a regra do
# produto.
def test_shared_parameter_two_qubits():
    theta = 0.7
    circuit = [
        GateInstruction(gate=RY, target=0, value=theta, index=0),
        GateInstruction(gate=RY, target=1, value=theta, index=0),
    ]
    operator = zz_on(0, 1)

    grad = parameter_shift_gradient(2, circuit, [theta], operator=operator)
    cost = psr_expectation(2, [theta], circuit, operator=operator)
    cost_exact = np.cos(theta) ** 2
    expected = -np.sin(2 * theta)

    show("Teste 6: Parametro compartilhado em dois qubits", cost, cost_exact, grad, [expected])
    assert close(cost, cost_exact)
    assert close(grad[0], expected)


# Circuito: RY(t) em q0;  Observavel: 0.5 Z0 + 0.3 X0
# Analitico: C = 0.5 cos(t) + 0.3 sin(t)
def test_multi_term_observable():
    theta = 0.7
    circuit = [GateInstruction(gate=RY, target=0, value=theta, index=0)]
    operator = lambda qubits: 0.5 * Z(qubits[0]) + 0.3 * X(qubits[0])

    grad = parameter_shift_gradient(1, circuit, [theta], operator=operator)
    cost = psr_expectation(1, [theta], circuit, operator=operator)
    cost_exact = 0.5 * np.cos(theta) + 0.3 * np.sin(theta)
    expected = -0.5 * np.sin(theta) + 0.3 * np.cos(theta)

    show("Teste 7: Observavel com multiplos termos", cost, cost_exact, grad, [expected])
    assert close(cost, cost_exact)
    assert close(grad[0], expected)


# Circuito: RX(t) em q0;  Observavel: Y0
# Analitico: C = -sin(t),  gradiente = -cos(t)
def test_y_observable():
    theta = 0.7
    circuit = [GateInstruction(gate=RX, target=0, value=theta, index=0)]
    operator = lambda qubits: Y(qubits[0])

    grad = parameter_shift_gradient(1, circuit, [theta], operator=operator)
    cost = psr_expectation(1, [theta], circuit, operator=operator)
    cost_exact = -np.sin(theta)
    expected = -np.cos(theta)

    show("Teste 8: Observavel Y", cost, cost_exact, grad, [expected])
    assert close(cost, cost_exact)
    assert close(grad[0], expected)


# Ansatz de dois qubits com emaranhamento em que os QUATRO gradientes sao nao
# nulos (e o mesmo circuito dos testes do Hadamard Test):
#   RY(t0) q0, RY(t1) q1, CNOT(0,1), RY(t2) q0, RY(t3) q1;  Observavel: Z0 Z1
# Propagando Z0 Z1 de tras para frente (RY^dagger Z RY = cos Z - sin X) e
# conjugando pela CNOT, obtem-se
#   C = c2 c3 c1 - c2 s3 c0 s1 + s2 s3 s0     (ci = cos ti, si = sin ti)
def test_two_qubit_ansatz_all_gradients_nonzero():
    t = [0.3, 1.1, -0.4, 0.8]
    c, s = np.cos(t), np.sin(t)
    circuit = two_qubit_ansatz(t)
    operator = zz_on(0, 1)

    cost = psr_expectation(2, t, circuit, operator=operator)
    cost_exact = c[2] * c[3] * c[1] - c[2] * s[3] * c[0] * s[1] + s[2] * s[3] * s[0]
    grad = parameter_shift_gradient(2, circuit, t, operator=operator)
    grad_exact = [
        c[2] * s[3] * s[0] * s[1] + s[2] * s[3] * c[0],
        -c[2] * c[3] * s[1] - c[2] * s[3] * c[0] * c[1],
        -s[2] * c[3] * c[1] + s[2] * s[3] * c[0] * s[1] + c[2] * s[3] * s[0],
        -c[2] * s[3] * c[1] - c[2] * c[3] * c[0] * s[1] + s[2] * c[3] * s[0],
    ]

    show("Teste 9: Ansatz de dois qubits (4 gradientes nao nulos)", cost, cost_exact, grad, grad_exact)
    assert close(cost, cost_exact)
    assert all_close(grad, grad_exact)
    assert all(abs(g) > 0.05 for g in grad_exact)


# RX e RY nao comutam. Com observavel Y:
#   RX(a) e depois RY(b):  <Y> = -sin(a)          grad = (-cos a, 0)
#   RY(b) e depois RX(a):  <Y> = -cos(b) sin(a)   grad = (-cos b cos a, sin b sin a)
# Os parametros sao (a, b) nos dois casos, com a em RX (indice 0) e b em RY
# (indice 1).
def test_gate_order_matters():
    a, b = 0.5, 0.9
    operator = lambda qubits: Y(qubits[0])
    rx = GateInstruction(RX, 0, a, 0)
    ry = GateInstruction(RY, 0, b, 1)

    cases = [
        ("RX depois RY", [rx, ry], -np.sin(a), [-np.cos(a), 0.0]),
        (
            "RY depois RX",
            [ry, rx],
            -np.cos(b) * np.sin(a),
            [-np.cos(b) * np.cos(a), np.sin(b) * np.sin(a)],
        ),
    ]
    print("=== Teste 10: Ordem das portas ===")
    for name, circuit, cost_exact, grad_exact in cases:
        cost = psr_expectation(1, [a, b], circuit, operator=operator)
        grad = parameter_shift_gradient(1, circuit, [a, b], operator=operator)
        print(f"  {name}: grad PSR = {[float(g) for g in grad]}  analitico = {grad_exact}")
        assert close(cost, cost_exact), name
        assert all_close(grad, grad_exact), name


# Uma porta fixa (index=None) com angulo diferente de zero deve entrar no estado
# mas nunca ser deslocada: C = cos(t + 0,4), gradiente = -sin(t + 0,4). O mesmo
# vale para uma instrucao com multiplicador 0, que e ignorada na soma.
def test_fixed_gate_with_nonzero_angle():
    theta = 0.7
    print("=== Teste 11: Porta fixa com angulo nao nulo ===")
    cases = [
        ("index=None", [GateInstruction(RY, 0, 0.4), GateInstruction(RY, 0, theta, 0)]),
        (
            "multiplicador 0",
            [GateInstruction(RY, 0, theta, 0), GateInstruction(RY, 0, 0.4, 0, 0.0)],
        ),
    ]
    for name, circuit in cases:
        cost = psr_expectation(1, [theta], circuit, operator=Z)
        grad = parameter_shift_gradient(1, circuit, [theta])
        expected = -np.sin(theta + 0.4)
        print(f"  {name}: grad PSR = {[float(g) for g in grad]}  analitico = {[expected]}")
        assert close(cost, np.cos(theta + 0.4)), name
        assert len(grad) == 1 and close(grad[0], expected), name


# Multiplicador negativo: RY(t) com multiplicador 1 e RY(-2t) com multiplicador
# -2. O custo e cos(t - 2t) = cos(t) e o gradiente e -sin(t).
def test_negative_multiplier():
    theta = 0.7
    circuit = [
        GateInstruction(RY, 0, theta, 0, 1.0),
        GateInstruction(RY, 0, -2 * theta, 0, -2.0),
    ]
    cost = psr_expectation(1, [theta], circuit, operator=Z)
    grad = parameter_shift_gradient(1, circuit, [theta])
    print("=== Teste 12: Multiplicador negativo ===")
    print(f"  grad PSR = {[float(g) for g in grad]}  analitico = {[-np.sin(theta)]}")
    assert close(cost, np.cos(theta))
    assert close(grad[0], -np.sin(theta))


# Parametros nao utilizados em qualquer posicao. Os parametros usados entram em
# portas RY no mesmo qubit, entao C = cos(soma dos valores usados) e o gradiente
# de cada parametro usado e -sin(soma). Os demais devem valer 0 na posicao certa.
def test_unused_parameters_keep_alignment():
    params = [0.2, 0.5, 0.9, 0.4]
    print("=== Teste 13: Parametros nao utilizados em varias posicoes ===")
    for used in ([1], [0, 3], [0], [3], [], [0, 1, 2, 3]):
        circuit = [GateInstruction(RY, 0, params[i], i) for i in used]
        total = sum(params[i] for i in used)
        expected = [-np.sin(total) if i in used else 0.0 for i in range(4)]
        grad = parameter_shift_gradient(1, circuit, params)
        print(f"  usados={used}: grad PSR = {[round(float(g), 6) for g in grad]}")
        assert len(grad) == 4, used
        assert all_close(grad, expected), used


# Casos vazios: sem portas, o gradiente de cada parametro e 0. Sem parametros, o
# resultado e uma lista vazia.
def test_empty_circuit_and_parameters():
    assert all_close(parameter_shift_gradient(1, [], [0.3, 0.7]), [0.0, 0.0])
    assert parameter_shift_gradient(1, [GateInstruction(RY, 0, 0.3, 0)], []) == []


# A ordem da tupla de target e a ordem dos argumentos da porta. RY(t) em q1 e
# CNOT(controle=q1, alvo=q0): <Z0> = cos(t), gradiente -sin(t). Com CNOT(q0, q1)
# o controle fica em |0> e nada acontece: <Z0> = 1 e gradiente 0.
def test_tuple_target_order():
    theta = 0.7
    print("=== Teste 14: Ordem dos qubits em target ===")
    cases = [
        ("target=(1, 0)", (1, 0), np.cos(theta), -np.sin(theta)),
        ("target=(0, 1)", (0, 1), 1.0, 0.0),
    ]
    for name, target, cost_exact, grad_exact in cases:
        circuit = [
            GateInstruction(RY, 1, theta, 0),
            GateInstruction(gate=cnot_gate, target=target, value=0.0),
        ]
        cost = psr_expectation(2, [theta], circuit, operator=z_on(0))
        grad = parameter_shift_gradient(2, circuit, [theta], operator=z_on(0))
        print(f"  {name}: grad PSR = {[float(g) for g in grad]}  analitico = {[grad_exact]}")
        assert close(cost, cost_exact), name
        assert close(grad[0], grad_exact), name


# Formas equivalentes de descrever uma porta de 1 qubit em registrador de 1
# qubit: target inteiro, tupla de um elemento e None (porta no registrador).
def test_equivalent_target_forms():
    theta = 0.7
    results = []
    for target in (0, (0,), None):
        circuit = [GateInstruction(RY, target, theta, 0)]
        results.append(parameter_shift_gradient(1, circuit, [theta], operator=z_on(0))[0])
    print("=== Teste 15: Formas equivalentes de target ===")
    print(f"  int, tupla, None: {results}  analitico = {-np.sin(theta)}")
    assert all(close(g, -np.sin(theta)) for g in results)


# Independencia do deslocamento com parametro compartilhado: o erro soma as
# contribuicoes das duas portas (soma dos multiplicadores = 2), e s negativo
# deve dar o mesmo resultado.
def test_shift_independence_with_shared_parameter():
    theta0 = 0.7
    circuit = [
        GateInstruction(RY, 0, 0.5 * theta0, 0, 0.5),
        GateInstruction(RY, 0, 1.5 * theta0, 0, 1.5),
    ]
    expected = -2 * np.sin(2 * theta0)
    print("=== Teste 16: Independencia do deslocamento s ===")
    for s in (0.5, 1.0, np.pi / 2, 2.0, 2.5, -1.0):
        grad = parameter_shift_gradient(1, circuit, [theta0], shift=s)
        print(f"  s={s:+.4f}  grad={grad[0]:.9f}  esperado={expected:.9f}")
        assert close(grad[0], expected, shift_tol(s, scale=2.0)), s


# RZ aplicada a |0> so altera a fase global: <Z> = 1 constante, gradiente 0.
def test_constant_cost_has_zero_gradient():
    theta = 0.7
    circuit = [GateInstruction(RZ, 0, theta, 0)]
    assert close(psr_expectation(1, [theta], circuit, operator=Z), 1.0)
    assert all_close(parameter_shift_gradient(1, circuit, [theta]), [0.0])


# Numero de avaliacoes do custo (circuitos executados). E a contagem que os
# experimentos de eficiencia assumem: 2 por par da regra, por instrucao que usa o
# parametro. Portas fixas, parametros nao utilizados e multiplicadores nulos nao
# contam.
def test_number_of_evaluations():
    theta = 0.7
    rule = two_frequency_rule()
    cases = [
        ("uma porta", [GateInstruction(RY, 0, theta, 0)], [theta], 1, 2),
        (
            "parametro compartilhado em duas portas",
            [
                GateInstruction(RY, 0, 0.5 * theta, 0, 0.5),
                GateInstruction(RY, 0, 1.5 * theta, 0, 1.5),
            ],
            [theta],
            1,
            4,
        ),
        ("parametro nao utilizado", [GateInstruction(RY, 0, theta, 0)], [theta, 0.1], 1, 2),
        (
            "multiplicador nulo",
            [GateInstruction(RY, 0, theta, 0), GateInstruction(RY, 0, 0.5, 0, 0.0)],
            [theta],
            1,
            2,
        ),
        (
            "porta fixa nao conta",
            [GateInstruction(RY, 0, 0.4), GateInstruction(RY, 0, theta, 0)],
            [theta],
            1,
            2,
        ),
        (
            "regra com dois pares",
            [GateInstruction(rx_pair, None, theta, 0, shift_rule=rule)],
            [theta],
            2,
            4,
        ),
        ("circuito vazio", [], [theta], 1, 0),
    ]
    print("=== Teste 17: Numero de avaliacoes ===")
    for name, circuit, params, num_qubits, expected in cases:
        with count_evaluations() as calls:
            parameter_shift_gradient(num_qubits, circuit, params, operator=z_on(0))
        print(f"  {name:<40} {len(calls)} (esperado {expected})")
        assert len(calls) == expected, name


# O calculo do gradiente nao pode alterar o circuito recebido, e shift_param deve
# devolver uma copia com apenas uma instrucao deslocada.
def test_circuit_is_not_modified():
    params = [0.3, 0.8]
    circuit = rx_ry_circuit(params)
    before = [(g.value, g.index, g.multiplier) for g in circuit]
    parameter_shift_gradient(1, circuit, params)
    assert [(g.value, g.index, g.multiplier) for g in circuit] == before

    shifted = shift_param(circuit, 1, 0.25)
    assert shifted[0].value == circuit[0].value
    assert close(shifted[1].value, circuit[1].value + 0.25, 1e-12)
    assert [(g.value, g.index, g.multiplier) for g in circuit] == before


# Chamadas intercaladas nao podem contaminar umas as outras.
def test_repeated_calls_are_independent():
    params = [0.3, 0.8]
    circuit = rx_ry_circuit(params)
    first = parameter_shift_gradient(1, circuit, params)
    parameter_shift_gradient(1, rx_ry_circuit([1.3, -0.4]), [1.3, -0.4])
    second = parameter_shift_gradient(1, circuit, params)
    assert all_close(first, second, 1e-12)


# Instrucao de DOIS qubits com gerador de 3 autovalores: RX(t) em q0 e q1 como
# uma unica instrucao tem gerador (X0 + X1)/2, com autovalores {-1, 0, 0, 1}. O
# custo de Z0 + Z0 Z1 e cos(t) + cos^2(t), com frequencias 1 e 2, e a regra de
# dois termos nao basta. Com a regra de duas frequencias, o resultado e exato:
#   C = cos(t) + cos^2(t),  gradiente = -sin(t) - sin(2t)
def test_custom_rule_two_frequencies():
    theta = 0.7
    circuit = [
        GateInstruction(rx_pair, None, theta, 0, shift_rule=two_frequency_rule())
    ]
    cost = psr_expectation(2, [theta], circuit, operator=pair_observable)
    grad = parameter_shift_gradient(2, circuit, [theta], operator=pair_observable)
    expected = -np.sin(theta) - np.sin(2 * theta)

    show(
        "Teste 18: Regra customizada com duas frequencias",
        cost,
        np.cos(theta) + np.cos(theta) ** 2,
        grad,
        [expected],
    )
    assert close(cost, np.cos(theta) + np.cos(theta) ** 2)
    assert close(grad[0], expected)


# Com shift_rule definido, o argumento shift nao e usado, nem para s = 0.
def test_custom_rule_ignores_shift():
    theta = 0.7
    circuit = [
        GateInstruction(rx_pair, None, theta, 0, shift_rule=two_frequency_rule())
    ]
    expected = -np.sin(theta) - np.sin(2 * theta)
    print("=== Teste 19: shift_rule ignora shift ===")
    for s in (0.0, 0.3, np.pi / 2):
        grad = parameter_shift_gradient(
            2, circuit, [theta], shift=s, operator=pair_observable
        )
        print(f"  shift={s:.4f}  grad={grad[0]:.9f}  esperado={expected:.9f}")
        assert close(grad[0], expected), s


# spectral_shift_rule recebe os autovalores do gerador G de e^{-i t G/2} (a mesma
# convencao dos testes de QAOA, em que se passam os autovalores de 2H para a
# evolucao e^{-i t H}). Para RY, G = Y, com autovalores {-1, 1}. Para RX em dois
# qubits, G = X0 + X1, com autovalores {-2, 0, 0, 2}.
def test_spectral_shift_rule_matches_closed_form():
    from gradients.utils.utils import spectral_shift_rule

    theta = 0.7
    single = [GateInstruction(RY, 0, theta, 0, shift_rule=spectral_shift_rule([-1, 1]))]
    grad_single = parameter_shift_gradient(1, single, [theta], operator=z_on(0))

    pair = [
        GateInstruction(
            rx_pair, None, theta, 0, shift_rule=spectral_shift_rule([-2, 0, 0, 2])
        )
    ]
    grad_pair = parameter_shift_gradient(2, pair, [theta], operator=pair_observable)

    print("=== Teste 20: spectral_shift_rule ===")
    print(f"  RY:      grad={grad_single[0]:.9f}  esperado={-np.sin(theta):.9f}")
    print(
        f"  RX x RX: grad={grad_pair[0]:.9f}  "
        f"esperado={-np.sin(theta) - np.sin(2 * theta):.9f}"
    )
    assert close(grad_single[0], -np.sin(theta))
    assert close(grad_pair[0], -np.sin(theta) - np.sin(2 * theta))


# ---------------------------------------------------------------------------
# B. Limitacoes da implementacao avancada
#
# Estes testes descrevem COMO a funcao se comporta fora do contrato. Passam hoje e
# documentam a limitacao. Os marcados com @known_failure descrevem um
# comportamento desejado que ainda nao existe.
# ---------------------------------------------------------------------------


# Sem shift_rule, a regra padrao tem dois termos e so captura a frequencia 1. Para
# a instrucao RX(t) x RX(t), o resultado e -sin(t) em vez de -sin(t) - sin(2t).
# Com observavel Z0 Z1 (so frequencia 2), o resultado e exatamente zero.
def test_default_rule_fails_for_multi_frequency_instruction():
    theta = 0.7
    circuit = [GateInstruction(rx_pair, None, theta, 0)]

    grad = parameter_shift_gradient(2, circuit, [theta], operator=pair_observable)
    grad_zz = parameter_shift_gradient(2, circuit, [theta], operator=zz_on(0, 1))
    exact = -np.sin(theta) - np.sin(2 * theta)

    print("=== Teste 21: regra padrao em instrucao com 2 frequencias ===")
    print(f"  Z0 + Z0Z1: obtido={grad[0]:+.6f}  verdadeiro={exact:+.6f}")
    print(f"  Z0Z1:      obtido={grad_zz[0]:+.6f}  verdadeiro={-np.sin(2 * theta):+.6f}")
    assert close(grad[0], -np.sin(theta))
    assert abs(grad[0] - exact) > 0.5
    assert close(grad_zz[0], 0.0)


# Uma instrucao com indice fora do intervalo de parametros e descartada em
# silencio: o gradiente tem o tamanho de parameters e nenhum valor para essa porta.
def test_out_of_range_index_is_dropped_silently():
    circuit = [GateInstruction(RY, 0, 0.7, 2)]
    grad = parameter_shift_gradient(1, circuit, [0.1, 0.2])
    assert len(grad) == 2
    assert all_close(grad, [0.0, 0.0], 1e-12)


@known_failure("indice fora do intervalo nao e validado")
def test_out_of_range_index_is_rejected():
    circuit = [GateInstruction(RY, 0, 0.7, 2)]
    try:
        parameter_shift_gradient(1, circuit, [0.1, 0.2])
    except (ValueError, IndexError):
        return
    raise AssertionError("indice 2 com 2 parametros deveria levantar erro")


# s = 0 faz coeficiente = 1 / (2 sin 0) = inf e inf * 0 = NaN. A funcao devolve
# NaN em silencio.
def test_zero_shift_returns_nan_silently():
    circuit = [GateInstruction(RY, 0, 0.7, 0)]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        grad = parameter_shift_gradient(1, circuit, [0.7], shift=0.0)
    assert np.isnan(grad[0])


# Comportamento desejado: rejeitar s multiplo de pi quando a regra padrao e usada
# (com shift_rule definido, shift nao e usado). Para corrigir, adicione antes do
# laco, quando alguma instrucao nao tiver shift_rule:
#
#     if abs(np.sin(shift)) < 1e-9:
#         raise ValueError(f"shift={shift} e multiplo de pi; a regra degenera")
@known_failure("a implementacao avancada nao valida shift")
def test_degenerate_shift_is_rejected():
    circuit = [GateInstruction(RY, 0, 0.7, 0)]
    for s in (0.0, np.pi, 2 * np.pi):
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                parameter_shift_gradient(1, circuit, [0.7], shift=s)
        except ValueError:
            continue
        raise AssertionError(f"shift={s} deveria levantar ValueError")


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
