import numpy as np
from ket.base import Process
from ket.gates import H, obs, RY, Y
from ket.operations import ExpValue


# Decomposes a 2x2 hermitian matrix into a combination of Pauli Words
def decomposition(matrix):
    a = matrix[0][0]
    b = matrix[0][1]
    c = matrix[1][0]
    d = matrix[1][1]

    cI = (a + d) / 2
    cZ = (a - d) / 2
    cX = (c + b) / 2
    cY = (c - b) * -1j / 2
    return {"I": cI, "Z": cZ, "X": cX, "Y": cY}


def hadamard_test(num_qubits: int):
    process = Process()

    # Preparação do ancilla
    ancilla = process.qalloc(1)
    ancilla.H()

    qubits = process.qalloc(num_qubits)
