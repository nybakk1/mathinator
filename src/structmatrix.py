# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import math as ma
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import inv
from entities.DivingBoard import DivingBoard


def create_matrix(n):
    """
    Create a CSR matrix.
    :param n: Width and height of CSR matrix.
    :return: A CSR matrix
    """
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e, -4*e, e], [-2, -1, 0, 1, 2], n, n)
    A = lil_matrix(A)
    B = csr_matrix([[ 16,       - 9,           8 /  3,  - 1 /  4],
                    [ 16 / 17,  -60 / 17,     72 / 17,  -28 / 17],
                    [-12 / 17,   96 / 17,   -156 / 17,   72 / 17]
                    ])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]

    return A.tocsc()


# Constants
g = 9.81        # Gravitational force in m/s^2.
segments = 10   # Number of segments

divingBoard = DivingBoard(2.00, 0.30, 0.03)
A = create_matrix(segments)     # Creating the A matrix
segm_len = divingBoard.length / segments    # Length of each segment
h4 = segm_len ** 4
force = (-480 * divingBoard.width * divingBoard.thickness * g * divingBoard.length) / segments  # Force acting on a slice of the diving board
b1 = [(force * h4) / (divingBoard.E * divingBoard.I)] * segments  # The b-vector
y = spsolve(A, b1)  # Solving the matrix

R = sp.ones(segments)
for i in range(1, 11):
    R[i - 1] = (force / (24 * divingBoard.E * divingBoard.I) * ((i / 5) ** 2) * ((i / 5) ** 2 - 8 * (i / 5) + 24))

A = create_matrix(10)
b2 = A.dot(R)

temp = sp.ones(segments)
tempx = sp.ones(segments)
for i in range(0, segments):
    temp[i] = (abs(b2[i] - b1[i]))

# print("Feilforstørring: ", (np.linalg.norm(temp, 3) / np.linalg.norm(b2, 3)) / 2 ** -52)
# print("Cond(A): ", norm(A) * norm(inv(A)))
# print("Foroverfeil (4e): ", np.linalg.norm(R - y, 1)/2**-52, "maskinepsilon")

for i in range(1, 12):
    segments = 10 * 2 ** i
    segm_len = divingBoard.length / segments  # Length of each segment
    h4 = segm_len ** 4
    A = create_matrix(segments)
    # cond = norm(A) * norm(inv(A))
    b = [(force * h4)/(divingBoard.E * divingBoard.I)] * segments
    y = spsolve(A, b)
    # print("Cond(A)", cond)
    # print(abs(y[-1] - R[-1]))
    # print(cond)
    # print(abs(y[-1] - R[-1]))

for i in range(1, 12):
    segments = 10 * 2 ** i
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    p = 100
    g = 9.81
    b = sp.ones(segments)
    for j in range(0, segments):
        b[j] = ((force - p * g * ma.sin((ma.pi * segm_len * j)/ divingBoard.length))*h4) / (divingBoard.E * divingBoard.I)
    y = spsolve(A, b)
    R1 = ((force*4)/24*divingBoard.E*divingBoard.I)*(4-16+24)-((9.81*100*2)/(divingBoard.E*divingBoard.I*ma.pi))*(8/(ma.pi**3)*ma.sin(ma.pi)-8/6+8/2-8/(ma.pi**2))
    print(abs(y[-1] - R1))

