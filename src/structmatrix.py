# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import math as ma
import matplotlib.pyplot as pl
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
force = (-480 * divingBoard.width * divingBoard.thickness * g) # Force acting on a slice of the diving board
b1 = [(force * h4) / (divingBoard.E * divingBoard.I)] * segments  # The b-vector
y = spsolve(A, b1)  # Solving the matrix

R = sp.ones(segments)
for i in range(1, 11):
    R[i - 1] = (((force / (24 * divingBoard.E * divingBoard.I)) * (i / 5) ** 2) * ((i / 5) ** 2 - 8 * (i / 5) + 24))

A = create_matrix(10)
b2 = A.dot(R)

temp = sp.ones(segments)
tempx = sp.ones(segments)
for i in range(0, segments):
    temp[i] = (abs(b2[i] - b1[i]))

# print("Feilforstørring: ", (np.linalg.norm(temp, 3) / np.linalg.norm(b2, 3)) / 2 ** -52)
# print("Cond(A): ", norm(A) * norm(inv(A)))
# print("Foroverfeil (4e): ", np.linalg.norm(R - y, 1)/2**-52, "maskinepsilon")

# cond5 = sp.ones(11)
# X5 = sp.ones(11)
# Y5 = sp.ones(11)
# Z5 = sp.ones(11)
# H5 = sp.ones(11)
# R5 = sp.ones(11)

for i in range(1, 12):
    segments = 10 * 2 ** i
    # X5[i-1] = segments
    segm_len = divingBoard.length / segments  # Length of each segment
    h4 = segm_len ** 4
    A = create_matrix(segments)
    if i < 9:
        cond = norm(A) * norm(inv(A))
    else:
        cond = 0
    b = [(force * h4)/(divingBoard.E * divingBoard.I)] * segments
    y = spsolve(A, b)
    # cond5[i-1] = cond * (2 ** -52)
    # R5[i - 1] = ((force * 4) / (24 * divingBoard.E * divingBoard.I)) * (4 - 16 + 24)
    # Y5[i-1] = y[-1]
    # Z5[i - 1] = abs(y[-1] - R5[i - 1])
    # H5[i-1] = 4/(segments**2)
    # print("Cond(A)", cond)
    # print(abs(y[-1] - R[-1]))

# pl.plot(np.log(X5), np.log(Z5))  # Oppgave 6c and 6d
# pl.plot(np.log(X5), np.log(cond5)) # Oppgave 6d
# pl.plot(np.log(X5), np.log(H5))  # Oppgave 6d
# pl.show()

X = sp.ones(11)
Y = sp.ones(11)
R1 = sp.ones(11)
Z = sp.ones(11)
Cond = sp.ones(11)
H2 = sp.ones(11)
for i in range(1, 12):
    segments = 10 * 2 ** i
    X[i-1] = segments
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    p = 100
    g = 9.81
    b = sp.ones(segments)
    for j in range(0, segments):
        b[j] = ((force - p * g * ma.sin((ma.pi * segm_len * j) / divingBoard.length))*h4) / (divingBoard.E * divingBoard.I)
    y = spsolve(A, b)
    if i < 9:
        Cond[i-1] = norm(A) * norm(inv(A)) * 2**-52
    else:
        Cond[i-1] = 0
    R1[i-1] = ((force*4)/(24*divingBoard.E*divingBoard.I))*(4-16+24)-((9.81*100*2) /
            (divingBoard.E*divingBoard.I*ma.pi))*(8/(ma.pi**3)*ma.sin(ma.pi)-8/6+8/2-8/(ma.pi**2))
    Z[i-1] = abs(y[-1] - R1[i-1])
    Y[i-1] = (y[-1])
    H2[i-1] = 4/(segments**2)
# pl.plot(X, Y)  # Oppgave 6b
# pl.plot(X, R1)  # Oppgave 6b
pl.plot(np.log(X), np.log(Z))  # Oppgave 6c and 6d
pl.plot(np.log(X), np.log(Cond)) # Oppgave 6d
pl.plot(np.log(X), np.log(H2))  # Oppgave 6d
pl.show()


for i in range(1, 2):
    segments = 10000
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    g = 9.81
    b = sp.ones(segments)
    segm_start = 0
    segm_stop = segm_len
    TOL = 0.00001
    f_pers = (g * 50) / 0.3
    for j in range(0, segments):
        if segm_stop > 1.70:
            if segm_start > 1.70:
                b[j] = ((force - f_pers) * h4) / (divingBoard.E * divingBoard.I)
            else:
                b[j] = ((force - (f_pers/(segm_stop - 1.7)) * h4) / (divingBoard.E * divingBoard.I))
        else:
            b[j] = ((force * h4) / (divingBoard.E * divingBoard.I))

        # if segm_stop < 1.70 + TOL and segm_stop > 1.70 - TOL:
        if 1.70 - TOL < segm_stop < 1.70 + TOL:
            b[j] = ((force * h4) / (divingBoard.E * divingBoard.I))
        if 1.70 - TOL < segm_start < 1.70 + TOL:
            b[j] = ((force - f_pers) * h4) / (divingBoard.E * divingBoard.I)
        segm_start += segm_len
        segm_stop += segm_len
    y = spsolve(A, b)
    # print(y[-1])
