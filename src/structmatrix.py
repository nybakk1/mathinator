# -*- coding: utf-8 -*-

import scipy as sp
import numpy as np
import math as ma
import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import inv
from entities.DivingBoard import DivingBoard


def create_matrix(n):
    """
    Create a CSC matrix.
    :param n: Width and height of CSR matrix.
    :return: A CSC matrix
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

print("Oppgave 3: ")
# Constants
g = 9.81        # Gravitational force in m/s^2.
segments = 10   # Number of segments

divingBoard = DivingBoard(2.00, 0.30, 0.03)
A = create_matrix(segments)     # Creating the A matrix
segm_len = divingBoard.length / segments    # Length of each segment
h4 = segm_len ** 4
force = (-480 * divingBoard.width * divingBoard.thickness * g)  # Force acting on a slice of the diving board
b1 = [(force) / (divingBoard.E * divingBoard.I)] * segments  # The b-vector
y = spsolve(A, b1)  # Solving the matrix
np.set_printoptions(formatter={'float': lambda y: "{0:0.20f}".format(y)})

# Eksakt løsning for y
print(y)  # [-0.00018062473846150817 -0.00067484750769219038 -0.00141698658461512916 -0.00234908750769186828 -0.00342092307692241438 -0.00458999335384523578 -0.00582152566153726261 -0.00708847458461388680 -0.00837152196922896080 -0.00965907692307479970]

print()
print("Oppgave 4c: ")
R = sp.ones(segments)
for i in range(1, 11):
    R[i - 1] = (((force / (24 * divingBoard.E * divingBoard.I)) * (i / 5) ** 2) * ((i / 5) ** 2 - 8 * (i / 5) + 24))

A = create_matrix(10)
b2 = 1/h4 * A.dot(R)
print(b2)

# Eksakt løsning for den fjerdederiverte.
print(b2)  # [-0.00482953846153815099 -0.00482953846153943815 -0.00482953846153537283 -0.00482953846154242014 -0.00482953846153212022 -0.00482953846154404644 -0.00482953846152886761 -0.00482953846154513065 -0.00482953846153754123 -0.00482953846153970964]

temp = sp.ones(segments)
tempx = sp.ones(segments)
nyB1 = sp.ones(segments)
for i in range(0, segments):
    temp[i] = (abs(b2[i] - b1[i] * 1/h4))
    nyB1[i] = b1[i] / h4

print("nyB1: ", nyB1)

print("\nOppgave 4d: ")
print("Foroverfeil: ", np.linalg.norm(temp))
print("Relativ foroverfeil: ", np.linalg.norm(temp)/np.linalg.norm(b2))
print("Feilforstørring: ", (np.linalg.norm(temp, 3) / np.linalg.norm(b2, 3)) / 2 ** -52)
print("Cond(A): ", norm(A) * norm(inv(A)))

print("\nOppgave 4e: ")
print("Foroverfeil: ", np.linalg.norm(abs(R - y), np.inf)/2**-52, "maskinepsilon")

err = sp.ones(11)
Cond = sp.ones(11)
for i in range(1, 12):
    segments = 10 * 2 ** i
    segm_len = divingBoard.length / segments  # Length of each segment
    h4 = segm_len ** 4
    A = create_matrix(segments)
    if i < 9:  #
        cond = norm(A) * norm(inv(A))
    else:
        cond = 0
    b = [(force * h4)/(divingBoard.E * divingBoard.I)] * segments
    y = spsolve(A, b)
    err[i - 1] = abs(R[-1] - y[-1])
    Cond[i - 1] = cond

print("\nOppgave 5:")
print("\nError:")
print(err)
print("\nKondisjon:")
print(Cond)

X = sp.ones(11)
Y = sp.ones(11)
R1 = sp.ones(11)
Z = sp.ones(11)
Cond = sp.ones(11)
H2 = sp.ones(11)
H = sp.ones(11)
for i in range(1, 12):
    segments = 10 * 2 ** i
    X[i-1] = segments
    H[i - 1] = divingBoard.length / segments
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    p = 100
    g = 9.81
    b = sp.ones(segments)
    for j in range(0, segments):
        b[j] = ((force - p * g * ma.sin((ma.pi * segm_len * (j+1)) / divingBoard.length))*h4) / (divingBoard.E * divingBoard.I)
    y = spsolve(A, b)
    if i < 9:
        Cond[i-1] = norm(A) * norm(inv(A)) * 2**-52
    else:
        Cond[i-1] = 0.1  # To avoid dividing by zero.
    R1[i-1] = ((force*4)/(24*divingBoard.E*divingBoard.I))*(4-16+24)-((g*100*2) /
            (divingBoard.E*divingBoard.I*ma.pi))*(8/(ma.pi**3)*ma.sin(ma.pi)-8/6+8/2-8/(ma.pi**2))
    Z[i-1] = abs(y[-1] - R1[i-1])
    Y[i-1] = (y[-1])
    H2[i-1] = 4/(segments**2)

pl.plot(X, Y, label = 'Beregnet verdi')  # Oppgave 6b
pl.plot(X, R1, label = 'Eksakt verdi')  # Oppgave 6b
pl.show()

Y = sp.ones(11)
pl.plot(np.log(X), np.log(Z), label='Beregnet feil')  # Oppgave 6c and 6d
pl.plot(np.log(X), np.log(Cond), label='Kondisjonstall')  # Oppgave 6d
pl.plot(np.log(X), np.log(H2), label='Teoretisk feil')  # Oppgave 6d

pl.legend(bbox_to_anchor=(1, 1),
           bbox_transform=pl.gcf().transFigure)

pl.show()

print("Oppgave 7")
for i in range(1, 12):
    segments = 10 * 2 ** i
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
    Y[i - 1] = y[-1]

print("Oppgave 7: ")
print(Y)
