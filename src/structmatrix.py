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

divingBoard = DivingBoard(2.00, 0.30, 0.03)  # Initializing the divingboard.
A = create_matrix(segments)     # Creating the A matrix
segm_len = divingBoard.length / segments    # Length of each segment
h4 = segm_len ** 4
force = (-480 * divingBoard.width * divingBoard.thickness * g)  # Force acting on a slice of the diving board
b1 = [(force * h4) / (divingBoard.E * divingBoard.I)] * segments  # The b-vector
y = spsolve(A, b1)  # Solving the matrix
np.set_printoptions(formatter={'float': lambda y: "{0:0.20f}".format(y)})

# Exact solution for y
print(y)  # [-0.00018062473846150817 -0.00067484750769219038 -0.00141698658461512916 -0.00234908750769186828 -0.00342092307692241438 -0.00458999335384523578 -0.00582152566153726261 -0.00708847458461388680 -0.00837152196922896080 -0.00965907692307479970]

print()
R = sp.zeros(segments)
# Estimated solution for y
for i in range(1, 11):
    R[i - 1] = (((force / (24 * divingBoard.E * divingBoard.I)) * (i / 5) ** 2) * ((i / 5) ** 2 - 8 * (i / 5) + 24))

# Finding the exact solution for the forth derivative.
A = create_matrix(10)
b2 = 1/h4 * A.dot(R)

print("Oppgave 4c: ")
print(b2)  # [-0.00482953846153815099 -0.00482953846153943815 -0.00482953846153537283 -0.00482953846154242014 -0.00482953846153212022 -0.00482953846154404644 -0.00482953846152886761 -0.00482953846154513065 -0.00482953846153754123 -0.00482953846153970964]

temp = sp.zeros(segments)
# Error in the estimation of the forth derivative.
for i in range(0, segments):
    temp[i] = (abs(b2[i] - b1[i] * 1/h4))

print("\nOppgave 4d: ")
print("Foroverfeil: ", np.linalg.norm(temp, np.inf))
print("Relativ foroverfeil: ", np.linalg.norm(temp, np.inf)/np.linalg.norm(b2, np.inf))
print("Feilforstørring: ", (np.linalg.norm(temp, np.inf) / np.linalg.norm(b2, np.inf)) / 2 ** -52)
print("Cond(A): ", norm(A) * norm(inv(A)))

print("\nOppgave 4e: ")
print("Foroverfeil: ", np.linalg.norm(abs(R - y), np.inf)/2**-52, "maskinepsilon")

# Array for containing the error in the point x = L. (The end of the divingboard)
err = sp.zeros(11)
# Array for containing the condition of the A matrix for different values of n.
Cond = sp.zeros(11)

for i in range(1, 12):
    segments = 10 * 2 ** i
    segm_len = divingBoard.length / segments
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

X = sp.zeros(11)  # Number of segments in each sample.
Y = sp.zeros(11)  # Estimated values in the point x = L.
R1 = sp.zeros(11)  # Exact values in the point x = L.
Z = sp.zeros(11)  # Error in the estimate in the point x = L.
Cond = sp.zeros(11)  # Condition of the structure matrix for different values of n.
H2 = sp.zeros(11)  # The theoretical error (h^2)
for i in range(1, 12):
    segments = 10 * 2 ** i
    X[i-1] = segments
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    p = 100
    g = 9.81
    b = sp.ones(segments)
    # Filling in the force acting on the diving board.
    for j in range(0, segments):
        b[j] = ((force - p * g * ma.sin((ma.pi * segm_len * (j+1)) / divingBoard.length))*h4) / (divingBoard.E * divingBoard.I)
    y = spsolve(A, b)
    # FIlling in the condition number as far as python can handle it.
    if i < 9:
        Cond[i-1] = norm(A) * norm(inv(A)) * 2**-52
    else:
        Cond[i-1] = 0.1  # To avoid dividing by zero.
    R1[i-1] = ((force*4)/(24*divingBoard.E*divingBoard.I))*(4-16+24)-((g*100*2) /
            (divingBoard.E*divingBoard.I*ma.pi))*(8/(ma.pi**3)*ma.sin(ma.pi)-8/6+8/2-8/(ma.pi**2))
    Z[i-1] = abs(y[-1] - R1[i-1])
    Y[i-1] = (y[-1])
    H2[i-1] = 4/(segments**2)

pl.plot(X, Y, label ='Beregnet verdi')  # Oppgave 6b
pl.plot(X, R1, label ='Eksakt verdi')  # Oppgave 6b
# Close this window to continue code and show the next plot.
pl.show()

Y = sp.ones(11)
pl.plot(np.log(X), np.log(Z), label='Beregnet feil')  # Oppgave 6c and 6d
pl.plot(np.log(X), np.log(Cond), label='Kondisjonstall')  # Oppgave 6d
pl.plot(np.log(X), np.log(H2), label='Teoretisk feil')  # Oppgave 6d

pl.legend(bbox_to_anchor=(1, 1),
           bbox_transform=pl.gcf().transFigure)

pl.show()

# Excercise 7
# A person standing at the edge of the divingboard.
# Calculating the displacement of the beam at the end of the divingboard for different number of segments.
for i in range(1, 12):
    segments = 10 * 2 ** i
    segm_len = divingBoard.length / segments
    h4 = segm_len ** 4
    A = create_matrix(segments)
    g = 9.81
    b = sp.ones(segments)
    segm_start = 0
    segm_stop = segm_len
    TOL = 0.000001
    f_pers = (g * 50) / 0.3
    for j in range(0, segments):
        if segm_stop > 1.70:
            if segm_start > 1.70:  # Adding the weight of the person for the end of the divingboard.
                b[j] = ((force - f_pers) * h4) / (divingBoard.E * divingBoard.I)
            else:  # A segment that has some part of the weight of the person, but not all.
                b[j] = (((force - (f_pers/(segm_stop - 1.7))) * h4) / (divingBoard.E * divingBoard.I))
        else:
            b[j] = ((force * h4) / (divingBoard.E * divingBoard.I))
        if 1.70 - TOL < segm_stop < 1.70 + TOL:  # In case of small rounding errors.
            b[j] = ((force * h4) / (divingBoard.E * divingBoard.I))
        if 1.70 - TOL < segm_start < 1.70 + TOL:  # In case of small rounding errors.
            b[j] = ((force - f_pers) * h4) / (divingBoard.E * divingBoard.I)
        segm_start += segm_len  # Incrementing the start of the segment
        segm_stop += segm_len  # Incrementing the end of the segment
    y = spsolve(A, b)  # Solving for the displacement.
    Y[i - 1] = y[-1]  # Saving the displacement at the end of the divingboard.

print("Oppgave 7: ")
print(Y)
