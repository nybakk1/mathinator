import scipy as sp
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import norm
from scipy.sparse.linalg import inv


def lagA(n):
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e, -4*e, e], [-2, -1, 0, 1, 2], n, n)
    A = lil_matrix(A)
    B = csr_matrix([[16, -9, 8/3, -1/4],  # TODO: Spør Rivertz om vi kan hardkode her.
                    [16/17, -60/17, 72/17, -28/17],
                    [-12/17, 96/17, -156/17, 72/17]
                    ])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]

    return A.tocsc()

L = 2.00  # Length of the beam
n = 10  # Number of segments
A = lagA(n)  # Creating the A matrix
h = L / n  # Length of each segment
h4 = h ** 4
w = 0.30  # Width
d = 0.03  # Thickness
E = 1.3 * (10 ** 10)  # Young’s modulus of the wood
I = (w * (d ** 3))/12  # The area moment of inertia I around the center of mass of a beam
g = 9.81  # Gravitational force
force = (-480 * w * d * g * L) / n  # Force acting on a slice of the diving board
b1 = [(force * h4)/(E * I)]*n  # The b-vector
y = spsolve(A, b1)  # Solving the matrix
# print(b1)

R = []
for i in range(1, 11):
    R.append(force/(24*E*I)*((i/5)**2)*((i/5)**2 - 8*(i/5)+24))

A = lagA(10)
b2 = A.dot(R)
# b2 = b2 * (1/h4)
# print(b2)

temp = sp.ones(n)
tempx = sp.ones(n)
for i in range (0, n):
    temp[i] = (abs(b2[i]-b1[i]))

print("Feilforstørring: ", (np.linalg.norm(temp, 3)/np.linalg.norm(b2, 3))/2**-52)
print("Cond(A): ", norm(A) * norm(inv(A)))

