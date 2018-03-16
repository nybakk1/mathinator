# -*- coding: utf-8 -*-

import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def create_matrix(n):
    """
    Create a matrix CSR matrix.
    :param n: Width and height of matrix.
    :return: CSR matrix
    """
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e, -4*e, e], [-2, -1, 0, 1, 2], n, n)
    A = lil_matrix(A)
    B = csr_matrix([[ 16,        -9,           8 /  3,    - 1 /  4],
                    [ 16 / 17,   -60 / 17,    72 / 17,    -28 / 17],
                    [-12 / 17,    96 / 17,  -156 / 17,     72 / 17]
                    ])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]

    return A.tocsr()


length = 2.00       # Length of the beam
segments = 10       # Number of segments
A = create_matrix(segments)  # Creating the A matrix
segm_len = length / segments  # Length of each segment
h4 = segm_len ** 4
width = 0.30    # Width
d = 0.03        # Thickness
E = 1.3 * (10 ** 10)    # Young’s modulus of the wood
I = (width * (d ** 3)) / 12     # The area moment of inertia I around the center of mass of a beam
g = 9.81    # Gravitational force
force = (-480 * width * d * g * length) / segments  # Force acting on a slice of the diving board
b1 = [(force * h4)/(E * I)] * segments  # The b-vector
y = spsolve(A, b1)  # Solving the matrix
# print(y)
print(b1)

R = []
for i in range(1, 11):
    R.append(force / (24 * E * I) * ((i/5) ** 2) * ((i / 5) ** 2 - 8 * (i / 5) + 24))

A = create_matrix(10)
b2 = A.dot(R)
b2 = b2 * (1 / h4)
print(b2)

print(b2[0] / b1[0])



