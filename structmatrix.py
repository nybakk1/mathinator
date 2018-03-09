import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def lagA(n):
    e = sp.ones(n)
    A = spdiags([e, -4*e, 6*e, -4*e, e], [-2, -1, 0, 1, 2], n, n)
    A = lil_matrix(A)  # TODO: Spør Rivertz om vi kan hardkode her.
    B = csr_matrix([[16, -9, 8/3, -1/4],
                    [16/17, -60/17, 72/17, -28/17],
                    [-12/17, 96/17, 156/17, 72/17]
                    ])

    A[0, 0:4] = B[0, :]
    A[n-2, n-4:n] = B[1, :]
    A[n-1, n-4:n] = B[2, :]

    return A


L = 2.00
n = 10
A = lagA(n)
h = L / n
h4 = h ** 4
w = 0.30
d = 0.03
E = 1.3 * (10 ** 10)
I = (w * (d ** 3))/12
g = -9.81
weight = -480 * w * d * g
b = [(weight * h4)/(E * I)]*n

y = spsolve(A, b)
print(y)
