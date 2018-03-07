import scipy as sp
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix


def lagA(n):
    e = sp.one(n)
    A = spdiags([e,-4*e,6*e,-4*e,e],[-2,-1,0,1,2],n,n)
    A = lil_matrix(A)
    B = 0 ## TODO: FIX MATRIX!

    A[0,0:4] = B[0,:]
    A[n-2,n-4:n] = B[1,:]
    A[n-1,n-4:n] = B[2,:]

    return A
