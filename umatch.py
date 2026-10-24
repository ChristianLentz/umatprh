"""
This file builds a `umatch` object. The constructor builds the 
U-match factorization of a `LIL` boundary matrix and includes 
helper functions for working with the factorization. 

This is adapted from [Hang, Giusti, Ziegelmeier & Henselman-Petrusek](https://arxiv.org/pdf/2108.08831). 
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class umatch: 
    
    def __init__(self, D): 
        """
        Construct a sparse U-Match decomposition:
        - We require the input to be a sparse LIL matrix, which is optimized for the row reduction
        - The resulting matrices are `CSC` format, optimized for column look-ups
        
        NOTE: we favor CSC representation here as we typically require column access with our workflow.
        
        # Parameters
    
        :param D: a sparse LIL boundary matrix. 
        
        # Returns 

        Constructs a tuple of matrices (T, M, D, S) giving a U-match TM = DS. These matrices are 
        - self.target_comb         (CSC)  : the target COMB
        - self.source_comb         (CSC)  : the source COMB
        - self.mapping             (CSC)  : a copy of the input matrix
        - self.matching_matrix_csc (CSC)  : the CSC matching matrix; a reduced version of the matrix D
        - self.matching_matrix_csr (CSR)  : the CSR matching matrix; a reduced version of the matrix D
        """
        
        # assert correct input
        if not sp.isspmatrix_lil(D):
            raise TypeError("Input must be a SciPy sparse LIL matrix")
        self.mapping = D.copy()

        # initialize matrices and data to return
        m, n = D.shape
        Tinv = sp.eye(m, format="lil") # --> initialize LIL identity matrix
        Sinv = sp.eye(n, format="lil") # --> initialize LIL identity matrix
        M = sp.lil_matrix((m, n))

        # -------------------------------------------------
        # Row operation loop (bottom-to-top)
        # - use LIL matrices here, these are optimized for 
        # iterative row operations
        # - construct T^{-1}
        # - identify pivots of M
        # -------------------------------------------------
        for i in range(m - 2, -1, -1):
            # NOTE: 
            # - J holds all row indices below row D[i,], sorted in descending order (bottom-to-top). 
            # - J is equipped with this order since the reduction D to M uses a bottom-to-top row reduction.
            J = list(range(m - 1, i, -1))
            # - current iteration of the row reduction clears leading entries in row D[i,] using pivots in rows which appear below it 
            # - we proceed left-to-right along row D[i,]
            # - the index of the current leading entry in row D[i,] is referred to by k 
            row_idx = 0
            while row_idx < len(J):
                # if we have cleared each entry in row vector D[i,] ...
                if len(D.rows[i]) == 0:
                    break  
                # get the leading column index in the row
                k = D.rows[i][0]   
                j = J[row_idx]
                # if row D[j,] is nonzero and its leading entry appears in the same column, k, as the leading entry in D[i,] ... 
                if len(D.rows[j]) > 0 and D.rows[j][0] == k:
                    # cancel this leading entry in D[i,] using the entry D[j,k] as a pivot
                    a = D[i, k] / D[j, k]
                    D[i, :] = D.getrow(i) - a * D.getrow(j)
                    # record these operations in Tinv
                    Tinv[i, :] = Tinv.getrow(i) - a * Tinv.getrow(j)
                    # the loop variable for the row needs to be reset, as our pivot is now in a new column
                    row_idx = 0  
                else:
                    # the loop variable for the rows should increment and the loop proceeds
                    row_idx += 1

        # -------------------------------------------------
        # Convert to CSC after elimination
        # -------------------------------------------------
        D = D.tocsc()
        Tinv = Tinv.tocsc()
        D.eliminate_zeros()
        Tinv.eliminate_zeros()
        
        # -------------------------------------------------
        # "Column operation" loop
        # - infer column operation matrix S by using the row
        # reduction of D to construct S^{-1}
        # - this follows from the so-called "inner identities" in 
        # -------------------------------------------------
        for i in range(m):
            row_i = D.getrow(i)
            if row_i.nnz > 0:
                k = row_i.indices[0]
                pivot_val = row_i[0, k]
                Sinv[k, :] = (1.0 / pivot_val) * row_i
                M[i, k] = pivot_val

        Sinv = Sinv.tocsc()
        M_csc = M.tocsc()
        M_csr = M.tocsr()

        # -------------------------------------------------
        # Sparse inversion via PA = LU factorization
        # -------------------------------------------------
        T_lu = spla.splu(Tinv.tocsc())
        S_lu = spla.splu(Sinv.tocsc())
        target = sp.csc_matrix(T_lu.solve(sp.eye(m).toarray()))
        source = sp.csc_matrix(S_lu.solve(sp.eye(n).toarray()))
        
        # -------------------------------------------------
        # Store results of the factorization
        # -------------------------------------------------
        self.target_comb = target
        self.source_comb = source 
        self.matching_matrix_csc = M_csc
        self.matching_matrix_csr = M_csr

# a quick test script
def main(): 
    # construct and u-match factor a matrix
    # NOTE: this is a boundary matrix with mod-2 coefficients
    rows = np.array([1,1,2,2,4,4,5,5,6,6,7,7,8,8]) - 1
    cols = np.array([8,13,8,14,9,10,9,11,11,13,12,14,10,12]) - 1
    vals = np.array([1,-1,-1,1,1,1,-1,1,-1,1,1,-1,-1,-1])
    DD = sp.csr_matrix((vals, (rows, cols)), shape=(14,14))
    umatch_D = umatch(sp.lil_matrix(DD))
    
    # assert correct factorization 
    T = umatch_D.target_comb
    M = umatch_D.matching_matrix_csc
    S = umatch_D.source_comb
    
    # This computes TM and DS and then counts how many nonzero entries are in TM - DS
    assert( (T.dot(M) - DD.dot(S)).nnz == 0 ) 
    print("U-match Test Passed!")
    
if __name__ == '__main__':
    main()