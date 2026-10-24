"""
This file implements the U-match PRH algorithm and related methods. 
"""

import oat_python as oat
import generate_data as gen
import boundary_matrix as bm
import numpy as np
import scipy.sparse as sp
import umatch as um
import plot as pltter
import pandas as pd
from scipy.sparse import lil_matrix
import time

class prh: 
    
    def __init__(self, point_cloud, subcloud_indices, max_homology_dimension: int, dissimilarity_max, benchmarking=False):
        """
        From a point cloud and a subset, construct the boundary matrix of a quotient chain complex 
        and U-match factor it. 
        
        # Parameters
        
        :param point_cloud: a point cloud stored in a way that is compatible with `sklearn.neighbors.radius_neighbors_graph`
        :param subcloud_indices: An array of tuples (or a 2D array) of the form `(idx, diameter)` where `idx` is an integer giving the 
        index of a point in the full cloud which belongs to the subcloud, and `diameter` is a float giving the diameter at which this point enters the subcomplex. If `diameter=0.0` then the point enters the subcomplex at the same time as it enters the full complex.
        :param max_homology_dimension: the maximum dimension of a relative homology classes to compute
        :param dissimilarity_max: maximum diameter for simplices in the Vietoris-Rips filtration. 
        :param benchmarking an optinal boolean to determine whether or not the U-match factorization should be benchmarked. 
        
        # Safety Checks
        
        No additional checks on top of those performed by the constructors for 'bifiltered_boundary_matrix' and 'bifiltered_simplicial_complex'. 
        
        # Instance Data 
        
        - self.boundary_matrix: the `bifiltered_boundary_matrix` object
        - self.index_matching_csc: a CSC representation of the matching matrix 
        - self.index_matching_csr: a CSR representation of the matching matrix
        - self.source_comb: a CSC representation of the source COMB
        - self.target_comb: a CSC representation of the target COMB
        - self.max_homology_dimension
        - self.time_to_factor: time (in seconds) taken to construct and U-match factor
        """
        
        if benchmarking: start = time.perf_counter()
        
        # construct the `bifiltered_boundary_matrix`` 
        # NOTE: this contains the bifiltered simplicial complex object as well 
        distance_matrix_full, distance_matrix_sub = gen.matrices_from_nested_clouds(point_cloud, subcloud_indices[:,0].astype(int), dissimilarity_max)
        two_functions = np.any(np.array(subcloud_indices)[:,1] > 0.0)
        self.boundary_matrix = bm.bifiltered_boundary_matrix(distance_matrix_full, distance_matrix_sub, max_homology_dimension+1, subcloud_indices, two_functions)
        
        # construct a sparse LIL boundary matrix 
        # NOTE: we do this since the `bifiltered_boundary_matrix` object only holds the standard boundary matrix. Here we will: 
        # 1) Construct a LIL matrix holding the boundary matrix with rows permuted by subcomplex membership. This will be a series of O(1)-time look-up operations 
        # 2) This new matrix will then be U-match factored. 
        # 3) Note that this newly constructed boundary matrix is dropped from memory once this constructor is complete.
        n = self.boundary_matrix.shape[0]
        boundary_lil = sp.lil_matrix((n,n))
        for i in range(n): boundary_lil[i,:] = self.boundary_matrix.get_row_by_integer_index(i)
        
        # U-match factor the permuted CSR boundary matrix and store (1) the index matching and (2) the COMBs 
        boundary_factored = um.umatch(boundary_lil)
        if benchmarking: 
            self.time_to_factor = time.perf_counter() - start
        else: self.time_to_factor = None
        self.index_matching_csc = boundary_factored.matching_matrix_csc
        self.index_matching_csr = boundary_factored.matching_matrix_csr
        self.source_comb = boundary_factored.source_comb
        self.target_comb = boundary_factored.target_comb
        self.max_homology_dimension = max_homology_dimension
    
    def simplices_subcomplex_order_data_frame(self): 
        """
        Get a list of all simplices in the simplicial complex sorted by subcomplex filtration order. 
        """
        
        simplices = self.boundary_matrix.get_sorted_simplices_subcomplex_order()
        df = pd.DataFrame({ 
            "simplex" : [simp.vertices for simp in simplices], 
            "dimension" : [simp.dimension for simp in simplices],
            "diameter" : [simp.diameter for simp in simplices],   
            "subcomplex_filtration" : [simp.diameter_subcomplex for simp in simplices]            
        })
        return df
    
    def simplices_full_complex_order_data_frame(self): 
        """
        Get a list of all simplices in the simplicial complex sorted by full complex filtration order. 
        """
            
        simplices = self.boundary_matrix.get_sorted_simplices_full_complex_order()
        df = pd.DataFrame({ 
            "simplex" : [simp.vertices for simp in simplices], 
            "dimension" : [simp.dimension for simp in simplices],
            "diameter" : [simp.diameter for simp in simplices],   
            "subcomplex_filtration" : [simp.diameter_subcomplex for simp in simplices]            
        })
        return df
    
    def target_comb_data_frame(self): 
        """
        Get a data frame containing the target COMB.
        
        Note that this matrix has identically indexed rows and columns, so the `simplex_idx` column of the 
        returned data frame describes the row and column indices of the target COMB in ascending order. 
        """
        
        simplices = self.boundary_matrix.get_sorted_simplices_subcomplex_order()
        rdsi = self.boundary_matrix.bifiltration.row_dictionary_simplex_to_integer
        df = pd.DataFrame({ 
            "simplex_idx" : [simp.vertices for simp in simplices], 
            "chain" : 
                [self.boundary_matrix.bifiltration.get_simplicial_chain_from_vector_of_coefficients(
                    zip( 
                        self.target_comb[:,rdsi[simp.vertices]].indices, 
                        self.target_comb[:,rdsi[simp.vertices]].data
                        ), 
                    order="row"
                    )
                for simp in simplices],
            "dimension" : [simp.dimension for simp in simplices],             
        })
        return df
    
    def source_comb_data_frame(self): 
        """
        Get a data frame containing the source COMB.
            
        Note that this matrix has identically indexed rows and columns, so the `simplex_idx` column of the 
        returned data frame describes the row and column indices of the source COMB in ascending order. 
        """
        
        simplices = self.boundary_matrix.get_sorted_simplices_full_complex_order()
        cdsi = self.boundary_matrix.bifiltration.column_dictionary_simplex_to_integer
        df = pd.DataFrame({ 
            "simplex_idx" : [simp.vertices for simp in simplices], 
            "chain" : 
                [self.boundary_matrix.bifiltration.get_simplicial_chain_from_vector_of_coefficients(
                    zip( 
                        self.source_comb[:,cdsi[simp.vertices]].indices, 
                        self.source_comb[:,cdsi[simp.vertices]].data
                        ), 
                    order="column"
                    )
                for simp in simplices],
            "dimension" : [simp.dimension for simp in simplices],             
            })
        return df
    
    def relative_boundary_matrix_data_frame(self):
        """
        Get a data frame containing the relative boundary matrix.
                    
        We return the chains alongs the rows of this matrtix together with their simplex indices.  
        """
        
        simplices = self.boundary_matrix.get_sorted_simplices_subcomplex_order()
        # note that we use `cdsi` here since the copy of the boundary matrix that we store is NOT permuted! 
        cdsi = self.boundary_matrix.bifiltration.column_dictionary_simplex_to_integer
        df = pd.DataFrame({ 
            "simplex_idx" : [simp.vertices for simp in simplices], 
            "chain" : 
                [self.boundary_matrix.bifiltration.get_simplicial_chain_from_vector_of_coefficients(
                    zip( 
                        self.boundary_matrix.boundary_csr[cdsi[simp.vertices],:].indices, 
                        self.boundary_matrix.boundary_csr[cdsi[simp.vertices],:].data
                        ), 
                    )
                for simp in simplices],             
            })
        return df
    
    def judge_relative_cycle_birth(self, integer_idx): 
        """
        Judge whether (and if so, when) a column vector `self.source_comb[:,integer_idx]` gives birth to a relative cycle. 
        
        # Parameters 
        
        :param integer_idx: an integer index for a column of `self.source_comb` 
        
        # Returns 
        
        A finite real number giving the birth of the column as a relative cycle, or `np.inf`. 
        """
        
        # get the simplex index for this column of `self.source_comb` and check its birth filtration in full complex 
        simplex_idx = self.boundary_matrix.bifiltration.column_dictionary_integer_to_simplex[integer_idx]
        birth_of_chain_in_full_complex = simplex_idx.diameter
        
        # check if identically indexed column of `self.index_matching` is nonzero, and handle the cases accordingly
        # NOTE: since we know the structure of the index mathing, we can implement this in constant time! 
        start_of_column_slice = self.index_matching_csc.indptr[integer_idx]
        if start_of_column_slice != self.index_matching_csc.indptr[integer_idx + 1]: # relative (but not absolute) cycle case
            # the given chain is not an absolute cycle, so check if its has subcomplex boundary
            row_idx_of_pivot_as_integer = self.index_matching_csc.indices[start_of_column_slice]
            row_idx_of_pivot_as_simplex = self.boundary_matrix.bifiltration.row_dictionary_integer_to_simplex[row_idx_of_pivot_as_integer]
            subcomplex_diameter_of_row_idx_of_pivot_as_simplex = row_idx_of_pivot_as_simplex.diameter_subcomplex
            if subcomplex_diameter_of_row_idx_of_pivot_as_simplex != None: 
                birth_of_boundary_of_chain_in_subcomplex = subcomplex_diameter_of_row_idx_of_pivot_as_simplex
            else: 
                birth_of_boundary_of_chain_in_subcomplex = np.inf
        else: # absolute cycle case ... the boundary of the chain is the zero vector, which is a subcomplex chain trivially! 
            birth_of_boundary_of_chain_in_subcomplex = 0.0
        
        # this states that a relative cycle must be a chain which 
        # - exists in the full complex
        # - has a boundary in the subcomplex 
        return np.max( [birth_of_boundary_of_chain_in_subcomplex, birth_of_chain_in_full_complex] )

    def relative_cycle_basis(self): 
        """
        Return a 2d array where each row gives a tuple `(idx, birth)` where idx is an integer index of `self.source_comb` and 
        `birth` is a float giving the birth of `self.source_comb[:,idx]` as a relative cycle. The 2d array should be sorted in 
        ascending order by 
        (1) dimension of the indexing simplex corresponding to `idx`, 
        (2) `birth` as relative cycle and 
        (3) lexicographic order on corresponding indexing simplex for `idx`. 
        """
        
        # construct data frame
        dimensions = np.array([self.boundary_matrix.bifiltration.column_dictionary_integer_to_simplex[integer_idx].dimension for integer_idx in range(self.boundary_matrix.shape[0])])
        relative_cycle_data = pd.DataFrame({ 
            "integer idx" : [integer_idx for integer_idx in range(self.boundary_matrix.shape[0])],
            "birth filtration" : [self.judge_relative_cycle_birth(integer_idx) for integer_idx in range(self.boundary_matrix.shape[0])],
            "dimension" : dimensions,
            "simplex" : [self.boundary_matrix.bifiltration.column_dictionary_integer_to_simplex[integer_idx].vertices for integer_idx in range(self.boundary_matrix.shape[0])]
        })
        # sort data frame 
        relative_cycle_data = relative_cycle_data.sort_values(
                by = ["dimension", "birth filtration", "simplex"], 
                ascending=[True, True, True]
            ).reset_index(drop=True)
        # return 
        return relative_cycle_data
        
        
    def judge_relative_boundary_birth(self, integer_idx): 
        """
        Judge whether (and if so, when) a column vector `self.target_comb[:,integer_idx]` gives birth to a relative boundary. 
        
        # Parameters 
        
        :param integer_idx: an integer index for a column of `self.target_comb` 
        
        # Returns 
        
        A finite real number giving the birth of the column as a relative boundary, or `np.inf`. 
        """
        
        # get the simplex index for this column of `self.target_comb` and check its birth filtration in the subcomplex 
        simplex_idx = self.boundary_matrix.bifiltration.row_dictionary_integer_to_simplex[integer_idx]
        subcomplex_diameter_of_simplex_idx = simplex_idx.diameter_subcomplex
        if subcomplex_diameter_of_simplex_idx != None: 
            birth_of_chain_in_subcomplex = subcomplex_diameter_of_simplex_idx
        else: 
            birth_of_chain_in_subcomplex = np.inf
        
        # check if identically indexed row of `self.index_matching` is nonzero, and get the column index (as a simplex) of the pivot
        # NOTE: since we know the structure of the index mathing, we can implement this in constant time! 
        start_of_row_slice = self.index_matching_csr.indptr[integer_idx]
        if start_of_row_slice != self.index_matching_csr.indptr[integer_idx + 1]:
            column_idx_of_pivot_as_integer = self.index_matching_csr.indices[start_of_row_slice]
            column_idx_of_pivot_as_simplex = self.boundary_matrix.bifiltration.column_dictionary_integer_to_simplex[column_idx_of_pivot_as_integer]
            birth_of_chain_as_absolute_boundary = column_idx_of_pivot_as_simplex.diameter
        else: 
            birth_of_chain_as_absolute_boundary = np.inf
        
        # this states that a relative boundary is a relative cycle which is either 
        # - a chain in the subcomplex
        # - an absolute boundary 
        return np.min( [birth_of_chain_in_subcomplex, birth_of_chain_as_absolute_boundary] ) 
    
    def relative_boundary_basis(self): 
        """
        Return a 2d array where each row gives a tuple `(idx, birth)` where idx is an integer index of `self.target_comb` and 
        `birth` is a float giving the birth of `self.target_comb[:,idx]` as a relative boundary. The 2d array should be sorted in 
        ascending order by 
        (1) dimension of the indexing simplex corresponding to `idx`, 
        (2) `birth` as relative boundary and 
        (3) lexicographic order on corresponding indexing simplex for `idx`. 
        """

        # construct data frame
        dimensions = np.array([self.boundary_matrix.bifiltration.row_dictionary_integer_to_simplex[integer_idx].dimension for integer_idx in range(self.boundary_matrix.shape[0])])
        relative_boundary_data = pd.DataFrame({ 
            "integer idx" : [integer_idx for integer_idx in range(self.boundary_matrix.shape[0])],
            "birth filtration" : [self.judge_relative_boundary_birth(integer_idx) for integer_idx in range(self.boundary_matrix.shape[0])],
            "dimension" : dimensions,
            "simplex" : [self.boundary_matrix.bifiltration.row_dictionary_integer_to_simplex[integer_idx].vertices for integer_idx in range(self.boundary_matrix.shape[0])]
        })
        # sort data frame 
        relative_boundary_data = relative_boundary_data.sort_values(
                by = ["dimension", "birth filtration", "simplex"], 
                ascending=[True, True, True]
            ).reset_index(drop=True)
        # return
        return relative_boundary_data
    
    def matched_basis(self, gamma, beta): 
        """
        Compute the matched basis from a filtered cycle basis and a filtered boundary basis. 
        
        See [Lentz, Ziegelmeier & Henselman-Petrusek](https://arxiv.org/abs/2602.03163)
        """
        
        # The most important step: 
        # permute columns of S (resp. T) by relative cycle (resp. boundary) birth
        # NOTE: we also permute the rows of A to ensure that multiplication A^{-1}B is compatible!  
        # NOTE: `perm` is an array such that perm[i] is the integer index of the row in the permuted 
        # boundary matrix which appears as the ith row of the original boundary matrix. 
        row_indices_of_B = self.boundary_matrix.get_sorted_simplices_full_complex_order()
        rdsi = self.boundary_matrix.bifiltration.row_dictionary_simplex_to_integer
        perm = [rdsi[simplex_idx.vertices] for simplex_idx in row_indices_of_B]
        A = self.target_comb[perm,:][:,beta]
        B = self.source_comb[:,gamma]
        X = sp.linalg.inv(A) @ B
        basis_matching_umatch = um.umatch(lil_matrix(X))
        matching = basis_matching_umatch.matching_matrix_csc
        matched_basis = A @ basis_matching_umatch.target_comb
        pi = matching.indices
        
        return (matched_basis, pi)
        
    
    def main_algorithm(self, testing=False): 
        """
        Implement the main algorithm of [Lentz, Ziegelmeier & Henselman-Petrusek](https://arxiv.org/abs/2602.03163)
        """
        
        if self.time_to_factor is not None: start = time.perf_counter()
        
        # construct relative cycle basis ==========
        relative_cycle_data = self.relative_cycle_basis()
        gamma = relative_cycle_data["integer idx"].values
        if testing: 
            print( " ========== Print Permutation of SOURCE Column Indices by Relative Cycle Birth ========== " )
            print(gamma)
        
        # construct relative boundary basis ==========
        relative_boundary_data = self.relative_boundary_basis()
        beta = relative_boundary_data["integer idx"].values
        if testing: 
            print( " ========== Print Permutation of TARGET Column Indices by Relative Boundary Birth ========== " )
            print(beta)
        
        # construct matched basis ==========
        matched_basis, pi = self.matched_basis(gamma, beta)
        if testing: 
            print( " ========== Print the column indices of 'matched_basis' sorted by ascending relative cycle birth  ========== " )
            print(pi)
    
        # NOTE: The indices of the matched basis are: 
        # - Rows --> identical to the row indices of the original boundary matrix we factored 
        # - Columns --> respect relative boundary order 
        # NOTE: the array 'pi' allows us to iterate over columns of `matched_basis` in order of relative cycle birth
        
        # decompose into interval modules ==========
        # NOTE: we iterate over the columns of the matched basis in relative cycle order using `pi`
        # NOTE: this information is stored in a data frame where each row contains 
        # - dimension --> same as the dimension of the chain generating the class!
        # - birth --> use Theorem 15
        # - death --> use Theorem 14
        # - chain --> a vector of (simplex, coefficient) pairs
        m = np.sum(relative_cycle_data["birth filtration"].values != np.inf) 
        n = np.sum(relative_boundary_data["birth filtration"].values != np.inf) 
        if n > m: 
            raise ValueError(f"Computed more relative boundaries than relative cycles!")
        decomposition = pd.DataFrame(columns=["dimension", "birth", "death", "chain", "chain nnz"])
        for i in range(m): 
            idx = pi[i]
            dimension = self.boundary_matrix.bifiltration.column_dictionary_integer_to_simplex[gamma[i]].dimension
            if dimension > self.max_homology_dimension: 
                continue
            birth = relative_cycle_data.loc[relative_cycle_data["integer idx"] == gamma[i], "birth filtration"].iloc[0] # relative_cycle_data[gamma[i]][1]
            death = relative_boundary_data.loc[relative_boundary_data["integer idx"] == beta[idx], "birth filtration"].iloc[0] # relative_boundary_data[beta[idx]][1]
            if birth > death: 
                raise ValueError(f"Computed a class whose death appears before its birth!")
            if birth == death: 
                continue 
            chain = self.boundary_matrix.bifiltration.get_simplicial_chain_from_vector_of_coefficients(
                zip( matched_basis[:,idx].indices, matched_basis[:,idx].data,), 
                order="column"
            )
            chain_nnz = len(chain)
            # push the new class and its information to the homology df 
            decomposition.loc[len(decomposition)] = [dimension, birth, death, chain, chain_nnz]
        
        if self.time_to_factor is not None: 
            return decomposition, time.perf_counter()-start
        else: 
            return decomposition

# test script
def main():
    
    # NOTE: write AC and AB to denote absolute cycle and absolute boundary, and similarly introduce RC and RB
    
    # generate a point cloud 
    pi = np.pi 
    radians = np.array( [ i*pi/4 for i in range(2, -6, -1)] )
    xvals = np.round(5*np.cos(radians), decimals = 8)
    yvals = np.round(5*np.sin(radians), decimals = 8)
    cloud = np.array(list(zip(xvals, yvals)))
    
    # TEST 1
    
    # pick points to constitute a subspace 
    subcloud_indices = [1,5]
    subcloud_indices_with_custom_diameters = np.array([ (1,0.25), (5,0.5) ])
    # plot the nested point clouds
    enclosing = oat.dissimilarity.enclosing_radius_for_points(cloud, argminmax=False)
    dissimilarity_matrix_full_space, dissimilarity_matrix_subspace = gen.matrices_from_nested_clouds(
        cloud, 
        subcloud_indices, 
        enclosing + 0.0000001 # to account for numerical error
    )
    fig1 = pltter.plot_point_cloud_with_subcloud_2d(
        "Testing PRH 1", 
        xvals, 
        yvals,  
        subcloud_indices, 
        dissimilarity_matrix_full_space, 
        dissimilarity_matrix_subspace, 
        3.9 + 0.0000001
    )
    # construct the prh object
    prh_algorithm_manager = prh(cloud, subcloud_indices_with_custom_diameters, 1, 3.9 + 0.0000001)
    # compute birth and death filtration values
    relative_cycle_births = np.array( 
        [
            (integer_idx, prh_algorithm_manager.judge_relative_cycle_birth(integer_idx)) for 
            integer_idx in range(prh_algorithm_manager.boundary_matrix.shape[0])
        ] 
    )
    relative_boundary_births = np.array(
        [
            (integer_idx, prh_algorithm_manager.judge_relative_boundary_birth(integer_idx)) for 
            integer_idx in range(prh_algorithm_manager.boundary_matrix.shape[0])
        ]
    )
    # assert correct dimension of bases
    assert( np.sum(relative_cycle_births[:,1] != np.inf) == 10 )
    assert( np.sum(relative_boundary_births[:,1] != np.inf) == 8 )
    # ==========
    # PRINT RELATIVE CYCLE INFORMATION 
    # print( " ========== Print Relative Cycle Births as (integer_index, birth) pairs ========== " )
    # print(relative_cycle_births)
    # print( " ========== Print Simplex Indices for Source COMB ========== " )
    # print( [x.vertices for x in prh_algorithm_manager.boundary_matrix.get_sorted_simplices_full_complex_order()] )
    # print( " ========== Print dense Source COMB to view chains ========== ")
    # print(  prh_algorithm_manager.source_comb.todense() )
    # # PRINT RELATIVE BOUNDARY INFORMATION 
    # print( " ========== Print Relative Boundary Births as (integer_index, birth) pairs ========== " )
    # print(relative_boundary_births)
    # print( " ========== Print Simplex Indices for Target COMB ========== " )
    # print( [x.vertices for x in prh_algorithm_manager.boundary_matrix.get_sorted_simplices_subcomplex_order()] )
    # print( " ========== Print dense Target COMB to view chains ========== ")
    # print(  prh_algorithm_manager.target_comb.todense() )
    # ==========
    # Case 1: AC subsumed by AB
    assert(relative_cycle_births[0,1] == 0.0)
    assert(relative_boundary_births[2,1] == 3.8268343258520847)
    # Case 2: AC subsumed by RB
    assert(relative_cycle_births[1,1] == 0.0)
    assert(relative_boundary_births[0,1] == 0.25)
    assert(relative_cycle_births[0,1] == 0.0)
    assert(relative_boundary_births[1,1] == 0.5)
    # Case 3: Essential AC
    # this is the whole loop
    assert(relative_cycle_births[15,1] == 3.8268343258520847)
    assert( prh_algorithm_manager.source_comb.getcol(15).getnnz() == 8)
    # Case 4: Essential RC
    # this is the half loop whose boundary is the 0-chain [1] + [5]
    assert(relative_cycle_births[13,1] == 3.8268343258520847)
    assert( prh_algorithm_manager.source_comb.getcol(13).getnnz() == 4 )
    # Case 5: Chain which is not a RC
    # this chain is just the simplex [0,1], which does not have a subcomplex boundary! 
    assert(relative_cycle_births[8,1] == np.inf)
    # this also holds for the following chains, each of which are 1-simplices with no subcomplex boundary: 
    assert(relative_cycle_births[9,1] == np.inf)
    assert(relative_cycle_births[10,1] == np.inf)
    assert(relative_cycle_births[11,1] == np.inf)
    assert(relative_cycle_births[12,1] == np.inf)
    assert(relative_cycle_births[14,1] == np.inf)
    # show the nested clouds to reference
    # fig1.show()
    
    # TEST 2
    # this test case runs the main algorithm and checks the interval decomposition
    print(prh_algorithm_manager.main_algorithm(testing=True))
    
    # TEST 3
    # this test case will use the same point cloud as in Test 1 above, but will 
    # - handle the RC cases
    # - permute the row order dictionaires a bit more 
    
    # pick points to constitute a subspace 
    subcloud_indices = [1,2,3,4,5]
    subcloud_indices_with_custom_diameters = np.array([ (1,0.25), (2,3.85), (3,3.85), (4, 3.85), (5,0.5)])
    # plot the nested point clouds
    enclosing = oat.dissimilarity.enclosing_radius_for_points(cloud, argminmax=False)
    dissimilarity_matrix_full_space, dissimilarity_matrix_subspace = gen.matrices_from_nested_clouds(
        cloud, 
        subcloud_indices, 
        enclosing + 0.0000001 # to account for numerical error
    )
    fig2 = pltter.plot_point_cloud_with_subcloud_2d(
        "Testing PRH 2", 
        xvals, 
        yvals, 
        subcloud_indices, 
        dissimilarity_matrix_full_space, 
        dissimilarity_matrix_subspace, 
        3.9 + 0.0000001
    )
    # construct the prh object
    prh_algorithm_manager = prh(cloud, subcloud_indices_with_custom_diameters, 1, 3.9 + 0.0000001)
    # compute birth and death filtration values
    relative_cycle_births = np.array( 
        [
            (integer_idx, prh_algorithm_manager.judge_relative_cycle_birth(integer_idx)) for 
            integer_idx in range(prh_algorithm_manager.boundary_matrix.shape[0])
        ] 
    )
    relative_boundary_births = np.array(
        [
            (integer_idx, prh_algorithm_manager.judge_relative_boundary_birth(integer_idx)) for 
            integer_idx in range(prh_algorithm_manager.boundary_matrix.shape[0])
        ]
    )
    # ==========
    # PRINT RELATIVE CYCLE INFORMATION 
    # print( " ========== Print Relative Cycle Births as (integer_index, birth) pairs ========== " )
    # print(relative_cycle_births)
    # print( " ========== Print Simplex Indices for Source COMB ========== " )
    # print( [x.vertices for x in prh_algorithm_manager.boundary_matrix.get_sorted_simplices_full_complex_order()] )
    # print( " ========== Print dense Source COMB to view chains ========== ")
    # print(  prh_algorithm_manager.source_comb.todense() )
    # # PRINT RELATIVE BOUNDARY INFORMATION 
    # print( " ========== Print Relative Boundary Births as (integer_index, birth) pairs ========== " )
    # print(relative_boundary_births)
    # print( " ========== Print Simplex Indices for Target COMB ========== " )
    # print( [x.vertices for x in prh_algorithm_manager.boundary_matrix.get_sorted_simplices_subcomplex_order()] )
    # print( " ========== Print dense Target COMB to view chains ========== ")
    # print(  prh_algorithm_manager.target_comb.todense() )
    # ==========
    # Case 6: RC subsumed by AB
    # this cannot happen ... a chain with nontrivial subspace boundary CANNOT be an absolute boundary, which has trivial boundary 
    # Case 7: RC subsumed by RB
    assert(relative_cycle_births[13,1] == 3.8268343258520847)
    assert(relative_boundary_births[8,1] == 3.85)
    # show the nested clouds to reference
    # fig2.show()
    
if __name__ == '__main__':
    main()