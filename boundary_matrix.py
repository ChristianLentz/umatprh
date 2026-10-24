"""
This file constructs the boundary matrix of a chain complex that is compatible 
with two filtration functions.  
"""

from itertools import combinations
import oat_python as oat
import generate_data as gen
import bifiltered_simplices as bs
import scipy as sp
import numpy as np

class bifiltered_boundary_matrix: 
    """
    The boundary matrix for the chain complex of a filtered simplicial complex which is compatible with two (nested) filtrations. 
    """
    
    def __init__(self, distance_matrix_full_cloud, distance_matrix_subcloud, dimension_max: int, subcloud_indices, two_functions=False): 
        """
        Construct the Vietoris-Rips boundary matrix of an arbitrarily bifiltered chain complex that is compatible with
        both filtration orders. 
        
        The user must provide two distance matrices of the same size encoding nested point clouds. For example, if 
        the matrices are D1 and D2, with D2 holding the subcloud, then any entry of D2 indexed by a point of D1 which 
        is not present in D2 should be zero, and any nonzero entry in D2 should be present in D1. That is, if D2[i,j]
        is nonzero then D1[i,j] is nonzero. 
        
        :param distance_matrix_full_cloud: A sparse Euclidean distance matrix for the point cloud.
        :param distance_matrix_subcloud: A sparse Euclidean distance matrix for some subcloud of the point cloud.
        :param dimension_max: The maximal dimension for simplices, meaning that we may only compute homology up to `dimension_max-1`.
        :param subcloud_indices: An array of tuples (or a 2D array) of the form `(idx, diameter)` where `idx` is an integer giving the 
        index of a point in the full cloud which belongs to the subcloud, and `diameter` is a float giving the diameter at which 
        this point enters the subcomplex. If `diameter=0.0` then the point enters the subcomplex at the same time as it enters the 
        full complex.
        :param two_functions: A boolean indicating whether or not subcomplex diameters are inherited from the full complex. 
        
        NOTE: the safety checks WILL fail if the provided distance matrices are not sparse. 
        
        # Safety Checks 
        
        - Ensure both provided distance matrices are the same size. 
        - Ensure that each nonzero entry in the subcloud distance matrix is also present in the full distance matrix.
        - Ensure both dissimilairty matrices are symmetric. 
        - Ensure that, for both the subcloud and the full cloud, that the filtration parameter of every edge is greater 
        than or equal to the filtration parameter of its vertices. 
        
        NOTE: the last two checks are completed in OAT when we construct the boundary matrix for the full cloud. 
        
        # Instance Data 
        
        Each instance of the `bifiltered_boundary_matrix` holds the following data: 
        
        - `self.boundary_csr` --> A CSR matrix giving the boundary matrix of the Vietoris-Rips complex of the full cloud.  
        - `self.bifiltration` --> An object holding dictionaries which map between row and column integer indices  
        and simplex indices of `self.boundary_csr` . 
        - `self.shape` --> The shape of `self`, which should be identical to that of `self.boundary_csr`. 
        """ 
        
        # check that provided distance matrices are the same size 
        if distance_matrix_full_cloud.shape != distance_matrix_subcloud.shape: 
            raise ValueError("Provided distance matrices must be the same size.")
        
        # check that provided subcloud distance matrix is symmetric and nested 
        for row_idx in range(distance_matrix_subcloud.shape[0]):
            start = distance_matrix_subcloud.indptr[row_idx]
            end = distance_matrix_subcloud.indptr[row_idx + 1]
            if start == end: # empty row
                continue
            row_cols = [j for j in distance_matrix_subcloud.indices[start:end] if j < row_idx]
            for col_idx in row_cols: 
                if distance_matrix_subcloud[row_idx,col_idx] != distance_matrix_subcloud[col_idx,row_idx]: 
                    raise ValueError("Provided subcloud distance matrix is not symmetric.")
                if distance_matrix_subcloud[row_idx,col_idx] != distance_matrix_full_cloud[row_idx,col_idx]: 
                    raise ValueError("Provided distance matrices are not nested.")                         

        # construct the Vietoris-Rips objects for the full cloud
        vietoris_rips = oat.core.vietoris_rips.VietorisRipsComplex(distance_matrix_full_cloud)
        vietoris_rips_subcloud = oat.core.vietoris_rips.VietorisRipsComplex(distance_matrix_subcloud)
        
        # boundary oracle from OAT
        boundary_oracle = vietoris_rips.boundary_matrix_oracle() 
        
        # extract simplices  
        simplices = vietoris_rips.simplices_for_dimensions( list(range(dimension_max+1)) )
        subcomplex_simplices = vietoris_rips_subcloud.simplices_for_dimensions( list(range(dimension_max+1)) )
        n = simplices.shape[0]
        
        # NOTE: this chunk updates all subcomplex simplices when the filtration is not inherited from the full complex. 
        # Since we are not working with the Rust code on the backend, we must enumerate all the simplices that have 
        # custom diameters and update them manually! 
        if two_functions: 
            # append an additional "dimension" column to the data frame that we will need later
            dimensions = [ len(vertices)-1 for vertices in subcomplex_simplices["simplex"] ]
            subcomplex_simplices["dimension"] = dimensions
            # enumerate all simplices in the subcomplex which have a custom diameter
            customized_subcomplex_vertices = np.array([i for i, d in subcloud_indices if d > 0.0]).astype(int)
            customized_subcomplex_simplices = [
                subset
                for r in range(1, (dimension_max + 1) + 1)
                for subset in combinations(customized_subcomplex_vertices, r)
                # trim to only include simplices (1) which actually exist in the subcomplex and (2) whose vertices are contained in 'customized_subcomplex_vertices'
                if not subcomplex_simplices.loc[subcomplex_simplices["simplex"] == subset, "filtration"].empty
            ]
            # update the diameters manually
            for vertices in customized_subcomplex_simplices:  
                current_custom_diameter = max([d for i, d in subcloud_indices if i in vertices])
                current_actual_diameter = subcomplex_simplices.loc[subcomplex_simplices["simplex"] == vertices, "filtration"].iloc[0]
                subcomplex_simplices.loc[subcomplex_simplices["simplex"] == vertices, "filtration"] = max(current_custom_diameter, current_actual_diameter)
            # sort the resulting data frame accordingly
            subcomplex_simplices = subcomplex_simplices.sort_values(
                by = ["dimension", "filtration", "simplex"], 
                ascending=[True, True, True]
            ).reset_index(drop=True)
            subcomplex_simplices = subcomplex_simplices.drop(columns=["dimension"])
    
        # the instance data for this matrix object     
        self.boundary_csr = sp.sparse.csr_matrix((n,n))          
        self.bifiltration = bs.bifiltered_simplicial_complex(
            simplices, 
            simplices.shape[0], 
            subcomplex_simplices, 
            subcomplex_simplices.shape[0]
        )                                
        self.shape = (n,n)
        
        # write the matrix to a CSR using `self.bifiltration` to sort row/column indices properly
        # NOTE: storing this matrix with efficient look-up operations prevents repeated sorting 
        rows = []
        cols = []
        data = []
        cdsi = self.bifiltration.column_dictionary_simplex_to_integer
        cdis = self.bifiltration.column_dictionary_integer_to_simplex
        for i in range(n):
            simplex_indexing_row_i = cdis[i]
            # rows with simplices that have dimension equal to `dimension_max` will be zero!
            if simplex_indexing_row_i.dimension < dimension_max:
                row_i = boundary_oracle.row_for_simplex(simplex_indexing_row_i.vertices)
                simplices_in_row_i = row_i["simplex"].tolist()
                coefficients_in_row_i = row_i["coefficient"].tolist()
                for simplex, coeff in zip(simplices_in_row_i, coefficients_in_row_i):
                    rows.append(i)
                    cols.append(cdsi[simplex])
                    data.append(coeff)
        self.boundary_csr = sp.sparse.coo_matrix(
            ( 
            np.asarray(data, dtype=np.float64), 
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))
            ),
            shape=(n, n)
        ).tocsr()
        
        # NOTE: the integer indices of `self.boundary_csr` should map to simplex indices using the dictionaries held by `self.bifiltration`
        
    def get_sorted_simplices_subcomplex_order(self): 
        """
        Return a list of simplices sorted by subcomplex order. These are the row indices of the permuted/relative boundary matrix. 
        """
        
        return [self.bifiltration.row_dictionary_integer_to_simplex[i] for i in range(self.shape[0])]
        
    def get_sorted_simplices_full_complex_order(self): 
        """
        Return a list of simplices indexing the columns in sorted order. These are the column indices of the standard boundary matrix 
        and the permuted boundary matrix. 
        """
        
        return [self.bifiltration.column_dictionary_integer_to_simplex[i] for i in range(self.shape[0])]
        
    def get_row_by_integer_index(self, row_idx): 
        """
        Lookup a row by provided integer index. 
        """
        
        row_idx_as_simplex = self.bifiltration.row_dictionary_integer_to_simplex[row_idx]
        new_row_idx_as_integer = self.bifiltration.column_dictionary_simplex_to_integer[row_idx_as_simplex.vertices]
        return self.boundary_csr[new_row_idx_as_integer,:]
    
    def get_column_by_integer_index(self, col_idx): 
        """
        Lookup a column by provided integer index. Note that the returned vector is NOT sorted by subcomplex order. 
        """
        
        return self.boundary_csr[:,col_idx]
        
    def get_row_by_simplex_index(self, row_idx): 
        """
        Lookup a row by provided simplex index. The simplex should be provided by its vertices.
        """
        
        row_idx_as_integer = self.bifiltration.row_dictionary_simplex_to_integer[row_idx]
        return self.boundary_csr[row_idx_as_integer,:]
    
    def get_column_by_simplex_index(self, col_idx): 
        """
        Lookup a column by provided simplex index. The simplex should be provided by its vertices. Note that the returned 
        vector is NOT sorted by subcomplex order.
        """
        
        col_idx_as_integer = self.bifiltration.column_dictionary_simplex_to_integer[col_idx]
        return self.boundary_csr[:,col_idx_as_integer]

    def get_boundary_of_chain(self, chain): 
        """
        Use a matrix-vector multiplication to compute the boundary of a provided chain. 
        
        The chain should be given by a vector with integer support and coefficients in the field of rationals. It
        is assumed that the support is sorted by the full complex order. 
        
        The vector returned is formatted identically. To convert these vectors to actual simplicial chains, use the 
        dictionaries held by `self.bifiltration`. 
        """
        
        return self.boundary_csr @ chain
        
# test script
def main():
    # construct nested point clouds
    n = 100
    cloud = gen.circle_sample(n_points = n)
    subcloud_indices = np.array([(i*10+9, 0.0) for i in range(n//10)])
    subcloud_indices[0,1] = 0.5
    subcloud_indices[4,1] = 0.75
    subcloud_indices[8,1] = 1
    
    # nested distance matrices
    full_cloud, subcloud = gen.matrices_from_nested_clouds(cloud, subcloud_indices[:,0].astype(int), 1)
    
    # construct the boundary matrix
    D = bifiltered_boundary_matrix(full_cloud, subcloud, 2, subcloud_indices, True)
    
    # assert that this is a boundary matrix 
    # (i.e. it is nilpotent with order 2, or the boundary of a boundary is trivial)
    boundary_of_boundary = D.boundary_csr @ D.boundary_csr
    assert( boundary_of_boundary.nnz == 0 )
    
if __name__ == '__main__':
    main()