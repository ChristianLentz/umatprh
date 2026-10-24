"""
This file implements objects for working with a bifiltered simplicial complex. 
"""

import generate_data as gen
import oat_python as oat
import numpy as np

class simplex_bifiltered: 
    """
    A simplex belonging to a bifiltered simplicial complex. 
    """
    
    def __init__(self, vertices, diameter, diameter_subcomplex): 
        """
        Construct a single simplex of a bifiltered simplicial complex. 
        
        :param vertices: A list of integers denoting the vertices of the simplex. 
        :param diameter: A float describing the diameter of the simplex with respect to the full cloud filtration. 
        :param diameter_subcomplex: A float describing the diameter of the simplex with respect to the subcloud filtration. 
        
        # Safety Checks 
        
        - Ensure that the diameter of the simplex in the subcomplex is at least the diameter of the simplex in the full complex. 
        
        # Instance Data 
        
        Each instance of the `simplex_bifiltered` holds the following data: 
        
        - `vertices`
        - `dimension`
        - `diameter`
        - `diameter_subcomplex`
        """ 
        
        if not diameter_subcomplex == None:
            if diameter_subcomplex < diameter: 
                raise ValueError(f"The simplex {vertices} has diameter {diameter}, but enters the subcloud filtration at diameter {diameter_subcomplex}, which is strictly less.")
        
        self.vertices = vertices
        self.dimension = len(vertices)-1
        self.diameter = diameter
        self.diameter_subcomplex = diameter_subcomplex
        
    def to_string(self):
        """
        A string representation of the simplex. Used for debugging. 
        """
        
        return f"Vertices: {self.vertices} | Diameter: {self.diameter} | Subcloud Diameter: {self.diameter_subcomplex}"
    
    def is_in_subcomplex(self): 
        """
        A boolean function determing the subcomplex membership of `self`. 
        """
        
        return not self.diameter_subcomplex == None
    
    def set_subcomplex_diameter(self, diameter): 
        """
        If `self.diameter_subcomplex` is `None`, update it. 
        """
        
        if self.diameter_subcomplex == None and self.diameter <= diameter: 
            self.diameter_subcomplex = diameter
            
    def set_subcomplex_diameter_by_delay(self, delay): 
        """
        If `self.diameter_subcomplex` is `None`, update it. 
        
        Specifically, set this subcomplex diameter by adding a dely to its diameter in the full complex. 
        """
        
        if self.diameter_subcomplex == None: 
            self.diameter_subcomplex = self.diameter + delay
    
    def is_lexicographically_greater_than(self, vertices_of_other_simplex): 
        """
        Compare `self` to another simplex, given by `vertices_of_other_simplex`. 

        Will return true if the vertices of `self` are lexicorgraphically greater than `vertices_of_other_simplex`.
        
        Note that this method is only used for comparing simplices of the same dimension, so we throw an error when 
        attempting to compare to simplices with different dimension. 
        """
        
        if self.dimension != len(vertices_of_other_simplex)-1: 
            raise ValueError(f"Attempted to compare two simplices of different dimension.") 
        else: 
            for x,y in zip(self.vertices, vertices_of_other_simplex):
                if x < y: return False
                else: return True
        
        
class bifiltered_simplicial_complex:
    """
    A bifiltered simplicial complex. 
    """
    
    def __init__(self, simplices_full_complex, num_full_complex_simplices, simplices_subcomplex, num_subcomplex_simplices): 
        """
        Construct a bifiltered simplicial complex from a list of simplices. This object is designed to provide efficient 
        means to interact with the boundary matrix of the associated filtered chain complex. For instance, a column of 
        this boundary matrix with integer support can be written as a simplicial chain using look-up methods which are 
        O(1) complexity. More generally, we can do this for any simplicial chain, not just those that appear in the rows 
        or columns of the associated boundary matrix. 
        
        Let n be the toal number of simplices in the complex, and m the total number of simplices in the subcomplex. 
        
        :param simplices_full_complex: An array with shape (n, 2) containing the simplices of the full complex. The first and 
        second columns should have respective labels "simplex" and "filtration", respectively given by a list of vertices and a float. 
        :param num_full_complex_simplices: Number of simplices contained in the full complex. 
        :param simplices_subcomplex: An array with shape (m, 2) with the exact structure as 'simplices_full_complex'. 
        :param num_subcomplex_simplices: Number of simplices contained in the subcomplex.
        
        NOTE: 'simplices_full_complex' and 'simplices_subcomplex' should be in the exact format returned by the function 
        'simplices_for_dimensions' in `oat_python/core/vietoris_rips`. 
    
        # Safety Checks 
        
        - Ensure that the diameter of each simplex in the subcomplex is at least the diameter of the simplex in the full complex. 
        This is checked by the constructor for the class `simplex_bifiltered`.
        - Ensure that the number of simplices in the subcomplex is no more than the number of simplices in the full complex 
        
        # Instance Data 
        
        Each instance of the `bifiltered_simplicial_complex` holds the following data:
        
        - `self.column_dictionary_integer_to_simplex` --> maps an integer indexing a column of a the relative boundary matrix to the assocaited `simplex_bifiltered` object.
        - `self.column_dictionary_simplex_to_integer` --> maps vertices of a simplex to the assocaited integer indexing a column of the relative boundary matrix.
        - `self.row_dictionary_integer_to_simplex` --> maps an integer indexing a row of the relative boundary matrix to the associated `simplex_bifiltered` object. 
        - `self.row_dictionary_simplex_to_integer` --> maps vertices of a simplex to the assocaited integer indexing a row of the relative boundary matrix.
        - `self.num_subcomplex_simplices` 
        
        NOTE: each of these items is a python dictionary without any duplicate keys or values, so lookup operations are O(1) time 
        complexity. The goal here to make sure that look-up is efficient and does not require repeated sorting operations. Having four 
        dictionaries is cumbersome, but acheives this while still scaling linearly with the number of simplices.
        """ 
        
        # safety check 
        if num_subcomplex_simplices > num_full_complex_simplices: 
            raise ValueError(f"The full complex has {num_full_complex_simplices} simplices, which is strictly less than {num_subcomplex_simplices}, or the number of simplices in the subcomplex.")
        
        # items needed to construct the object
        self.column_dictionary_integer_to_simplex = dict()
        self.column_dictionary_simplex_to_integer = dict()
        self.row_dictionary_integer_to_simplex = dict()
        self.row_dictionary_simplex_to_integer = dict()
        self.num_subcomplex_simplices = num_subcomplex_simplices
        
        # indexing array and boolean mask used to construct the dictionaries 
        indices = np.array(range(num_full_complex_simplices))
        boolean_mask = np.ones(num_full_complex_simplices, dtype=bool)
        
        # insert the subcomplex simplices first 
        for i in range(num_subcomplex_simplices):
            # create the bifiltered simplex object
            vertices_of_i = simplices_subcomplex["simplex"][i]
            mask_to_determine_row_index = simplices_full_complex["simplex"]==vertices_of_i
            row_idx_of_full_complex_df = simplices_full_complex.index[mask_to_determine_row_index][0]
            diameter_of_i = simplices_full_complex["filtration"][row_idx_of_full_complex_df]
            subcomplex_diameter_of_i = simplices_subcomplex["filtration"][i] 
            simplex_i = simplex_bifiltered(vertices_of_i, diameter_of_i, subcomplex_diameter_of_i)
            # column dictionaries
            self.column_dictionary_integer_to_simplex[row_idx_of_full_complex_df] = simplex_i
            self.column_dictionary_simplex_to_integer[vertices_of_i] = row_idx_of_full_complex_df
            # row dictionaries
            self.row_dictionary_integer_to_simplex[i] = simplex_i
            # ===== DEPRECATED =====
            # arr = (simplices_subcomplex["simplex"] == vertices_of_i).to_numpy()
            # row_idx_before_permutation = np.where(arr == True)[0][0]
            # self.row_index_dictionary_integer_to_simplex[row_idx_before_permutation] = simplex_i
            # ===== DEPRECATED =====
            self.row_dictionary_simplex_to_integer[vertices_of_i] = i
            # update boolean mask for indices array 
            boolean_mask[row_idx_of_full_complex_df] = False
        
        # now insert the remaining full complex simplices 
        remaining_indices = indices[boolean_mask]
        for k in range(num_full_complex_simplices - num_subcomplex_simplices):
            # handles indexing 
            permuted_index_to_insert_at = num_subcomplex_simplices+k
            i = remaining_indices[k] 
            # create the bifiltered simplex object
            vertices_of_i = simplices_full_complex["simplex"][i]
            diameter_of_i = simplices_full_complex["filtration"][i]
            simplex_i = simplex_bifiltered(vertices_of_i, diameter_of_i, None)
            # column dictionaries
            self.column_dictionary_integer_to_simplex[i] = simplex_i
            self.column_dictionary_simplex_to_integer[vertices_of_i] = i
            # row dictionaries
            self.row_dictionary_integer_to_simplex[permuted_index_to_insert_at] = simplex_i
            self.row_dictionary_simplex_to_integer[vertices_of_i] = permuted_index_to_insert_at
                
    def get_bifiltered_simplex_object_by_vertices(self, vertices):
        """
        Given the vertices of a simplex, return the associated `bifiltered_simplex` object. 
        
        :param vertices: a tuple of integers
        """
        
        column_index_as_integer = self.column_dictionary_simplex_to_integer[vertices]
        simplex_bifiltered = self.column_dictionary_integer_to_simplex[column_index_as_integer]
        return simplex_bifiltered
    
    def get_simplicial_chain_from_vector_of_coefficients(self, vector_of_coefficients, order="column"): 
        """
        Given a vector with integer support and coefficients in the field of rationals, return the associated simplicial chain as 
        a list of (`simplex`, `coefficient`) tuples. 
        
        :param vector_of_coefficients: a vector with integer support and coefficients in the field of rationals. 
        :param order: a string specifying which ordered list to use when constructing the simplicial chain. Should be either "column" or "row". 
        Note that `order` = "row" indicates that we use the `row_dictionary_integer_to_simplex` list. 
        
        NOTE: The integer indices of the `vector_of_coefficients` should be interpreted as the (integer) row or column indices of the 
        boundary matrix. The user must specify which order to use. 
        """
        
        if order == "column":
            simplicial_chain = [(self.column_dictionary_integer_to_simplex[idx].vertices, coef) for (idx, coef) in vector_of_coefficients]
        elif order == "row": 
            simplicial_chain = [(self.row_dictionary_integer_to_simplex[idx].vertices, coef) for (idx, coef) in vector_of_coefficients]
        else:
            raise ValueError(f"Invalid order specified. Should be either 'column' or 'row'.")
        return simplicial_chain
    
    def is_chain_in_subcomplex(self, chain, order="column"): 
        """
        Given a vector with integer support and coefficients in the field of rationals, determine if the associated simplicial
        chain is in the subcomplex. 
        
        :param vector_of_coefficients: a vector with integer support and coefficients in the field of rationals. 
        :param order: a string specifying which ordered list to use when constructing the simplicial chain. Should be either "column" or "row"
        
        NOTE: The integer indices of the `vector_of_coefficients` should be interpreted as the (integer) row or column indices of the 
        boundary matrix. The user must specify which order to use. 
        """
        
        simplicial_chain = self.get_simplicial_chain_from_vector_of_coefficients(chain, order=order)
        if order == "column":
            integer_indices_from_row_dictionary = np.array([self.row_dictionary_simplex_to_integer[simplex.vertices] for (simplex, _) in simplicial_chain])
        elif order == "row": 
            integer_indices_from_row_dictionary = np.array([idx for (idx, _) in chain])
        else:
            raise ValueError(f"Invalid order specified. Should be either 'column' or 'row'.")
        return np.max(integer_indices_from_row_dictionary) < self.num_subcomplex_simplices
    
# a test script
def main():
    # construct nested point clouds
    n = 20
    cloud = gen.circle_sample(n_points = n)
    subcloud_indices = [0,5,10,15]
    
    # nested distance matrices
    distance_matrix_full_cloud, distance_matrix_subcloud = gen.matrices_from_nested_clouds(cloud, subcloud_indices, 2.5)
    
    # construct the Vietoris-Rips objects
    vietoris_rips = oat.core.vietoris_rips.VietorisRipsComplex(distance_matrix_full_cloud)
    vietoris_rips_subcloud = oat.core.vietoris_rips.VietorisRipsComplex(distance_matrix_subcloud)
        
    # extract simplices 
    simplices = vietoris_rips.simplices_for_dimensions(range(3))
    subcloud_simplices = vietoris_rips_subcloud.simplices_for_dimensions(range(3))
    n = simplices.shape[0]
    m = subcloud_simplices.shape[0]
    
    # construct bifiltration data structure
    bifiltration = bifiltered_simplicial_complex(simplices, n, subcloud_simplices, m)
    
    # test function 'get_bifiltered_simplex_object_by_vertices'
    for vertices in subcloud_simplices["simplex"].tolist():
        simplex_bifiltered_object = bifiltration.get_bifiltered_simplex_object_by_vertices(vertices)
        assert( vertices == simplex_bifiltered_object.vertices )
        
    # test functions `get_simplicial_chain_from_vector_of_coefficients` and `is_chain_in_subcomplex`
    # NOTE: the coefficients are not important here, only the simpices of the linear combination
    chain = np.array([ (0, 5.0), (5, 1.3), (10, 1.2), (15, 1.4) ])
    assert( bifiltration.is_chain_in_subcomplex(chain, order="column") )
    assert( not bifiltration.is_chain_in_subcomplex(chain, order="row") )
    chain_false = np.array([ (0, 5.0), (5, 1.3), (10, 1.2), (15, 32.2), (16, 3.14) ])
    assert( bifiltration.is_chain_in_subcomplex(chain_false, order="column") )
    
    # NOTE: for the tests below we only consider the integer -> simplex dictionaries since the simplex -> integer dictionaries are 
    # constructed identically with keys and values swapped
    
    # --- check that row indexing is correct ---
    for i in range(m-1): # subcomplex simplices
        rdis = bifiltration.row_dictionary_integer_to_simplex
        simplex_i = rdis[i]
        assert( simplex_i.is_in_subcomplex() )
        simplex_after_i = rdis[i+1]
        if simplex_after_i.dimension == simplex_i.dimension:
            assert( simplex_i.diameter_subcomplex <= simplex_after_i.diameter_subcomplex )
        else:
            assert(simplex_after_i.dimension == simplex_i.dimension+1)
    for i in range(n-m-1): # other simplices
        rdis = bifiltration.row_dictionary_integer_to_simplex
        simplex = rdis[i+m]
        assert( not simplex.is_in_subcomplex() )
        simplex_after = rdis[i+m+1]
        if simplex_after.dimension == simplex.dimension:
            assert( simplex.diameter <= simplex_after.diameter )
        else:
            assert(simplex_after.dimension == simplex.dimension+1)
    
    # --- check that column indexing is correct ---
    for i in range(n-1): 
        cdis = bifiltration.column_dictionary_integer_to_simplex
        simplex_i = cdis[i]
        simplex_after_i = cdis[i+1]
        if simplex_after_i.dimension == simplex_i.dimension:
            assert( simplex_i.diameter <= simplex_after_i.diameter )
        else:
            assert(simplex_after_i.dimension == simplex_i.dimension+1)
    
if __name__ == '__main__':
    main()