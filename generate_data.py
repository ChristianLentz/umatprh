"""
A few functions for generating simple test data. 
"""

import oat_python as oat
import numpy as np
import random
import scipy

def grid_sample_points(radius, dx, dy): 
    """
    Given a radius, generate a grid of sample points centered at the origin in the Euclidean plane such that 
    for any two points (x1, y1), (x2, y2) in the sample we have <x1-x2, y1-y2> = n*<dx, dy> for some integer n.  
    Thus, the size of the sample is determined by the radius of the disk and the basis vector <dx, dy>. 

    :param radius: the radius of the grid
    :param dx: horizontal difference between points 
    :param dy: vertical difference between points
    """
    x = np.arange(-radius, radius+dx, dx)
    y = np.arange(-radius, radius+dy, dy)
    grid_x, grid_y = np.meshgrid(x, y)
    
    return grid_x.ravel(), grid_y.ravel()

def randomized_grid_sample_points(radius, dx, dy, stack=False): 
    """
    Generate a sample of points using `oat.point_cloud.grid_sample_points()` and slightly randomize them. 

    :param radius: the radius of the grid
    :param dx: horizontal difference between points of the unrandomized grid sampled
    :param dy: vertical difference between points of the unrandomized grid sampled
    """
    x_unrandomized, y_unrandomized = grid_sample_points(radius, dx, dy)
    x_randomized = [x + random.uniform(-dx*0.4, dx*0.4) for x in x_unrandomized]
    y_randomized = [y + random.uniform(-dy*0.4, dy*0.4) for y in y_unrandomized]
    
    if stack: 
        return np.column_stack( (x_randomized, y_randomized) )
    else: 
        return x_randomized, y_randomized

def circle_sample(n_points=100, radius=1.0, band_width=0.05):
    """
    Generate random points near the unit circle.
    
    # Parameters
    
    :param n_points: Number of points to generate.
    :param radius: Radius of circle (defaults to  1.0).
    :param band_width: Half-width of the band around the radius.
    
    # Returns
    
    np.ndarray: Array of shape (n_points, 2) with x, y coordinates.
    """
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if band_width < 0:
        raise ValueError("band_width must be non-negative.")

    # random angles uniformly distributed in [0, 2pi)
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    
    # random radii in [radius - band_width, radius + band_width]
    radii = np.random.uniform(radius - band_width, radius + band_width, n_points)
    
    # return cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def matrices_from_nested_clouds( full_cloud, subcloud_indices, dissimilarity_max ):
    """
    Returns two sparse (Scipy CSR) Euclidean distance matrices, where all entries with value strictly greater than `dissimilarity_max` are dropped.

    # Arguments

    :param full_cloud: any format for a point cloud compatible with `sklearn.neighbors.radius_neighbors_graph`
    :param subcloud_indices: an array of integers indices which give points of `cloud` that constitute the "subcloud"
    :param dissimilarity_max float: a non-negative real number; all distances with value above this threshold are dropped

    # Returns 
    
    Function returns a pair of sparse dissimilarity matrices with the same shape; one constructed from `cloud`, and the other constructed using only the 
    points `cloud[subcloud_indices]`. We use the function `dissimilarity.matrix_from_cloud` to help acheive this. 
    
    # Notes 
    
    - This function does not use `dissimilarity.matrix_from_cloud` to construct the dissimilarity matrix for the points `cloud[subcloud_indices]`, as 
    this would result in two dissimilarity matrices with different shape. 
    - The dissimilarity matrix for the points `cloud[subcloud_indices]` is constructed in one of four ways, based on the length of the vector 
    `subcloud_indices`. For these notes, let `dissimilarity_full` = `dissimilarity.matrix_from_cloud`, `len(full_cloud)` = `x`, and 
    `len(subcloud_indices)` = 'y': 
    
        a) Copy `dissimilarity_full` and iteratively remove entries. This is more efficient as the size of the subcloud grows. This approach is used 
        when y/x >= 0.5. 
        
        b) Initialize empty Scipy CSR matrix and iteratively copy entries from `dissimilarity_full`. This is less efficient as the size of
        the subcloud grows. This approach is used when y/x < 0.5. 
        
        c) If `subcloud_indices` is empty, return an empty matrix with the same shape as `dissimilarity_full`. 
        
        d) If `subcloud_indices` gives a subcloud identical to the full cloud, return `dissimilairty_full` and a deep copy of it. 
    
    """
    
    dissimilarity_matrix_full_cloud = oat.oat_python.dissimilarity.sparse_matrix_for_points(points=full_cloud, max_dissimilarity=dissimilarity_max)
    x = np.shape(full_cloud)[0]
    y = len(subcloud_indices)
    
    # ensure subspace indices is a valid array of indices 
    # NOTE: we cast `subcloud_indices` to a set ... 
    # - now searching the set for a specific item is a constant time operation! 
    # - duplicate elements are dropped 
    subcloud_set = set(subcloud_indices)
    
    # CASE: invalid `subcloud_indices`
    if len(subcloud_set) != y:
        raise Exception("\n\n Provided list of subcloud indices is invalid: contains duplicate elements. \n\n") 
    
    # CASE: invalid `subcloud_indices`
    all_in_bounds = all(index < x for index in subcloud_set)
    if not all_in_bounds: 
        raise Exception("\n\n Provided list of subcloud indices is invalid: contains one or more out of bounds indices. \n\n")
    
    # NOTE: if the first two checks above (no duplicae elements, all elements in bounds) pass, then we must have y <= x ... all corresponding cases appear below
    
    # CASE: full cloud and subcloud are identical 
    if y == x: 
        return dissimilarity_matrix_full_cloud, dissimilarity_matrix_full_cloud.copy()
    
    # CASE: subcloud is empty 
    elif y == 0:
        dissimilarity_matrix_subcloud = scipy.sparse.csr_matrix((x,x))
        return dissimilarity_matrix_full_cloud, dissimilarity_matrix_subcloud 
    
    else: 
        
        # CASE: use approach (a)
        if y/x >= 0.5: 
            M = dissimilarity_matrix_full_cloud.copy()     
            # loop variables
            indices_to_remove = []
            counter = 1
            for i in range(x):                                         
                for j in range(x):                                     
                    if i not in subcloud_set or j not in subcloud_set:
                        indices_to_remove.extend( [(i,j), (j,i)] ) 
                counter+=1 
            # cast the copy of `dissimilarity_matrix_full_cloud` to a COO matrix for efficient element-wise updates
            M_coo = M.tocoo()
            # create a mask for the COO matrix which removes any entry from `indices_to_remove`
            mask = [(r, c) not in indices_to_remove for r, c in zip(M_coo.row, M_coo.col)]
            dissimilarity_matrix_subcloud = scipy.sparse.csr_matrix((M_coo.data[mask], (M_coo.row[mask], M_coo.col[mask])), shape=M.shape)
            return dissimilarity_matrix_full_cloud, dissimilarity_matrix_subcloud
        
        # CASE: use approach (b) 
        if y/x < 0.5: 
            dissimilarity_matrix_subcloud = scipy.sparse.csr_matrix((x,x)) 
            counter = 1
            for i in range(len(subcloud_indices)):                    
                for j in range(counter, len(subcloud_indices)):
                    r = subcloud_indices[i]                            
                    c = subcloud_indices[j]
                    dissimilarity_matrix_subcloud[r,c] = dissimilarity_matrix_full_cloud[r,c]
                    dissimilarity_matrix_subcloud[c,r] = dissimilarity_matrix_full_cloud[c,r]
                counter+=1
            # NOTE: we need to drop zero entries to ensure proper safety checks
            dissimilarity_matrix_subcloud.eliminate_zeros()
            # NOTE: we need to manually set diagonal entries to zero for points in the subspace 
            for k in subcloud_indices: 
                dissimilarity_matrix_subcloud[k,k] = 0
            return dissimilarity_matrix_full_cloud, dissimilarity_matrix_subcloud