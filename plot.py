"""
This file includes functions which use plotly to visualize data and related 
persistent (relative) homology computations. 

Many of these functions either use or wrap functions in the Open Applied 
Topology Library: https://openappliedtopology.github.io/
"""

import oat_python as oat
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import scipy as sp
import random
from itertools import combinations
import generate_data as gen

### ========== global variables ==========


default_color = "white"
highlight_color = "red"
paint_colors = ["red", "green", "blue"]


### ========== for creating plots with plotly.graph_objects ==========


# create a 2D scatter for the 0-skeleton of a point cloud 
def scatter_2d_zero_skeleton(xvals, yvals, mode, colors):
    """
    Returns the 2D scatter of a point cloud in the Euclidean plane. Equivalently, the 0-skeleton of a simplicial complex. 
    
    :param xvals: a 1D array for x-coordinates of each point in the cloud. 
    :param yvals: a 1D array for y-coordinates of each point in the cloud. 
    :param mode: a string ]designating the 'mode' field of `ploty.graph_objects.scatter()`
    :param colors: a 1D array assigning a colors to each point of the 2D scatter. Specifically, `colors[i]` is the color of the point given by `(xvals[i], yvals[i])`. 
    
    NOTE: The `colors` array allows us to designate a subspace of the 0-skeleton! 
    """
    
    if colors == None: 
        return go.Scatter(
            x=xvals, 
            y=yvals, 
            mode=mode, 
        )
    else:
        return go.Scatter(
            x=xvals, 
            y=yvals, 
            mode=mode, 
            marker=dict(opacity=1, size=4, color=colors), 
        )

# create the 2D scatter for the 1-skeleton of a point cloud using a dissimilarity matrix and maximal diameter for edges
def scatter_2d_one_skeleton(coo, dissimilarity_matrix, dissimilarity_max):
    """
    Returns an array of 2D scatters. Each element of this array encodes an edge of the 1-skeleton of a point cloud given by `coo`. 
    
    :param coo: the points of the underlying point cloud. 
    :param dissimilarity_matrix: a dissimilarity matrix indexed by the points of `coo`, thus encoding all pairwise distances between points in `coo`. 
    :param dissimilarity_max: the maximal diameter for edges to be added to the 2D scatter that is returned. 
    
    We iterate over entries of `dissimilarity_matrix` and get the vertices associated with structural nonzero entries. Next, using 
    the function `oat.oat_python.plot.trace_2d_for_edge`, we generate each individual edge. Check documentation for this function if its use is not clear here.
    """
    edges_to_return = []
    counter = 1
    nedges = 0
    for i in range(len(coo)):                    
        for j in range(counter, len(coo)):
            current_entry = dissimilarity_matrix[i,j]
            if current_entry != 0 and current_entry <= dissimilarity_max + 0.0000001:
                current_edge_trace = oat.oat_python.plot.trace_2d_for_edge([i,j], coo)
                showlegend = nedges == 0 # --> use this if we want to have a legend! 
                current_edge_trace.update(line=dict(color="white", width=0.75), showlegend=False, name="1-skeleton")
                edges_to_return.append(current_edge_trace)
                nedges = nedges + 1
        counter = counter + 1
    return edges_to_return

# create the 2D scatter for the 2-skeleton of a point cloud using a dissimilarity matrix and maximal diameter for edges
def scatter_2d_two_skeleton(coo, dissimilarity_matrix, dissimilarity_max):
    """
    Returns an array of 2D scatters. Each element of this array encodes a triangle of the 2-skeleton of a point cloud given by `coo`. 
    
    :param coo: the points of the underlying point cloud. 
    :param dissimilarity_matrix: a dissimilarity matrix indexed by the points of `coo`, thus encoding all pairwise distances between points in `coo`. 
    :param dissimilarity_max: the maximal diameter for edges to be added to the 2D scatter that is returned. 
    
    We iterate over all unique triples of indices in `coo` and, if the triple gives a 2-simplex with diameter leq to the dissimilarity threshold, 
    we use the function `oat.oat_python.plot.trace_2d_for_triangle` to plot the triangle 
    """
    triangles_to_return = []
    triples = list( combinations(range(len(coo)), 3) )  
    ntriangles = 0
    for triple in triples:                              
        pairs = list( combinations(triple, 2) )         
        pairwise_diameters = [dissimilarity_matrix[pair[0], pair[1]] for pair in pairs] 
        if np.min(pairwise_diameters) != 0 and np.max(pairwise_diameters) <= dissimilarity_max + 0.0000001:  
            current_triangle_trace = oat.oat_python.plot.trace_2d_for_triangle(triangle=triple, points=coo)
            showlegend = ntriangles == 0 # --> use this if we want to have a legend! 
            current_triangle_trace.update(line=dict(color="white", width=0.75), showlegend=False, name="2-skeleton")
            triangles_to_return.append(current_triangle_trace)
            ntriangles = ntriangles + 1
    return triangles_to_return


### ========== for coloring subcomplex simplices for the paint palette examples ==========


# color the edges of a 1-skeleton according to subspace membership 
def generate_color_array_for_one_skeleton(coo, dissimilarity_matrix_full, dissimilarity_matrix_sub, dissimilarity_max, subspaces_array, colors_array): 
    """
    Returns a 1D array relating the edges of a simplicial complex to their colors in a 2D scatter generated by `plotly.graph_objects.scatter()`. 
    
    :param coo: the points of the underlying point cloud. 
    :param dissimilarity_matrix_full: a dissimilarity matrix indexed by the points of `coo`, thus encoding all pairwise distances between points in `coo`. 
    :param dissimilarity_matrix_sub: a dissimilarity matrix indexed by certain points of `coo` which together give a subspace. 
    :param dissimilarity_max: the maximal diameter for edges. 
    :param subspaces_array: an array that relates a point in `coo` to a given subspace. Particularly useful in the case that the subspace given by `dissimilarity_matrix_sub` is given by a union of disjoint subspaces. 
    :param colors_array: an array that maps `subspaces_array[i]` to its color, given by `colors_array[i]`. 
    
    NOTE: the dissimilarity matrices should be generated by `matrices_from_nested_clouds()`. 
    NOTE: in the case that the subcloud should only be one color, then `subspaces_arrary` and `colors_array` should each be length 1 arrays. 
    NOTE: notice that this function is VERY similar to `scatter_2d_one_skeleton()`. This is intentional, as the two are designed to be used togehter if desired! 
    """
    colors_to_return = []
    counter = 1
    for i in range(len(coo)):                    
        for j in range(counter, len(coo)):
            current_entry = dissimilarity_matrix_full[i,j]
            if current_entry != 0 and current_entry <= dissimilarity_max + 0.0000001:
                edge_color = "white"
                if dissimilarity_matrix_sub[i,j] != 0:
                    subspace_of_point_i = subspaces_array[i]
                    if subspace_of_point_i == subspaces_array[j]: 
                        edge_color = colors_array[subspace_of_point_i]
                colors_to_return.append(edge_color)
        counter = counter + 1
    return colors_to_return

# color the triangles of a 2-skeleton according to subspace membership 
def generate_color_array_for_two_skeleton(coo, dissimilarity_matrix_full, dissimilarity_matrix_sub, dissimilarity_max, subspaces_array, colors_array): 
    """
    Returns a 1D array relating the triangles of a simplicial complex to their colors in a 2D scatter generated by `plotly.graph_objects.scatter()`. 
    
    :param coo: the points of the underlying point cloud. 
    :param dissimilarity_matrix_full: a dissimilarity matrix indexed by the points of `coo`, thus encoding all pairwise distances between points in `coo`. 
    :param dissimilarity_matrix_sub: a dissimilarity matrix indexed by certain points of `coo` which together give a subspace. 
    :param dissimilarity_max: the maximal diameter for edges. 
    :param subspaces_array: an array that relates a point in `coo` to a given subspace. Particularly useful in the case that the subspace given by `dissimilarity_matrix_sub` is given by a union of disjoint subspaces. 
    :param colors_array: an array that maps `subspaces_array[i]` to its color, given by `colors_array[i]`. 
    
    NOTE: the dissimilarity matrices should be generated by `matrices_from_nested_clouds()`. 
    NOTE: in the case that the subcloud should only be one color, then `subspaces_arrary` and `colors_array` should each be length 1 arrays. 
    NOTE: notice that this function is VERY similar to `scatter_2d_two_skeleton()`. This is intentional, as the two are designed to be used togehter if desired! 
    """
    colors_to_return = []
    triples = list( combinations(range(len(coo)), 3) ) 
    # for each unique combination of three vertices ...
    for triple in triples:    
        # check if it is a triangle existing in the complex                          
        pairs = list( combinations(triple, 2) )      
        pairwise_diameters_full = [dissimilarity_matrix_full[pair[0], pair[1]] for pair in pairs] 
        if np.min(pairwise_diameters_full) != 0 and np.max(pairwise_diameters_full) <= dissimilarity_max + 0.0000001: 
            triangle_color = "white"
            # check if it is a triangle in the subcomplex
            pairwise_diameters_sub = [dissimilarity_matrix_sub[pair[0], pair[1]] for pair in pairs]
            if np.min(pairwise_diameters_sub) != 0:
                # check if all of its edges are in the same "paint blob" of the subcomplex
                subspace_of_point = subspaces_array[triple[0]]
                if subspace_of_point == subspaces_array[triple[1]] and subspace_of_point == subspaces_array[triple[2]]:
                    triangle_color = colors_array[subspace_of_point] 
            colors_to_return.append(triangle_color)
    return colors_to_return

# generate subcloud array for paint palette examples
def generate_subcloud_indices_array_for_paint_palletes(coo, sample_centers, radius):
    """
    Generates an array of indices which are used to identify members of a subcloud given `coo`. 
    
    :param coo: a point cloud.
    :param sample_centers: an array of unique indices giving a random sample of points in `coo`.
    :param raidus: a real number which constitues a disk around the points of `sample_centers`. Any points in the disks D(sample_centers[i], radius) is a member of the subcloud. 
    """
    subcloud_indices = []    # an array which can be used to generate a dissimilarity matrix for the subcloud
    disk_membership = []     # maps each point of `coo` to the disk which it is a member of, or None
    current_index = 0
    for pt in coo:           # loop over each `pt in coo`, determine which disk it may be a part of, and append it to `subcloud_indices` as necessary
        distances_from_centers = [np.linalg.norm(pt - coo[center_pt]) for center_pt in sample_centers]
        minimal_distance = min(distances_from_centers)
        if minimal_distance < radius: 
            disk_membership.append(distances_from_centers.index(minimal_distance))
            subcloud_indices.append(current_index)
        else: 
            disk_membership.append(None)
        current_index = current_index + 1
        
    return subcloud_indices, disk_membership

### ========== for creating paint palette figures with plotly ==========
# NOTE: these use all of the helper functions above! 
    
def create_paint_palette(radius, dx, dy, epsilon, blob_radius, centers_of_blobs, randomize_grid, title):
    """
    Use the `plotly.graph_objects` library and hepler functions in `oat_python` to generate example data for PRH. 
    The returned figure is a (possibly randomized) grid of data points. 
    
    :param radius: the radius of the grid.
    :param dx: the horizontal distance between points in the grid.
    :param dy: the vertical distance between points in the grid.
    :param epsilon: a real number which determines the maximal diameter of a simplex in the returned simplicial complex figure.
    :param blob_radius: a real number for the radius of each paint-blob subcomplex. 
    :param centers_of_blobs: an array of integers mapping to an index of the returned dissimilarity matrices, which can be used to alter the locality of the paint-blob subcomplexes. 
    :param randomize_grid: a boolean determining if the grid is randomized.
    :param title: the title of the plot to create.  
    
    NOTE: decreasing `dx` and `dy` creates a grid with a finer sample (more points). 
    
    ## Returns:
    
    - `coo`: a list of (x,y) points for the 0-simplices of the full complex. This is the raw data, and is useful for working with the figure after it is generated. 
    - `dissimilarity_matrix_full_space` and `dissimilarity_matrix_subspace`, from which we construct a simplicial complex using `coo`
    - `fig`: a plotly figure for the simplicial complex with the paint blob subcomplexes highlighted
    - `subcloud_indices`: an array of integers such that any `i` in subcloud indices gives a point of the subcomplex at `coo[i]`
    - `blob_membership`: an array mapping points of the full complex to their paint blob, or None. 
    """
    # GENERATE DATA
    if randomize_grid:
        cloud_x, cloud_y = gen.randomized_grid_sample_points(radius=radius, dx=dx, dy=dy)
    else:
        cloud_x, cloud_y = gen.grid_sample_points(radius=radius, dx=dx, dy=dy)
    coo = np.array(list(zip(cloud_x, cloud_y)))
    npoints = len(coo)
    # PAINT BLOBS
    if centers_of_blobs == None: 
        nblobs = len(paint_colors)
        centers_of_paint_blobs = random.sample(range(npoints), nblobs) # holds the indices of the points at the center of each paint blob 
        subcloud_indices, blob_membership = generate_subcloud_indices_array_for_paint_palletes(coo, centers_of_paint_blobs, blob_radius)
    else:
        subcloud_indices, blob_membership = generate_subcloud_indices_array_for_paint_palletes(coo, centers_of_blobs, blob_radius)
    # DISSIMILARITY MATRICES
    dissimilarity_matrix_full_space, dissimilarity_matrix_subspace = gen.matrices_from_nested_clouds(
        coo, 
        subcloud_indices, 
        epsilon + 0.0000001
    )
    # CREATE THE FIGURE
    fig = go.Figure()
    fig.update_layout(title=dict(text=title), template="plotly_dark", height=800, width=850)
    # generate triangles of the 2-skeleton
    triangle_traces = scatter_2d_two_skeleton(coo, dissimilarity_matrix_full_space, epsilon)
    triangle_colors = generate_color_array_for_two_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, epsilon, blob_membership, paint_colors)
    for i in range(len(triangle_traces)): 
        current_trace = triangle_traces[i]
        current_trace.update(legendgroup="triangles", fillcolor=triangle_colors[i])
        fig.add_trace(current_trace)
    # generate edges of the 1-skeleton
    edge_traces = scatter_2d_one_skeleton(coo, dissimilarity_matrix_full_space, epsilon)
    edge_colors = generate_color_array_for_one_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, epsilon, blob_membership, paint_colors)
    for i in range(len(edge_traces)): 
        current_trace = edge_traces[i]
        current_trace.update(legendgroup="edges", line=dict(color=edge_colors[i], width=0.75))
        fig.add_trace(current_trace)
    # generate vertices of the 0-skeleton
    vertex_colors = [paint_colors[blob_membership[i]] if i in subcloud_indices else default_color for i in range(npoints)]
    fig.add_trace(go.Scatter(x=cloud_x, y=cloud_y, mode="markers", marker=dict(opacity=1, size=5, color=vertex_colors), name="0-skeleton", showlegend=False))
    
    return coo, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, fig, subcloud_indices, blob_membership

def plot_point_cloud_with_subcloud_2d(title, cloud_x, cloud_y, subcloud_indices, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, dissimilarity_max): 
    """
    Plot a simplicial complex from a point cloud and highlight a subcomplex defined by a given subcloud. 

    :param title: the title of the plot to create 
    :param cloud_x: x coordinates of the points to plot
    :param cloud_y: y coordinates of the points to plot
    :param npoints: number of points to plot
    :param subcloud_indices: a 1D array describing indices of points in the subcloud. For example, if (a,b) is in the subcloud and cloud_x[i] = a, then subcloud_indices will contain i. 
    :param dissimilarity_matrix_full_space: Euclidean distance matrix for the point cloud
    :param dissimilarity_matrix_subspace: Euclidean distance matrix for the subcloud
    :param dissimilarity_max: maximal dissimilarity threshold used when constructing the dissimilarity matrices
    """
    
    coo = np.array(list(zip(cloud_x, cloud_y)))
    blob_membership = [1 if i in subcloud_indices else 0 for i in range(len(coo))]
    fig = go.Figure()
    fig.update_layout(title=dict(text=title), template="plotly_dark", height=800, width=850)
    # generate triangles of the 2-skeleton
    triangle_traces = scatter_2d_two_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_max)
    triangle_colors = generate_color_array_for_two_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, dissimilarity_max, blob_membership, paint_colors)
    for i in range(len(triangle_traces)): 
        current_trace = triangle_traces[i]
        current_trace.update(legendgroup="triangles", fillcolor=triangle_colors[i])
        fig.add_trace(current_trace)
    # generate edges of the 1-skeleton
    edge_traces = scatter_2d_one_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_max)
    edge_colors = generate_color_array_for_one_skeleton(coo, dissimilarity_matrix_full_space, dissimilarity_matrix_subspace, dissimilarity_max, blob_membership, paint_colors)
    for i in range(len(edge_traces)): 
        current_trace = edge_traces[i]
        current_trace.update(legendgroup="edges", line=dict(color=edge_colors[i], width=0.75))
        fig.add_trace(current_trace)
    # generate vertices of the 0-skeleton
    vertex_colors = [highlight_color if i in subcloud_indices else default_color for i in range(len(coo))]
    fig.add_trace(go.Scatter(x=cloud_x, y=cloud_y, mode="markers", marker=dict(opacity=1, size=5, color=vertex_colors), name="0-skeleton", showlegend=False))
    
    return fig

### ========== for plotting chains ==========


def plot_chain(edges, dimension, coo, fig, legend_group, name, line_color):
    """
    Plot/highlight a chain on a simplicial complex given its simplices. 
    
    :param edges: The edges of the chain to draw. In general, this is a list of n-simplices which form the n-chain. 
    :param dimension: The dimension of the chain, usually 0, 1 or 2. 
    :param coo: The points of the underlying point cloud. 
    :param fig: The plotly figure to use. 
    :param legend_group: The `legendgroup` of the plotly figure. 
    :param name: The `name` to be used in the legend of the figure. 
    :param line_color: The color of the trace. 
    """
    if dimension == 0: 
        traces = [ go.Scatter(x=[coo[edge[0]][0]], y=[coo[edge[0]][1]], mode='markers') for edge in edges ]
    elif dimension == 1: 
        traces = [ oat.oat_python.plot.trace_2d_for_edge(edge=edge, points=coo) for edge in edges ] 
    elif dimension == 2: 
        traces = [ oat.oat_python.plot.trace_2d_for_triangle(triangle=edge, points=coo) for edge in edges ]
    else: 
        raise Exception("\n\n Ensure that the chain you are attempting to plot has dimension in {0, 1, 2}. \n\n")
    for trace_number, trace in enumerate( traces ):
        showlegend = trace_number == 0
        trace.update(showlegend=showlegend, legendgroup=legend_group, opacity=0.8, name=name, line=dict(color=line_color, width=5))
        fig.add_trace(trace)
        
### ========== for plotting barcodes ==========

def plot_barcode(births, deaths, dimensions, title):
    """
    Plot a PRH barcode given arrays of `births`, `deaths` and `dimensions`. The user must 
    also provide a title for the resulting plot. 
    
    Returns a plotly figure contianing the barcode. 
    
    """

    # assert data is provided in correct format
    if not (len(births) == len(deaths) == len(dimensions)):
        raise ValueError("Arrays must have the same length")

    # sort by dimension and then birth time
    order = np.lexsort((births, deaths, dimensions))
    births = births[order]
    deaths = deaths[order]
    dimensions = dimensions[order]

    colors = {
        0: "#1f77b4",
        1: "#ff7f0e",
        2: "#2ca02c",
        3: "#d62728",
        4: "#9467bd",
    }
    fig = go.Figure()

    # add one invisible trace per dimension to create the legend
    for dim in np.unique(dimensions):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=colors.get(dim, "#555555"),width=6,),name=f"H{dim}"))

    # add barcode segments
    for i, (b, d, dim) in enumerate(zip(births, deaths, dimensions)):
        fig.add_shape(type="line", x0=b, x1=d, y0=i, y1=i, line=dict(color=colors.get(dim, "#555555"),width=5,))

    fig.update_layout(
        title=title,
        xaxis_title="filtration value",
        yaxis=dict(
            tickmode="array",
            tickvals=np.arange(len(births)),
            ticktext=[
                f"" for _ in dimensions
            ],
        ),
        template="plotly_white",
        hovermode="closest",
        width=600,
        height=400
    )

    return fig
        
# a quick test script
def main(): 
    # parameters to generate the data
    r = 0.5
    dx = 0.25
    dy = 0.25
    epsilon = np.sqrt(2 * dx**2) 
    blob_radius = epsilon + 0.0000001

    # generate the data
    coo_grid, dissimilarity_matrix_full_space_grid, dissimilarity_matrix_subspace_grid, fig_grid, subcloud_indices_grid, blob_membership_grid = create_paint_palette(r, dx, dy, epsilon, blob_radius, [4,20,24], False, "Demo Grid")

    # show the result
    fig_grid.show()
    
if __name__ == '__main__':
    main()