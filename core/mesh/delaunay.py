"""
Delaunay triangulation with hole filtering.

Used when node positions are already known (e.g. from dataset)
and we just need connectivity.

File location: elasticity_api/mesh/delaunay.py
"""

import numpy as np
from scipy.spatial import Delaunay


def delaunay_mesh(nodes, hole_mask=None):
    """
    Triangulate a 2D point cloud via Delaunay, optionally filtering
    elements whose all 3 nodes lie in a hole.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    hole_mask : (n_nodes,) bool array or None
        True for nodes identified as hole-interior. Elements with
        all 3 nodes masked are removed.

    Returns
    -------
    elements : (n_elem, 3) int array
    """
    tri = Delaunay(nodes)
    elements = tri.simplices

    if hole_mask is not None:
        hole_mask = np.asarray(hole_mask, dtype=bool)
        elements = np.array([t for t in elements if hole_mask[t].sum() < 3])

    return elements


def detect_hole_mask(nodes, disp_mag=None, outer_tol=0.02):
    """
    Heuristic hole detection from the dataset convention:
    interior nodes with large displacement magnitude are hole-boundary.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
    disp_mag : (n_nodes,) array or None
        Displacement magnitude. If None, returns all-False mask.
    outer_tol : float
        Tolerance for identifying outer boundary nodes.

    Returns
    -------
    is_hole : (n_nodes,) bool array
    """
    if disp_mag is None:
        return np.zeros(len(nodes), dtype=bool)

    is_outer = (
        (nodes[:, 0] < nodes[:, 0].min() + outer_tol) |
        (nodes[:, 0] > nodes[:, 0].max() - outer_tol) |
        (nodes[:, 1] < nodes[:, 1].min() + outer_tol) |
        (nodes[:, 1] > nodes[:, 1].max() - outer_tol)
    )
    threshold = disp_mag.mean() + disp_mag.std()
    is_hole = (~is_outer) & (disp_mag > threshold)
    return is_hole
