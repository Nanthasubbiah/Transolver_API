"""
Gmsh-based mesh generation for a unit cell with a hole.

Generates a proper triangular mesh from the rr (radii) hole boundary
description used in the elasticity dataset.

File location: elasticity_api/mesh/gmsh_mesh.py
"""

import numpy as np


def generate_unit_cell_mesh(rr, lc=0.03, center=(0.5, 0.5)):
    """
    Generate a triangular mesh for a unit square [0,1]^2 with a hole
    defined by radii at 42 equally-spaced angles.

    Parameters
    ----------
    rr : (42,) array
        Radii from center to hole boundary at each angle.
    lc : float
        Mesh element size (smaller = finer).
    center : tuple
        Center of the hole (default (0.5, 0.5)).

    Returns
    -------
    nodes : (n_nodes, 2) array
    elements : (n_elem, 3) int array
    boundary : dict
        {'top': set, 'bottom': set, 'left': set, 'right': set}
        Node index sets for each boundary edge.

    Raises
    ------
    ImportError
        If gmsh is not installed.
    """
    try:
        import gmsh
    except ImportError:
        raise ImportError("gmsh not installed. Run: pip install gmsh")

    angles = np.linspace(0, 2 * np.pi, len(rr), endpoint=False)
    hx = center[0] + rr * np.cos(angles)
    hy = center[1] + rr * np.sin(angles)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("unit_cell")

    # Outer square corners
    p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p2 = gmsh.model.geo.addPoint(1, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(1, 1, 0, lc)
    p4 = gmsh.model.geo.addPoint(0, 1, 0, lc)

    # Hole boundary points
    hole_pts = [gmsh.model.geo.addPoint(hx[i], hy[i], 0, lc)
                for i in range(len(rr))]

    # Outer edges
    l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
    l2 = gmsh.model.geo.addLine(p2, p3)  # right
    l3 = gmsh.model.geo.addLine(p3, p4)  # top
    l4 = gmsh.model.geo.addLine(p4, p1)  # left

    # Hole spline (closed)
    hole_spline = gmsh.model.geo.addSpline(hole_pts + [hole_pts[0]])

    # Surface with hole
    outer_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    hole_loop = gmsh.model.geo.addCurveLoop([hole_spline])
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole_loop])

    # Physical groups for BC identification
    gmsh.model.geo.addPhysicalGroup(1, [l1], tag=1)  # bottom
    gmsh.model.geo.addPhysicalGroup(1, [l2], tag=2)  # right
    gmsh.model.geo.addPhysicalGroup(1, [l3], tag=3)  # top
    gmsh.model.geo.addPhysicalGroup(1, [l4], tag=4)  # left
    gmsh.model.geo.addPhysicalGroup(2, [surface], tag=5)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Extract nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    coords = coords.reshape(-1, 3)[:, :2]
    node_map = {tag: i for i, tag in enumerate(node_tags)}

    # Extract triangles
    _, _, elem_nodes = gmsh.model.mesh.getElements(dim=2)
    elements = np.array([node_map[t] for t in elem_nodes[0]]).reshape(-1, 3)

    # Extract boundary nodes
    def get_boundary_nodes(phys_tag):
        entities = gmsh.model.getEntitiesForPhysicalGroup(1, phys_tag)
        node_set = set()
        for e in entities:
            tags, _, _ = gmsh.model.mesh.getNodes(1, e)
            for t in tags:
                if t in node_map:
                    node_set.add(node_map[t])
        return node_set

    boundary = {
        'bottom': get_boundary_nodes(1),
        'right':  get_boundary_nodes(2),
        'top':    get_boundary_nodes(3),
        'left':   get_boundary_nodes(4),
    }

    gmsh.finalize()
    return coords, elements, boundary
