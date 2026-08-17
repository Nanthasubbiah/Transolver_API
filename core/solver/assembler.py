"""
Global assembler for 2D triangular FEM.

Loops over elements, calls the material routine, and accumulates
element contributions into global sparse K and global f_int vector.
Material-agnostic: works with both linear_elastic and mooney_rivlin.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def assemble(u_vec, nodes, elements, material_func, mat_params):
    """
    Assemble global internal force vector and tangent stiffness matrix.

    Parameters
    ----------
    u_vec : (n_dof,) array
        Current displacement vector (2 DOFs per node: [u1x, u1y, u2x, ...]).
    nodes : (n_nodes, 2) array
        Reference nodal coordinates.
    elements : (n_elem, 3) int array
        Element connectivity (node indices).
    material_func : callable
        Element-level routine with signature:
            f_int_e, K_e = material_func(coords, u_elem, **mat_params)
        Must return (6,) force and (6,6) stiffness.
    mat_params : dict
        Material parameters passed as kwargs to material_func.
        E.g. {'E': 1e5, 'nu': 0.3} or {'C1': 1.863e5, 'C2': 9.79e3}.

    Returns
    -------
    F_int : (n_dof,) array
        Global internal force vector.
    K : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
        Global tangent stiffness matrix.
    """
    n_dof = len(u_vec)
    F_int = np.zeros(n_dof)
    K = lil_matrix((n_dof, n_dof))

    for elem in elements:
        coords = nodes[elem]
        dofs = np.array([[2 * n, 2 * n + 1] for n in elem]).ravel()
        u_e = u_vec[dofs]

        fi, Ki = material_func(coords, u_e, **mat_params)

        F_int[dofs] += fi
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += Ki[i, j]

    return F_int, csr_matrix(K)


def apply_neumann(f_ext, nodes, elements, boundary_edges, tractions):
    """
    Add traction contributions to the external force vector.

    Parameters
    ----------
    f_ext : (n_dof,) array
        External force vector (modified in place).
    nodes : (n_nodes, 2) array
        Nodal coordinates.
    elements : (n_elem, 3) int array
        Element connectivity (used to find boundary edges).
    boundary_edges : list of dict
        Each dict has:
            'nodes': set of node indices on this boundary
            'traction': (2,) array [tx, ty] force per unit length
    """
    for bc in boundary_edges:
        bc_nodes = bc['nodes']
        tx, ty = bc['traction']
        for elem in elements:
            for i in range(3):
                n1, n2 = elem[i], elem[(i + 1) % 3]
                if n1 in bc_nodes and n2 in bc_nodes:
                    L = np.linalg.norm(nodes[n1] - nodes[n2])
                    f_ext[2 * n1]     += tx * L / 2
                    f_ext[2 * n1 + 1] += ty * L / 2
                    f_ext[2 * n2]     += tx * L / 2
                    f_ext[2 * n2 + 1] += ty * L / 2

    return f_ext
