"""
Main user-facing API for the elasticity solver.

File location: elasticity_api/api.py

Usage
-----
    solver = ElasticitySolver(nodes, elements, material='mooney_rivlin',
                              mat_params={'C1': 1.863e5, 'C2': 9.79e3},
                              bc=bc)
    u, info = solver.solve(warm_start_u=u_ml, solver='scipy')
"""

import numpy as np
from scipy.spatial import Delaunay

from materials.linear_elastic import elem_fint_K
from materials.mooney_rivlin import elem_fint_KT
from solver.boundary import BoundaryConditions
from solver.newton_raphson import solve_newton_raphson


_MATERIAL_MAP = {
    'linear': elem_fint_K,
    'linear_elastic': elem_fint_K,
    'mooney_rivlin': elem_fint_KT,
    'hyperelastic': elem_fint_KT,
}


class ElasticitySolver:
    """
    Top-level elasticity solver API.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
        Nodal coordinates.
    elements : (n_elem, 3) int array or None
        Triangular connectivity. If None, auto-generated via Delaunay.
    material : str
        'linear' / 'linear_elastic' or 'mooney_rivlin' / 'hyperelastic'.
    mat_params : dict
        Material constants. E.g. {'E': 1e5, 'nu': 0.3} or
        {'C1': 1.863e5, 'C2': 9.79e3}.
    bc : BoundaryConditions
        Boundary conditions object.
    hole_mask : (n_nodes,) bool array or None
        If provided and elements is None, filters Delaunay triangles
        whose all 3 nodes are in the hole.
    """

    def __init__(self, nodes, elements, material, mat_params, bc,
                 hole_mask=None):
        self.nodes = np.asarray(nodes, dtype=float)
        self.mat_params = mat_params
        self.bc = bc

        # Material function
        if material not in _MATERIAL_MAP:
            raise ValueError(f"Unknown material '{material}'. "
                             f"Choose from: {list(_MATERIAL_MAP.keys())}")
        self.material_func = _MATERIAL_MAP[material]
        self.material_name = material

        # Elements: auto-mesh if not provided
        if elements is None:
            self.elements = self._auto_mesh(hole_mask)
        else:
            self.elements = np.asarray(elements, dtype=int)

        # Fix isolated nodes
        bc.fix_isolated_nodes(self.elements)

        # DOFs
        self.n_nodes = len(self.nodes)
        self.n_dof = 2 * self.n_nodes
        self.fixed_dofs, self.free_dofs = bc.get_dofs()

        # External force
        self.f_ext = bc.build_f_ext(self.elements)

    def _auto_mesh(self, hole_mask=None):
        """Delaunay triangulation with optional hole filtering."""
        tri = Delaunay(self.nodes)
        elements = tri.simplices
        if hole_mask is not None:
            elements = np.array([t for t in elements
                                 if hole_mask[t].sum() < 3])
        return elements

    def solve(self, warm_start_u=None, solver='scipy',
              petsc_opts=None, tol=1e-6, max_iter=30, verbose=True):
        """
        Run the solver.

        Parameters
        ----------
        warm_start_u : (n_nodes, 2) or (n_dof,) array or None
            Initial displacement guess (from ML model or previous solve).
        solver : str
            'scipy' or 'petsc'.
        petsc_opts : dict or None
            PETSc solver options (solver_type, precond, rtol, etc.).
        tol : float
            NR convergence tolerance.
        max_iter : int
            Max NR iterations.
        verbose : bool

        Returns
        -------
        u : (n_nodes, 2) array
            Displacement field.
        info : dict
            Solver info: iterations, converged, residual_history.
        """
        # Initial guess
        if warm_start_u is not None:
            u0 = np.asarray(warm_start_u, dtype=float).ravel()
            if len(u0) != self.n_dof:
                raise ValueError(f"warm_start_u has {len(u0)} DOFs, "
                                 f"expected {self.n_dof}")
            u0[self.fixed_dofs] = 0.0
        else:
            u0 = np.zeros(self.n_dof)

        if verbose:
            label = 'Warm Start' if warm_start_u is not None else 'Cold Start'
            print(f"\n{'='*50}")
            print(f"  {self.material_name} | {label} | solver={solver}")
            print(f"  Nodes: {self.n_nodes} | Elements: {len(self.elements)}")
            print(f"  Free DOFs: {len(self.free_dofs)} | "
                  f"Fixed DOFs: {len(self.fixed_dofs)}")
            print(f"{'='*50}")

        u_flat, info = solve_newton_raphson(
            u0, self.f_ext, self.nodes, self.elements, self.free_dofs,
            self.material_func, self.mat_params,
            tol=tol, max_iter=max_iter,
            solver=solver, petsc_opts=petsc_opts,
            verbose=verbose
        )

        u = u_flat.reshape(self.n_nodes, 2)
        return u, info

    def compute_stress(self, u):
        """
        Compute element-averaged Cauchy stress from displacement field.
        (Placeholder for post-processing — to be expanded.)

        Parameters
        ----------
        u : (n_nodes, 2) array

        Returns
        -------
        stress : (n_elem, 3) array
            [sigma_xx, sigma_yy, sigma_xy] per element.
        """
        from .utils.element import tri_area_and_grad, strain_displacement_matrix
        from .materials.linear_elastic import material_matrix

        u_flat = u.ravel()
        stress = np.zeros((len(self.elements), 3))

        if self.material_name in ('linear', 'linear_elastic'):
            D = material_matrix(self.mat_params['E'], self.mat_params['nu'])
            for i, elem in enumerate(self.elements):
                coords = self.nodes[elem]
                A0, dNdX, dNdY = tri_area_and_grad(coords)
                if A0 < 1e-14:
                    continue
                B = strain_displacement_matrix(dNdX, dNdY)
                dofs = np.array([[2*n, 2*n+1] for n in elem]).ravel()
                strain = B @ u_flat[dofs]
                stress[i] = D @ strain

        return stress
