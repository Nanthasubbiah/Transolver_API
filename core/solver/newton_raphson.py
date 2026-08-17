"""
Newton-Raphson nonlinear solver.

Orchestrates: assemble → check convergence → PETSc/scipy solve → update.
For linear problems, converges in 1 iteration.

File location: elasticity_api/solver/newton_raphson.py
"""

import numpy as np
from .assembler import assemble
from .petsc_solver import petsc_solve, scipy_solve


def solve_newton_raphson(u0, f_ext, nodes, elements, free_dofs,
                         material_func, mat_params,
                         tol=1e-6, max_iter=30,
                         solver='scipy', petsc_opts=None,
                         verbose=True):
    """
    Newton-Raphson iteration: K_T(u) * du = f_ext - f_int(u).

    Parameters
    ----------
    u0 : (n_dof,) array
        Initial guess (zero for cold start, ML prediction for warm start).
    f_ext : (n_dof,) array
        External force vector.
    nodes : (n_nodes, 2) array
    elements : (n_elem, 3) int array
    free_dofs : int array
        Unconstrained DOF indices.
    material_func : callable
        Element routine: (f_int_e, K_e) = func(coords, u_e, **mat_params).
    mat_params : dict
        Material parameters.
    tol : float
        Relative residual convergence tolerance.
    max_iter : int
        Maximum NR iterations.
    solver : str
        'scipy' for direct solve, 'petsc' for PETSc subprocess.
    petsc_opts : dict or None
        Options passed to petsc_solve: solver_type, precond, rtol, etc.
    verbose : bool

    Returns
    -------
    u : (n_dof,) array
        Converged displacement field.
    info : dict
        'iterations', 'converged', 'residual_history'.
    """
    u = u0.copy()
    petsc_opts = petsc_opts or {}
    residual_history = []
    linear_solve = petsc_solve if solver == 'petsc' else scipy_solve

    for it in range(max_iter):
        F_int, KT = assemble(u, nodes, elements, material_func, mat_params)
        residual = f_ext - F_int
        res_free = residual[free_dofs]
        res_norm = np.linalg.norm(res_free)

        if it == 0:
            res0 = res_norm + 1e-14
        rel_res = res_norm / res0
        residual_history.append(rel_res)

        if verbose:
            print(f"  NR iter {it:3d}: |R|={res_norm:.4e}, |R|/|R0|={rel_res:.4e}")

        if rel_res < tol:
            if verbose:
                print(f"  Converged in {it} iterations.")
            return u, {'iterations': it, 'converged': True,
                       'residual_history': residual_history}

        KT_ff = KT[free_dofs, :][:, free_dofs]
        x0_free = None
        du_free, _ = linear_solve(KT_ff, res_free, x0=x0_free, **petsc_opts)
        u[free_dofs] += du_free

    if verbose:
        print(f"  Warning: did not converge in {max_iter} iterations.")
    return u, {'iterations': max_iter, 'converged': False,
               'residual_history': residual_history}

