"""
Mooney-Rivlin hyperelastic material — plane stress CST element.

Strain energy: W = C1*(I1 - 3) + C2*(I2 - 3)
Plane stress with incompressibility assumption (sigma_33 = 0 to find pressure).

Provides:
  - elem_fint_KT: element internal force + numerical tangent stiffness

Refactored from fem_hyperelastic.py.
"""

import numpy as np
from utils.element import tri_area_and_grad


def _compute_pk1(F, C1, C2):
    """
    Compute first Piola-Kirchhoff stress P from deformation gradient F.

    Uses plane-stress Mooney-Rivlin with 3D invariants (C33 = 1 for
    in-plane formulation) and pressure determined from sigma_33 = 0.

    Parameters
    ----------
    F : (2, 2) array
        In-plane deformation gradient.
    C1, C2 : float
        Mooney-Rivlin material constants.

    Returns
    -------
    P : (2, 2) array
        First Piola-Kirchhoff stress tensor (in-plane).
    """
    J = np.linalg.det(F)
    B = F @ F.T  # left Cauchy-Green

    # 3D invariant with B33 = 1 (plane strain in thickness direction)
    I1 = B[0, 0] + B[1, 1] + 1.0

    # Pressure from sigma_33 = 0 condition
    p = 2.0 * C1 + 2.0 * C2 * (I1 - 1.0)

    # In-plane Cauchy stress
    sigma = -p * np.eye(2) + 2.0 * C1 * B + 2.0 * C2 * (I1 * B - B @ B)

    # PK1: P = J * sigma * F^{-T}
    P = J * sigma @ np.linalg.inv(F).T
    return P


def _elem_fint_only(coords_ref, u_elem, C1, C2):
    """
    Internal force vector for one element.

    Parameters
    ----------
    coords_ref : (3, 2) array
        Reference nodal coordinates.
    u_elem : (6,) array
        Element DOF displacements.
    C1, C2 : float

    Returns
    -------
    f_int : (6,) array
    """
    A0, dNdX, dNdY = tri_area_and_grad(coords_ref)
    if A0 < 1e-14:
        return np.zeros(6)

    # Build BX matrix (4x6) mapping u_elem -> [dux/dX, dux/dY, duy/dX, duy/dY]
    b = np.array([dNdX[0], dNdX[1], dNdX[2]])
    c = np.array([dNdY[0], dNdY[1], dNdY[2]])

    BX = np.zeros((4, 6))
    for i in range(3):
        BX[0, 2 * i] = b[i]      # dux/dX
        BX[1, 2 * i] = c[i]      # dux/dY
        BX[2, 2 * i + 1] = b[i]  # duy/dX
        BX[3, 2 * i + 1] = c[i]  # duy/dY

    # Displacement gradients
    grad_u = BX @ u_elem  # [dux/dX, dux/dY, duy/dX, duy/dY]

    # Deformation gradient F = I + grad_u
    F = np.array([
        [1.0 + grad_u[0], grad_u[1]],
        [grad_u[2],       1.0 + grad_u[3]]
    ])

    P = _compute_pk1(F, C1, C2)
    P_vec = np.array([P[0, 0], P[1, 0], P[0, 1], P[1, 1]])

    return A0 * BX.T @ P_vec


def elem_fint_KT(coords_ref, u_elem, C1, C2, eps=1e-7):
    """
    Element internal force and tangent stiffness (numerical differentiation).

    Parameters
    ----------
    coords_ref : (3, 2) array
        Reference nodal coordinates.
    u_elem : (6,) array
        Element DOF displacements.
    C1, C2 : float
        Mooney-Rivlin constants.
    eps : float
        Finite difference step for numerical tangent.

    Returns
    -------
    f_int : (6,) array
        Element internal force vector.
    K_T : (6, 6) array
        Element tangent stiffness matrix.
    """
    f_int = _elem_fint_only(coords_ref, u_elem, C1, C2)

    # Numerical tangent via central differences
    K_T = np.zeros((6, 6))
    for k in range(6):
        u_p = u_elem.copy()
        u_m = u_elem.copy()
        u_p[k] += eps
        u_m[k] -= eps
        fp = _elem_fint_only(coords_ref, u_p, C1, C2)
        fm = _elem_fint_only(coords_ref, u_m, C1, C2)
        K_T[:, k] = (fp - fm) / (2.0 * eps)

    return f_int, K_T
