"""
Linear elastic material — plane stress CST element.

Given Young's modulus E and Poisson's ratio nu, computes:
  - D matrix (3x3 constitutive matrix)
  - Element stiffness K_e = A * B^T D B
  - Element internal force f_int = K_e @ u_e

Since D is constant, K_e is assembled once and reused.
"""

import numpy as np
from utils.element import tri_area_and_grad, strain_displacement_matrix


def material_matrix(E, nu):
    """
    Plane stress constitutive matrix D (3x3).

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    D : (3, 3) array
    """
    coeff = E / (1.0 - nu ** 2)
    D = coeff * np.array([
        [1.0,  nu,  0.0],
        [nu,   1.0, 0.0],
        [0.0,  0.0, (1.0 - nu) / 2.0]
    ])
    return D


def elem_stiffness(coords, E, nu):
    """
    Element stiffness matrix for a CST triangle (plane stress).

    Parameters
    ----------
    coords : (3, 2) array
        Nodal coordinates in reference configuration.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    K_e : (6, 6) array
        Element stiffness matrix.
    """
    A0, dNdX, dNdY = tri_area_and_grad(coords)
    if A0 < 1e-14:
        return np.zeros((6, 6))

    B = strain_displacement_matrix(dNdX, dNdY)
    D = material_matrix(E, nu)
    K_e = A0 * (B.T @ D @ B)
    return K_e


def elem_fint(coords, u_elem, E, nu):
    """
    Element internal force for linear elasticity.

    Parameters
    ----------
    coords : (3, 2) array
    u_elem : (6,) array
        Element DOF displacements [u1x, u1y, u2x, u2y, u3x, u3y].
    E, nu : float

    Returns
    -------
    f_int : (6,) array
    """
    K_e = elem_stiffness(coords, E, nu)
    return K_e @ u_elem


def elem_fint_K(coords, u_elem, E, nu):
    """
    Combined interface matching hyperelastic signature.
    Returns (f_int, K_e) for use by the assembler.
    """
    K_e = elem_stiffness(coords, E, nu)
    return K_e @ u_elem, K_e
