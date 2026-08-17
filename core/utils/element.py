"""
Constant Strain Triangle (CST) element utilities.
Shape function derivatives, B-matrix, and area computation
for 3-node triangular elements in 2D.
"""

import numpy as np


def tri_area_and_grad(coords):
    """
    Compute triangle area and shape function gradients (dN/dX, dN/dY).

    Parameters
    ----------
    coords : (3, 2) array
        Reference coordinates of the three nodes.

    Returns
    -------
    A0 : float
        Element area (unsigned).
    dNdX : (3,) array
        dN_i / dX for each node.
    dNdY : (3,) array
        dN_i / dY for each node.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    A0 = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    # Shape function gradients (constant over element)
    b = np.array([y2 - y3, y3 - y1, y1 - y2])  # dN/dX * 2A
    c = np.array([x3 - x2, x1 - x3, x2 - x1])  # dN/dY * 2A

    dNdX = b / (2.0 * A0) if A0 > 1e-14 else np.zeros(3)
    dNdY = c / (2.0 * A0) if A0 > 1e-14 else np.zeros(3)

    return A0, dNdX, dNdY


def strain_displacement_matrix(dNdX, dNdY):
    """
    Build the 3x6 strain-displacement matrix B for a CST element.
    Strain ordering: [eps_xx, eps_yy, gamma_xy].

    Parameters
    ----------
    dNdX, dNdY : (3,) arrays
        Shape function gradients from tri_area_and_grad.

    Returns
    -------
    B : (3, 6) array
    """
    B = np.zeros((3, 6))
    for i in range(3):
        B[0, 2 * i]     = dNdX[i]       # eps_xx
        B[1, 2 * i + 1] = dNdY[i]       # eps_yy
        B[2, 2 * i]     = dNdY[i]       # gamma_xy
        B[2, 2 * i + 1] = dNdX[i]
    return B


def deformation_gradient_matrix(dNdX, dNdY):
    """
    Build the 4x6 matrix BX such that [du1/dX, du1/dY, du2/dX, du2/dY]^T = BX @ u_elem.
    Used for hyperelastic formulations that need the full deformation gradient F.

    Parameters
    ----------
    dNdX, dNdY : (3,) arrays

    Returns
    -------
    BX : (4, 6) array
    """
    BX = np.zeros((4, 6))
    for i in range(3):
        BX[0, 2 * i] = dNdX[i]      # dux/dX
        BX[1, 2 * i] = dNdY[i]      # dux/dY  (note: row 1 pairs with col 2i, not 2i+1)
        BX[2, 2 * i + 1] = dNdX[i]  # duy/dX
        BX[3, 2 * i + 1] = dNdY[i]  # duy/dY
    return BX
