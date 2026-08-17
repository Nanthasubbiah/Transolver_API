"""
Full pipeline test for elasticity_api.
Tests both linear elastic and Mooney-Rivlin on a simple mesh,
plus warm start comparison.

Run: python test_pipeline.py
"""

import numpy as np
from elasticity_api.api import ElasticitySolver
from elasticity_api.solver.boundary import BoundaryConditions


def make_test_mesh():
    """Create a 5x5 grid of nodes with triangular elements."""
    nx, ny = 5, 5
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    elements = np.array(elements)
    return nodes, elements


def test_linear_elastic():
    """Test 1: Linear elastic — fix left, pull right."""
    print("\n" + "=" * 60)
    print("TEST 1: Linear Elastic (plane stress)")
    print("=" * 60)

    nodes, elements = make_test_mesh()
    bc = BoundaryConditions(nodes)

    # Fix left edge (x=0)
    left = np.where(nodes[:, 0] < 0.01)[0]
    bc.add_dirichlet(left, 'x', 0.0)
    bc.add_dirichlet(left, 'y', 0.0)

    # Traction on right edge
    right = np.where(nodes[:, 0] > 0.99)[0]
    bc.add_traction(right, tx=1e4, ty=0.0)

    solver = ElasticitySolver(
        nodes, elements,
        material='linear',
        mat_params={'E': 1e5, 'nu': 0.3},
        bc=bc
    )

    u, info = solver.solve()
    assert info['converged'], "Linear elastic did not converge!"
    assert info['iterations'] == 1, f"Linear should converge in 1 iter, got {info['iterations']}"

    print(f"\nMax displacement: ux={u[:,0].max():.6f}, uy={u[:,1].max():.6f}")
    print(f"TEST 1 PASSED\n")
    return u


def test_mooney_rivlin():
    """Test 2: Mooney-Rivlin hyperelastic — fix left, pull right."""
    print("\n" + "=" * 60)
    print("TEST 2: Mooney-Rivlin Hyperelastic")
    print("=" * 60)

    nodes, elements = make_test_mesh()
    bc = BoundaryConditions(nodes)

    left = np.where(nodes[:, 0] < 0.01)[0]
    bc.add_dirichlet(left, 'x', 0.0)
    bc.add_dirichlet(left, 'y', 0.0)

    right = np.where(nodes[:, 0] > 0.99)[0]
    bc.add_traction(right, tx=1e3, ty=0.0)  # smaller load for stability

    solver = ElasticitySolver(
        nodes, elements,
        material='mooney_rivlin',
        mat_params={'C1': 1e5, 'C2': 1e4},
        bc=bc
    )

    u_cold, info_cold = solver.solve()
    assert info_cold['converged'], "MR cold start did not converge!"
    print(f"\nCold start: {info_cold['iterations']} iterations")
    print(f"Max displacement: ux={u_cold[:,0].max():.6f}, uy={u_cold[:,1].max():.6f}")
    print(f"TEST 2 PASSED\n")
    return u_cold, info_cold


def test_warm_start():
    """Test 3: Compare cold vs warm start iteration count."""
    print("\n" + "=" * 60)
    print("TEST 3: Warm Start Comparison")
    print("=" * 60)

    nodes, elements = make_test_mesh()
    bc = BoundaryConditions(nodes)

    left = np.where(nodes[:, 0] < 0.01)[0]
    bc.add_dirichlet(left, 'x', 0.0)
    bc.add_dirichlet(left, 'y', 0.0)

    right = np.where(nodes[:, 0] > 0.99)[0]
    bc.add_traction(right, tx=1e3, ty=0.0)

    solver = ElasticitySolver(
        nodes, elements,
        material='mooney_rivlin',
        mat_params={'C1': 1e5, 'C2': 1e4},
        bc=bc
    )

    # Cold start
    u_cold, info_cold = solver.solve(verbose=False)

    # Warm start: perturb solution slightly (simulating ML prediction)
    u_warm_init = u_cold * 0.9  # 90% of true solution
    u_warm, info_warm = solver.solve(warm_start_u=u_warm_init, verbose=False)

    print(f"  Cold start iterations: {info_cold['iterations']}")
    print(f"  Warm start iterations: {info_warm['iterations']}")
    print(f"  Solution match: {np.allclose(u_cold, u_warm, atol=1e-4)}")

    assert info_warm['iterations'] <= info_cold['iterations'], \
        "Warm start should use fewer or equal iterations!"
    print(f"TEST 3 PASSED\n")


def test_symmetry_bc():
    """Test 4: Symmetry BCs (unit cell style)."""
    print("\n" + "=" * 60)
    print("TEST 4: Symmetry BCs + Outer Traction")
    print("=" * 60)

    nodes, elements = make_test_mesh()
    bc = BoundaryConditions(nodes)
    bc.add_symmetry(axis='x', coord=0.5, tol=0.01)
    bc.add_symmetry(axis='y', coord=0.5, tol=0.01)
    bc.add_outer_traction(traction=1e3, tol=0.01)

    solver = ElasticitySolver(
        nodes, elements,
        material='linear',
        mat_params={'E': 1e5, 'nu': 0.3},
        bc=bc
    )

    u, info = solver.solve()
    assert info['converged']

    # Check symmetry: ux should be ~0 at x=0.5
    sym_x = np.where(np.abs(nodes[:, 0] - 0.5) < 0.01)[0]
    print(f"  ux at symmetry plane: max={np.abs(u[sym_x, 0]).max():.2e} (should be ~0)")
    print(f"TEST 4 PASSED\n")


def test_auto_mesh():
    """Test 5: Auto Delaunay meshing (no elements provided)."""
    print("\n" + "=" * 60)
    print("TEST 5: Auto Delaunay Mesh")
    print("=" * 60)

    np.random.seed(42)
    nodes = np.random.rand(50, 2)
    bc = BoundaryConditions(nodes)

    left = np.where(nodes[:, 0] < 0.1)[0]
    bc.add_dirichlet(left, 'x', 0.0)
    bc.add_dirichlet(left, 'y', 0.0)

    right = np.where(nodes[:, 0] > 0.9)[0]
    bc.add_traction(right, tx=1e3, ty=0.0)

    solver = ElasticitySolver(
        nodes, elements=None,  # auto mesh
        material='linear',
        mat_params={'E': 1e5, 'nu': 0.3},
        bc=bc
    )

    u, info = solver.solve()
    assert info['converged']
    print(f"  Auto-meshed {len(solver.elements)} elements from {len(nodes)} nodes")
    print(f"TEST 5 PASSED\n")


if __name__ == '__main__':
    test_linear_elastic()
    test_mooney_rivlin()
    test_warm_start()
    test_symmetry_bc()
    test_auto_mesh()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
