from ml.warm_start import load_warm_starter
from api import ElasticitySolver
from solver.boundary import BoundaryConditions
import numpy as np
import time
# u_pred = ws.predict(nodes)  # (972, 2)

from solver.boundary import BoundaryConditions
all_nodes = np.load("Random_UnitCell_XY_10.npy")
nodes = all_nodes[:, :, 25]                  # (972, 2) — first sample
print(nodes.shape)  # should be (972, 2)

bc = BoundaryConditions(nodes)

# ux=0 at x=0.5
bc.add_dirichlet(np.where(np.abs(nodes[:,0] - 0.5) < 0.02)[0], 'x')
# ux=0 at y=0,1
bc.add_dirichlet(np.where((nodes[:,1] < 0.02) | (nodes[:,1] > 0.98))[0], 'x')
# uy=0 at y=0.5
bc.add_dirichlet(np.where(np.abs(nodes[:,1] - 0.5) < 0.02)[0], 'y')
# uy=0 at x=0,1
bc.add_dirichlet(np.where((nodes[:,0] < 0.02) | (nodes[:,0] > 0.98))[0], 'y')

# traction left/right only
right = np.where(nodes[:,0] > 0.98)[0]
left  = np.where(nodes[:,0] < 0.02)[0]
bc.add_traction(right, tx=1e4)
bc.add_traction(left, tx=-1e4)

solver = ElasticitySolver(
    nodes=nodes, elements=None,
    material='mooney_rivlin',
    mat_params={'C1': 1.863e5, 'C2': 9.79e3},
    bc=bc
)

ws = load_warm_starter(
    checkpoint_path="checkpoints/elas_hyper_disp_Transolver.pt",
    train_disp_path="disp_hyper_xy.npy",
    model_config={'n_layers': 3, 'n_hidden': 64, 'n_heads': 4,
                  'slice_num': 32, 'ref': 8},
)

u_pred = ws.predict(nodes)

# --- Debug: plot each NR iteration ---
from solver.assembler import assemble
from solver.petsc_solver import scipy_solve
import matplotlib.pyplot as plt

u = u_pred.ravel().copy()
u[solver.fixed_dofs] = 0.0
print(f"u_pred range: [{u_pred.min():.6f}, {u_pred.max():.6f}]")

fig, axes = plt.subplots(2, 5, figsize=(20, 6))
for it in range(10):
    # Plot current displacement
    ax = axes[it // 5, it % 5]
    u_2d = u.reshape(-1, 2)
    sc = ax.tricontourf(nodes[:,0], nodes[:,1], u_2d[:,0], levels=20)
    ax.set_title(f'Iter {it}\nmax={u_2d[:,0].max():.4f}')
    plt.colorbar(sc, ax=ax)
    
    # NR step
    F_int, KT = assemble(u, nodes, solver.elements, solver.material_func, solver.mat_params)
    res = solver.f_ext - F_int
    res_free = res[solver.free_dofs]
    print(f"Iter {it}: |R|={np.linalg.norm(res_free):.4e}, u_max={np.abs(u).max():.6f}")
    
    KT_ff = KT[solver.free_dofs, :][:, solver.free_dofs]
    du_free, _ = scipy_solve(KT_ff, res_free)
    u[solver.free_dofs] += du_free

plt.tight_layout()
plt.savefig('warm_start_iterations.png')
plt.show()

t0 = time.time()
u_warm, info_warm = solver.solve(
    warm_start_u=u_pred,
    solver='petsc',
    petsc_opts={
        'solver_type': 'cg',      # or 'gmres', 'minres', 'bcgs'
        'precond': 'icc',          # or 'ilu', 'jacobi', 'gamg', 'none'
        'rtol': 1e-8,
        'atol': 1e-12,
        'max_iter': 1000,
    }
)
t_warm = time.time() - t0

# ── Cold start (no warm start) ──
# Cold start iteration-by-iteration
u_c = np.zeros(solver.n_dof)
for it in range(5):
    F_int, KT = assemble(u_c, nodes, solver.elements, solver.material_func, solver.mat_params)
    res = solver.f_ext - F_int
    res_free = res[solver.free_dofs]
    print(f"Cold Iter {it}: |R|={np.linalg.norm(res_free):.4e}, u_max={np.abs(u_c).max():.6f}")
    KT_ff = KT[solver.free_dofs, :][:, solver.free_dofs]
    du_free, _ = scipy_solve(KT_ff, res_free)
    u_c[solver.free_dofs] += du_free
t0 = time.time()
u_cold, info_cold = solver.solve(warm_start_u=None, solver='petsc')
t_cold = time.time() - t0
print(info_cold.keys())  # see what's available
print(info_warm.keys())
# ── Compare ──
print(f"Cold start: {info_cold['iterations']} NR iters, {t_cold:.3f}s")
print(f"Warm start: {info_warm['iterations']} NR iters, {t_warm:.3f}s")
print(f"Speedup: {t_cold/t_warm:.2f}x")
print(f"Solution match: {np.allclose(u_cold, u_warm, atol=1e-6)}")

# Load fem_batch output
disp_batch = np.load("disp_hyper_xy.npy")[:, :, 25]
print("API vs fem_batch match:", np.allclose(u_cold, disp_batch, atol=1e-6))
print("Max diff:", np.abs(u_cold - disp_batch).max())

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(10, 2))
for ax, u, title in zip(axes, 
    [u_pred, u_warm, u_cold, disp_batch],
    ['Transolver prediction', 'Warm start (converged)', 'Cold start (converged)', 'GT']):
    sc = ax.tricontourf(nodes[:,0], nodes[:,1], u[:,0], levels=20)
    ax.set_title(title)
    plt.colorbar(sc, ax=ax)
plt.tight_layout()
plt.savefig('displacement_comparison.png')
plt.show()