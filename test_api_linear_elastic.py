"""
Test the Elasticity Solver API endpoints.
Run the server first:  uvicorn app.main:app --reload

Then:  python test_api.py
"""
import requests
import numpy as np

BASE = "http://localhost:8000"


# ── 1. Register & get API key ────────────────────────────────────────────

print("=" * 60)
print("1. Register + Get API Key")
print("=" * 60)

r = requests.post(f"{BASE}/auth/register", json={"username": "testuser", "password": "testpass"})
print(f"Register: {r.status_code} — {r.json()}")

r = requests.post(f"{BASE}/auth/api-key", json={"username": "testuser", "password": "testpass"})
api_key = r.json()["api_key"]
print(f"API Key: {api_key}")

HEADERS = {"X-API-Key": api_key}


# ── 2. List models ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. GET /models")
print("=" * 60)

r = requests.get(f"{BASE}/models")
models = r.json()
for m in models:
    print(f"\n  Model: {m['id']}")
    print(f"  Material: {m['material']} — {m['mat_params']}")
    print(f"  Nodes: {m['nodes']}")
    print(f"  Topology: {m['topology']}")
    print(f"  BCs: {m['bc_description']}")
    print(f"  Screenshots: {m['screenshots']}")


# ── 3. Solve — Linear elastic with symmetry BCs ─────────────────────────

print("\n" + "=" * 60)
print("3. POST /solve — Linear Elastic (symmetry BCs)")
print("=" * 60)

# Create a simple 5x5 grid
nx, ny = 5, 5
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
xx, yy = np.meshgrid(x, y)
nodes = np.column_stack([xx.ravel(), yy.ravel()])

# Find BC nodes
sym_x = np.where(np.abs(nodes[:, 0] - 0.5) < 0.01)[0].tolist()
sym_y = np.where(np.abs(nodes[:, 1] - 0.5) < 0.01)[0].tolist()

r = requests.post(f"{BASE}/solve", headers=HEADERS, json={
    "nodes": nodes.tolist(),
    "model_id": "linear_elastic_2d",
    "bc": {
        "symmetry": [
            {"axis": "x", "coord": 0.5, "tol": 0.01},
            {"axis": "y", "coord": 0.5, "tol": 0.01}
        ],
        "outer_traction": {"traction": 1000.0, "tol": 0.01}
    },
    "use_warm_start": True,
    "solver": "scipy",
    "tol": 1e-6,
    "max_iter": 30
})

result = r.json()
if r.status_code == 200:
    u = np.array(result["displacement"])
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Warm started: {result['warm_started']}")
    print(f"  Max disp: ux={u[:,0].max():.6f}, uy={u[:,1].max():.6f}")
else:
    print(f"  Error {r.status_code}: {result}")

# ── 5. Cold vs Warm comparison ──────────────────────────────────────────

print("\n" + "=" * 60)
print("5. Cold vs Warm Start Comparison")
print("=" * 60)

for warm in [False, True]:
    r = requests.post(f"{BASE}/solve", headers=HEADERS, json={
        "nodes": nodes.tolist(),
        "model_id": "hyperelastic_2d",
        "bc": {
            "dirichlet": [
                {"node_ids": sym_x_nodes, "component": "x"},
                {"node_ids": bnd_xy_nodes, "component": "x"},
                {"node_ids": sym_y_nodes, "component": "y"},
                {"node_ids": bnd_yx_nodes, "component": "y"}
            ],
            "tractions": [
                {"node_ids": right_nodes, "tx": 10000.0},
                {"node_ids": left_nodes, "tx": -10000.0}
            ]
        },
        "use_warm_start": warm,
        "solver": "petsc",
        "tol": 1e-6,
        "max_iter": 30
    })
    result = r.json()
    label = "Warm" if warm else "Cold"
    if r.status_code == 200:
        print(f"  {label}: {result['iterations']} iters, converged={result['converged']}, warm_started={result['warm_started']}")
    else:
        print(f"  {label}: Error {r.status_code} — {result}")


print("\n" + "=" * 60)
print("ALL TESTS DONE")
print("=" * 60)