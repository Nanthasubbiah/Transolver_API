"""
FastAPI application entry point.
Run locally:  uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from .db import init_db
from .auth.routes import router as auth_router
from .solver.routes import router as solver_router
from .solver.registry import get_registry
from .config import SCREENSHOTS_DIR
import os

app = FastAPI(
    title="Elasticity Solver API",
    version="1.0.0",
    docs_url=None,      # disable Swagger
    redoc_url=None,      # disable ReDoc
)

app.mount("/screenshots", StaticFiles(directory=SCREENSHOTS_DIR), name="screenshots")
app.include_router(auth_router)
app.include_router(solver_router)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    """Main documentation page."""
    models = get_registry()

    # Build model cards
    model_cards = ""
    for m in models:
        screenshots_html = ""
        for s in m.get("screenshots", []):
            fname = os.path.basename(s)
            screenshots_html += f'<img src="/screenshots/{fname}" alt="{fname}">'

        if m["material"] == "mooney_rivlin":
            badge = '<span class="badge badge-hyper">Hyperelastic</span>'
            traction_val = "10000.0"
        else:
            badge = '<span class="badge badge-linear">Linear Elastic</span>'
            traction_val = "1.0"

        model_cards += f"""
        <div class="card">
            <h3>{m['id']} {badge}</h3>
            <p>{m['description']}</p>
            <div class="screenshots">{screenshots_html}</div>
            <div class="info-grid">
                <div class="info-box"><strong>Material:</strong> {m['material']}</div>
                <div class="info-box"><strong>Params:</strong> <code>{m['mat_params']}</code></div>
                <div class="info-box"><strong>Nodes:</strong> {m['nodes']}</div>
                <div class="info-box"><strong>Topology:</strong> {m['topology']}</div>
            </div>
            <p><strong>BCs:</strong> {m['bc_description']}</p>
        </div>
        """

    html = """
    <html><head><title>Elasticity Solver API — Documentation</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }
        h1 { color: #0f3460; border-bottom: 3px solid #0f3460; padding-bottom: 10px; }
        h2 { color: #16213e; margin-top: 30px; border-left: 4px solid #0f3460; padding-left: 12px; }
        h3 { color: #0f3460; margin-top: 0; }
        .card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
        .screenshots { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
        .screenshots img { max-width: 800px; border: 1px solid #ddd; border-radius: 4px; }
        .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
        .info-box { padding: 8px 12px; background: #f8f9fa; border-radius: 4px; border-left: 3px solid #0f3460; font-size: 0.95em; }
        code { background: #e8ecf0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
        pre { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px; overflow-x: auto; line-height: 1.5; font-size: 0.9em; }
        pre .comment { color: #6c7086; }
        pre .string { color: #a6e3a1; }
        pre .keyword { color: #cba6f7; }
        .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
        .badge-linear { background: #d4edda; color: #155724; }
        .badge-hyper { background: #fff3cd; color: #856404; }
        .step { display: flex; gap: 15px; margin: 10px 0; }
        .step-num { background: #0f3460; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0; }
        .step-text { flex: 1; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th { background: #0f3460; color: white; padding: 10px; text-align: left; }
        td { padding: 8px 10px; border-bottom: 1px solid #e0e0e0; }
        tr:nth-child(even) { background: #f8f9fa; }
        .download-btn { display: inline-block; background: #0f3460; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: bold; margin: 10px 0; }
        .download-btn:hover { background: #16213e; }
        .toc { background: white; border-radius: 8px; padding: 15px 20px; margin: 15px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
        .toc a { text-decoration: none; color: #0f3460; display: block; padding: 4px 0; }
        .toc a:hover { text-decoration: underline; }
    </style></head><body>

    <h1>Elasticity Solver API</h1>
    <p>FEM elasticity solver with ML (Transolver) warm start. Predict displacement with a neural network, 
    then use it as initial guess for Newton-Raphson + PETSc to converge faster.</p>

    <div class="toc">
        <strong>Contents</strong>
        <a href="#models">1. Available Models</a>
        <a href="#commands">2. API Commands</a>
        <a href="#petsc">3. PETSc Solver Options</a>
        <a href="#example">4. Complete Python Example</a>
        <a href="#custom">5. Using Your Own Model</a>
        <a href="#format">6. Node & Element Table Format</a>
    </div>

    <!-- ═══════ MODELS ═══════ -->
    <h2 id="models">1. Available Models</h2>
    <p>These are the pre-trained Transolver models. Use their <code>id</code> in <code>POST /solve</code>.</p>
    """ + model_cards + """

    <!-- ═══════ COMMANDS ═══════ -->
    <h2 id="commands">2. API Commands</h2>
    <p>Your Python code needs only these <code>requests</code> calls:</p>

    <div class="card">
        <h3>Step 1 — Register (once) — gives you your API key</h3>
<pre>
<span class="keyword">import</span> requests

BASE = <span class="string">"http://localhost:8000"</span>

<span class="comment"># Register — you get your API key immediately</span>
r = requests.post(f<span class="string">"{BASE}/auth/register"</span>, json={
    <span class="string">"username"</span>: <span class="string">"your_name"</span>,
    <span class="string">"password"</span>: <span class="string">"your_password"</span>
})
data = r.json()
api_key = data[<span class="string">"api_key"</span>]  <span class="comment"># "elas_abc123..." — SAVE THIS!</span>
HEADERS = {<span class="string">"X-API-Key"</span>: api_key}
print(data)
</pre>
    </div>

    <div class="card">
        <h3>Lost your API key? Reset it</h3>
<pre>
<span class="comment"># Login to deactivate old key and get a new one</span>
r = requests.post(f<span class="string">"{BASE}/auth/reset-key"</span>, json={
    <span class="string">"username"</span>: <span class="string">"your_name"</span>,
    <span class="string">"password"</span>: <span class="string">"your_password"</span>
})
api_key = r.json()[<span class="string">"api_key"</span>]  <span class="comment"># new key, old one is dead</span>
HEADERS = {<span class="string">"X-API-Key"</span>: api_key}
</pre>
    </div>

    <div class="card">
        <h3>Step 3 — See Available Models</h3>
<pre>
r = requests.get(f<span class="string">"{BASE}/models"</span>)
models = r.json()
<span class="keyword">for</span> m <span class="keyword">in</span> models:
    print(m[<span class="string">"id"</span>], m[<span class="string">"material"</span>], m[<span class="string">"nodes"</span>], <span class="string">"nodes"</span>)
</pre>
    </div>

    <div class="card">
        <h3>Step 4 — Solve</h3>
<pre>
<span class="keyword">import</span> numpy <span class="keyword">as</span> np

<span class="comment"># Your nodes — e.g. from .npy file or mesh generator</span>
nodes = np.load(<span class="string">"your_nodes.npy"</span>)  <span class="comment"># shape (n_nodes, 2)</span>

<span class="comment"># Find BC nodes</span>
etol = 0.02
mid_x  = np.where(np.abs(nodes[:, 0] - 0.5) < etol)[0].tolist()
edge_y = np.where((nodes[:, 1] < etol) | (nodes[:, 1] > 1 - etol))[0].tolist()
mid_y  = np.where(np.abs(nodes[:, 1] - 0.5) < etol)[0].tolist()
edge_x = np.where((nodes[:, 0] < etol) | (nodes[:, 0] > 1 - etol))[0].tolist()
right  = np.where(nodes[:, 0] > 1 - etol)[0].tolist()
left   = np.where(nodes[:, 0] < etol)[0].tolist()

r = requests.post(f<span class="string">"{BASE}/solve"</span>, headers=HEADERS, json={
    <span class="string">"nodes"</span>: nodes.tolist(),
    <span class="string">"model_id"</span>: <span class="string">"hyperelastic_2d"</span>,  <span class="comment"># or "linear_elastic_2d"</span>
    <span class="string">"bc"</span>: {
        <span class="string">"dirichlet"</span>: [
            {<span class="string">"node_ids"</span>: mid_x,  <span class="string">"component"</span>: <span class="string">"x"</span>, <span class="string">"value"</span>: 0.0},
            {<span class="string">"node_ids"</span>: edge_y, <span class="string">"component"</span>: <span class="string">"x"</span>, <span class="string">"value"</span>: 0.0},
            {<span class="string">"node_ids"</span>: mid_y,  <span class="string">"component"</span>: <span class="string">"y"</span>, <span class="string">"value"</span>: 0.0},
            {<span class="string">"node_ids"</span>: edge_x, <span class="string">"component"</span>: <span class="string">"y"</span>, <span class="string">"value"</span>: 0.0}
        ],
        <span class="string">"tractions"</span>: [
            {<span class="string">"node_ids"</span>: right, <span class="string">"tx"</span>: 10000.0, <span class="string">"ty"</span>: 0.0},
            {<span class="string">"node_ids"</span>: left,  <span class="string">"tx"</span>: -10000.0, <span class="string">"ty"</span>: 0.0}
        ]
    },
    <span class="string">"use_warm_start"</span>: <span class="keyword">True</span>,
    <span class="string">"solver"</span>: <span class="string">"petsc"</span>,
    <span class="string">"petsc_opts"</span>: {
        <span class="string">"solver_type"</span>: <span class="string">"cg"</span>,
        <span class="string">"precond"</span>: <span class="string">"icc"</span>,
        <span class="string">"rtol"</span>: 1e-8
    },
    <span class="string">"tol"</span>: 1e-6,
    <span class="string">"max_iter"</span>: 30
})

result = r.json()
u = np.array(result[<span class="string">"displacement"</span>])  <span class="comment"># shape (n_nodes, 2)</span>
print(f<span class="string">"Converged: {result['converged']}, Iterations: {result['iterations']}"</span>)
print(f<span class="string">"Warm started: {result['warm_started']}"</span>)
print(f<span class="string">"Max disp: ux={u[:,0].max():.6f}, uy={u[:,1].max():.6f}"</span>)
</pre>
    </div>

    <!-- ═══════ PETSC OPTIONS ═══════ -->
    <h2 id="petsc">3. PETSc Solver Options</h2>
    <p>Set in the <code>"petsc_opts"</code> field:</p>
    <div class="card">
        <table>
            <tr><th>Combo</th><th>Best For</th><th>JSON</th></tr>
            <tr>
                <td><strong>Direct (LU)</strong></td>
                <td>Small problems, guaranteed convergence</td>
                <td><code>{"solver_type": "preonly", "precond": "lu"}</code></td>
            </tr>
            <tr>
                <td><strong>CG + ICC</strong></td>
                <td>Elasticity (symmetric), recommended</td>
                <td><code>{"solver_type": "cg", "precond": "icc", "rtol": 1e-8}</code></td>
            </tr>
            <tr>
                <td><strong>GMRES + ILU</strong></td>
                <td>General purpose</td>
                <td><code>{"solver_type": "gmres", "precond": "ilu", "rtol": 1e-8}</code></td>
            </tr>
            <tr>
                <td><strong>GMRES + AMG</strong></td>
                <td>Large problems (1000+ nodes)</td>
                <td><code>{"solver_type": "gmres", "precond": "gamg", "rtol": 1e-8}</code></td>
            </tr>
            <tr>
                <td><strong>MinRes + Jacobi</strong></td>
                <td>Lightweight, quick</td>
                <td><code>{"solver_type": "minres", "precond": "jacobi"}</code></td>
            </tr>
        </table>
    </div>

    <!-- ═══════ COMPLETE EXAMPLE ═══════ -->
    <h2 id="example">4. Complete Python Example</h2>
    <p>Download a working test script that calls every endpoint:</p>
    <a href="/download/test_api.py" class="download-btn">⬇ Download test_api.py</a>

    <!-- ═══════ CUSTOM MODEL ═══════ -->
    <h2 id="custom">5. Using Your Own Model</h2>
    <p>If you trained your own Transolver <code>.pt</code> checkpoint:</p>
    <div class="card">
<pre>
<span class="comment"># 1. Upload your .pt file (and optional .npy normalizer)</span>
<span class="keyword">with</span> open(<span class="string">"my_model.pt"</span>, <span class="string">"rb"</span>) <span class="keyword">as</span> ckpt:
    r = requests.post(f<span class="string">"{BASE}/upload-model"</span>,
        headers=HEADERS,
        files={<span class="string">"checkpoint"</span>: (<span class="string">"my_model.pt"</span>, ckpt)}
    )
upload_id = r.json()[<span class="string">"upload_id"</span>]

<span class="comment"># 2. Solve with your model</span>
r = requests.post(f<span class="string">"{BASE}/solve/custom"</span>,
    headers=HEADERS,
    params={<span class="string">"upload_id"</span>: upload_id},
    json={
        <span class="string">"nodes"</span>: nodes.tolist(),
        <span class="string">"material"</span>: <span class="string">"mooney_rivlin"</span>,
        <span class="string">"mat_params"</span>: {<span class="string">"C1"</span>: 1.863e5, <span class="string">"C2"</span>: 9.79e3},
        <span class="string">"bc"</span>: { <span class="comment">... same as above ...</span> },
        <span class="string">"solver"</span>: <span class="string">"petsc"</span>
    }
)
</pre>
    </div>

    <!-- ═══════ NODE/ELEMENT FORMAT ═══════ -->
    <h2 id="format">6. Node & Element Table Format</h2>

    <div class="card">
        <h3>Nodes — <code>.npy</code> or <code>.csv</code>, shape <code>(n_nodes, 2)</code></h3>
<pre>
<span class="comment"># Load from .npy</span>
nodes = np.load(<span class="string">"nodes.npy"</span>)  <span class="comment"># shape (972, 2) — [[x0,y0], [x1,y1], ...]</span>

<span class="comment"># Or from .csv (no header, comma separated)</span>
<span class="comment"># 0.3745, 0.9507</span>
<span class="comment"># 0.7320, 0.5987</span>
<span class="comment"># 0.1560, 0.1560</span>
</pre>
        <p><strong>Domain:</strong> Unit square [0,1]×[0,1] with circular hole at center (radius ≈ 0.15).
        972 nodes. Coordinates must be in the same domain as the training data.</p>
        <a href="/download/example_nodes.npy" class="download-btn">⬇ example_nodes.npy</a>
        <a href="/download/example_nodes.csv" class="download-btn">⬇ example_nodes.csv</a>
    </div>

    <div class="card">
        <h3>Elements (optional) — <code>.npy</code> or <code>.csv</code>, shape <code>(n_elements, 3)</code></h3>
<pre>
<span class="comment"># Load from .npy</span>
elements = np.load(<span class="string">"elements.npy"</span>)  <span class="comment"># shape (1913, 3) — [[n0,n1,n2], ...]</span>

<span class="comment"># Or from .csv (no header, 0-indexed triangle connectivity)</span>
<span class="comment"># 251, 565, 808</span>
<span class="comment"># 251, 495, 236</span>
<span class="comment"># 495, 251, 339</span>

<span class="comment"># If not provided → auto Delaunay triangulation</span>
</pre>
        <a href="/download/example_elements.npy" class="download-btn">⬇ example_elements.npy</a>
        <a href="/download/example_elements.csv" class="download-btn">⬇ example_elements.csv</a>
    </div>

    <div class="card">
        <h3>Using in your code</h3>
<pre>
<span class="comment"># Option A: pass nodes as list (auto Delaunay mesh)</span>
nodes = np.load(<span class="string">"example_nodes.npy"</span>)
r = requests.post(f<span class="string">"{BASE}/solve"</span>, headers=HEADERS, json={
    <span class="string">"nodes"</span>: nodes.tolist(),
    <span class="string">"model_id"</span>: <span class="string">"hyperelastic_2d"</span>,
    <span class="string">"bc"</span>: { ... }
})

<span class="comment"># Option B: pass both nodes + elements (skip Delaunay)</span>
elements = np.load(<span class="string">"example_elements.npy"</span>)
r = requests.post(f<span class="string">"{BASE}/solve"</span>, headers=HEADERS, json={
    <span class="string">"nodes"</span>: nodes.tolist(),
    <span class="string">"elements"</span>: elements.tolist(),  <span class="comment"># ← your own mesh</span>
    <span class="string">"model_id"</span>: <span class="string">"hyperelastic_2d"</span>,
    <span class="string">"bc"</span>: { ... }
})
</pre>
    </div>

    <div style="text-align:center; padding:30px; color:#666; font-size:0.9em;">
        Elasticity Solver API v1.0
    </div>

    </body></html>
    """
    return html


@app.get("/download/test_api.py")
def download_test_api():
    """Download the example test_api.py script."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test_api.py")
    if os.path.exists(path):
        return FileResponse(path, filename="test_api.py", media_type="text/x-python")
    return {"error": "test_api.py not found"}


@app.get("/download/example_nodes.npy")
def download_example_nodes_npy():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "example_nodes.npy")
    if os.path.exists(path):
        return FileResponse(path, filename="example_nodes.npy")
    return {"error": "file not found"}


@app.get("/download/example_nodes.csv")
def download_example_nodes_csv():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "example_nodes.csv")
    if os.path.exists(path):
        return FileResponse(path, filename="example_nodes.csv")
    return {"error": "file not found"}


@app.get("/download/example_elements.npy")
def download_example_elements_npy():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "example_elements.npy")
    if os.path.exists(path):
        return FileResponse(path, filename="example_elements.npy")
    return {"error": "file not found"}


@app.get("/download/example_elements.csv")
def download_example_elements_csv():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "example_elements.csv")
    if os.path.exists(path):
        return FileResponse(path, filename="example_elements.csv")
    return {"error": "file not found"}
