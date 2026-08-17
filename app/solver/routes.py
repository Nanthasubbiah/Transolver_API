"""
Solver endpoints: POST /solve, POST /solve/custom, GET /models.
"""
import os
import shutil
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from ..auth.middleware import get_current_user
from ..auth.models import User
from ..config import CHECKPOINTS_DIR, TRAIN_DATA_DIR, DEVICE, UPLOADS_DIR
from .schemas import (
    SolveRequest, SolveResponse, ModelInfo,
    CustomSolveRequest, BoundaryConditionsInput,
)
from .registry import get_registry, get_model_entry
import io
from fastapi import Form

router = APIRouter(tags=["solver"])

# Cache warm starters so we don't reload .pt every request
_warm_starters: dict = {}


def _build_bc(nodes: np.ndarray, bc_input: BoundaryConditionsInput):
    """Convert Pydantic BC input → BoundaryConditions object."""
    from core.solver.boundary import BoundaryConditions

    bc = BoundaryConditions(nodes)
    for d in bc_input.dirichlet:
        bc.add_dirichlet(d.node_ids, d.component, d.value)
    for t in bc_input.tractions:
        bc.add_traction(t.node_ids, t.tx, t.ty)
    for s in bc_input.symmetry:
        bc.add_symmetry(s.axis, s.coord, s.tol)
    if bc_input.outer_traction:
        bc.add_outer_traction(
            bc_input.outer_traction.traction,
            bc_input.outer_traction.tol,
        )
    return bc


def _get_warm_starter(model_entry: dict):
    """Load or return cached WarmStarter for built-in models."""
    model_id = model_entry["id"]
    if model_id in _warm_starters:
        return _warm_starters[model_id]

    from core.ml.warm_start import load_warm_starter

    ckpt = os.path.join(CHECKPOINTS_DIR, model_entry["file"])
    train_disp = None
    if model_entry.get("train_disp"):
        train_disp = os.path.join(TRAIN_DATA_DIR, model_entry["train_disp"])

    ws = load_warm_starter(
        checkpoint_path=ckpt,
        train_disp_path=train_disp,
        model_config=model_entry.get("model_config"),
        device=DEVICE,
    )
    _warm_starters[model_id] = ws
    return ws


def _get_custom_warm_starter(
    checkpoint_path: str,
    train_disp_path: str = None,
    model_config: dict = None,
):
    """Load WarmStarter from a user-uploaded .pt file."""
    cache_key = f"custom_{os.path.basename(checkpoint_path)}"
    if cache_key in _warm_starters:
        return _warm_starters[cache_key]

    from core.ml.warm_start import load_warm_starter

    ws = load_warm_starter(
        checkpoint_path=checkpoint_path,
        train_disp_path=train_disp_path,
        model_config=model_config,
        device=DEVICE,
    )
    _warm_starters[cache_key] = ws
    return ws


# ── GET /models ──────────────────────────────────────────────────────────

@router.get("/models", response_model=list[ModelInfo])
def list_models():
    """List all available built-in solver models with topology details."""
    return [
        ModelInfo(
            id=m["id"],
            material=m["material"],
            mat_params=m["mat_params"],
            nodes=m["nodes"],
            topology=m["topology"],
            domain=m["domain"],
            bc_description=m["bc_description"],
            bc_details=m["bc_details"],
            screenshots=[
                f"/screenshots/{os.path.basename(s)}" for s in m["screenshots"]
            ],
            description=m["description"],
        )
        for m in get_registry()
    ]


# ── POST /solve (built-in model) ────────────────────────────────────────

@router.post("/solve", response_model=SolveResponse)
def solve(
    req: SolveRequest,
    user: User = Depends(get_current_user),
):
    """Run FEM solve using a built-in model for warm start."""
    from core.api import ElasticitySolver

    try:
        model_entry = get_model_entry(req.model_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

    nodes = np.array(req.nodes)
    elements = np.array(req.elements) if req.elements else None
    bc = _build_bc(nodes, req.bc)

    # Warm start
    warm_u = None
    warm_started = False
    if req.use_warm_start:
        try:
            ws = _get_warm_starter(model_entry)
            warm_u = ws.predict(nodes)
            warm_started = True
        except Exception as e:
            print(f"Warm start failed, falling back to cold: {e}")

    solver = ElasticitySolver(
        nodes=nodes, elements=elements,
        material=model_entry["material"],
        mat_params=model_entry["mat_params"],
        bc=bc,
    )

    u, info = solver.solve(
        warm_start_u=warm_u, solver=req.solver,
        petsc_opts=req.petsc_opts,
        tol=req.tol, max_iter=req.max_iter, verbose=False,
    )

    return SolveResponse(
        displacement=u.tolist(),
        iterations=info["iterations"],
        converged=info["converged"],
        residual_history=[float(r) for r in info.get("residual_history", [])],
        model_id=req.model_id,
        warm_started=warm_started,
    )


# ── POST /upload-model ──────────────────────────────────────────────────

@router.post("/upload-model")
async def upload_model(
    checkpoint: UploadFile = File(..., description=".pt model file"),
    train_disp: UploadFile = File(None, description=".npy normalizer data (optional)"),
    user: User = Depends(get_current_user),
):
    """
    Upload a custom .pt checkpoint (and optional .npy normalizer data).
    Returns an upload_id to use with POST /solve/custom.
    """
    if not checkpoint.filename.endswith(".pt"):
        raise HTTPException(400, "Checkpoint must be a .pt file")

    # Save checkpoint
    user_dir = os.path.join(UPLOADS_DIR, user.id)
    os.makedirs(user_dir, exist_ok=True)

    ckpt_path = os.path.join(user_dir, checkpoint.filename)
    with open(ckpt_path, "wb") as f:
        shutil.copyfileobj(checkpoint.file, f)

    # Save normalizer data if provided
    disp_path = None
    if train_disp and train_disp.filename:
        if not train_disp.filename.endswith(".npy"):
            raise HTTPException(400, "Training data must be a .npy file")
        disp_path = os.path.join(user_dir, train_disp.filename)
        with open(disp_path, "wb") as f:
            shutil.copyfileobj(train_disp.file, f)

    upload_id = checkpoint.filename.replace(".pt", "")
    return {
        "upload_id": upload_id,
        "checkpoint": ckpt_path,
        "train_disp": disp_path,
        "message": f"Model uploaded. Use upload_id='{upload_id}' with POST /solve/custom",
    }


# ── POST /solve/custom ──────────────────────────────────────────────────

@router.post("/solve/custom", response_model=SolveResponse)
def solve_custom(
    req: CustomSolveRequest,
    upload_id: str,
    user: User = Depends(get_current_user),
):
    """
    Run FEM solve using a user-uploaded .pt model for warm start.
    First upload via POST /upload-model, then use the upload_id here.
    """
    from core.api import ElasticitySolver

    # Find uploaded files
    user_dir = os.path.join(UPLOADS_DIR, user.id)
    ckpt_path = os.path.join(user_dir, f"{upload_id}.pt")
    if not os.path.exists(ckpt_path):
        raise HTTPException(404, f"No uploaded model '{upload_id}'. Upload first via POST /upload-model")

    # Check for optional normalizer data
    disp_files = [f for f in os.listdir(user_dir) if f.endswith(".npy")]
    disp_path = os.path.join(user_dir, disp_files[0]) if disp_files else None

    nodes = np.array(req.nodes)
    elements = np.array(req.elements) if req.elements else None
    bc = _build_bc(nodes, req.bc)

    # Warm start with custom model
    warm_u = None
    warm_started = False
    if req.use_warm_start:
        try:
            ws = _get_custom_warm_starter(
                checkpoint_path=ckpt_path,
                train_disp_path=disp_path,
                model_config=req.model_config_override,
            )
            warm_u = ws.predict(nodes)
            warm_started = True
        except Exception as e:
            print(f"Custom warm start failed, falling back to cold: {e}")

    solver = ElasticitySolver(
        nodes=nodes, elements=elements,
        material=req.material,
        mat_params=req.mat_params,
        bc=bc,
    )

    u, info = solver.solve(
        warm_start_u=warm_u, solver=req.solver,
        petsc_opts=req.petsc_opts,
        tol=req.tol, max_iter=req.max_iter, verbose=False,
    )

    return SolveResponse(
        displacement=u.tolist(),
        iterations=info["iterations"],
        converged=info["converged"],
        residual_history=[float(r) for r in info.get("residual_history", [])],
        model_id=f"custom_{upload_id}",
        warm_started=warm_started,
    )


@router.post("/solve/upload", response_model=SolveResponse)
async def solve_with_files(
    nodes_file: UploadFile = File(..., description=".npy or .csv, shape (n_nodes, 2)"),
    model_id: str = Form(...),
    solver: str = Form("petsc"),
    use_warm_start: bool = Form(True),
    tol: float = Form(1e-6),
    max_iter: int = Form(30),
    petsc_opts: str = Form("{}"),
    bc_json: str = Form(..., description="BC as JSON string"),
    elements_file: UploadFile = File(None, description=".npy or .csv, shape (n_elements, 3). Skip = auto Delaunay"),
    user: User = Depends(get_current_user),
):
    """Solve by uploading node/element files directly."""
    import json

    # Read nodes
    contents = await nodes_file.read()
    if nodes_file.filename.endswith(".npy"):
        nodes = np.load(io.BytesIO(contents))
    else:  # csv
        nodes = np.loadtxt(io.BytesIO(contents), delimiter=",")

    # Read elements (optional)
    elements = None
    if elements_file and elements_file.filename:
        el_contents = await elements_file.read()
        if elements_file.filename.endswith(".npy"):
            elements = np.load(io.BytesIO(el_contents))
        else:
            elements = np.loadtxt(io.BytesIO(el_contents), delimiter=",").astype(int)

    bc_input = BoundaryConditionsInput(**json.loads(bc_json))
    opts = json.loads(petsc_opts)

    # ... rest same as /solve