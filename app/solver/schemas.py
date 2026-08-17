"""
Pydantic models for the /solve and /solve/custom endpoints.
Examples drawn from test_api.py and test_api_hyper.py.
"""
from typing import Optional
from pydantic import BaseModel, Field


class DirichletBC(BaseModel):
    node_ids: list[int] = Field(..., description="Node indices to fix")
    component: str = Field(..., description="'x' or 'y'")
    value: float = Field(0.0, description="Prescribed displacement value")

    model_config = {"json_schema_extra": {
        "examples": [{"node_ids": [0, 5, 10], "component": "x", "value": 0.0}]
    }}


class TractionBC(BaseModel):
    node_ids: list[int]
    tx: float = 0.0
    ty: float = 0.0


class SymmetryBC(BaseModel):
    axis: str = Field(..., description="'x' or 'y'")
    coord: float = Field(..., description="Location of symmetry plane")
    tol: float = 0.02


class OuterTractionBC(BaseModel):
    traction: float = Field(..., description="Magnitude of outward traction")
    tol: float = 0.02


class BoundaryConditionsInput(BaseModel):
    dirichlet: list[DirichletBC] = []
    tractions: list[TractionBC] = []
    symmetry: list[SymmetryBC] = []
    outer_traction: Optional[OuterTractionBC] = None


# ── Solve with built-in model ────────────────────────────────────────────
class SolveRequest(BaseModel):
    """
    Solve using a built-in model.
    
    Workflow:
    1. GET /models to see available models and their BCs
    2. Load your nodes from .npy file (972 nodes, 2D)
    3. POST /solve with nodes, model_id, and BCs
    
    See test_api_hyper.py for full Python example.
    """
    nodes: list[list[float]] = Field(
        ..., description="Node coords [[x,y],...] from Random_UnitCell_XY_10.npy[:,:,sample]",
        example=[[0.5, 0.0], [0.48, 0.02], [0.52, 0.03]]
    )
    model_id: str = Field(
        ..., description="From GET /models",
        example="hyperelastic_2d"
    )
    bc: BoundaryConditionsInput = Field(
        ..., example={
            "dirichlet": [
                {"node_ids": [0, 5, 10], "component": "x", "value": 0.0},
                {"node_ids": [0, 5, 10], "component": "y", "value": 0.0}
            ],
            "tractions": [
                {"node_ids": [20, 21, 22], "tx": 10000.0, "ty": 0.0},
                {"node_ids": [30, 31, 32], "tx": -10000.0, "ty": 0.0}
            ]
        }
    )
    elements: Optional[list[list[int]]] = Field(None, description="Auto Delaunay if None")
    use_warm_start: bool = Field(True, description="Use Transolver prediction as initial guess")
    solver: str = Field("petsc", example="petsc")
    petsc_opts: Optional[dict] = Field(None, example={
        "solver_type": "cg", "precond": "icc",
        "rtol": 1e-8, "atol": 1e-12, "max_iter": 1000
    })
    tol: float = Field(1e-6, description="NR convergence tolerance")
    max_iter: int = Field(30, description="Max NR iterations")


# ── Solve with custom uploaded model ────────────────────────────────────

class CustomSolveRequest(BaseModel):
    """
    Solve using a user-uploaded .pt model.
    First upload via POST /upload-model, then use the upload_id here.
    You must specify the material type and parameters yourself.
    """
    nodes: list[list[float]]
    material: str = Field(
        ..., description="'linear_elastic' or 'mooney_rivlin'"
    )
    mat_params: dict = Field(
        ..., description="Material params. Linear: {E: 1.0, nu: 0.3}. "
        "Mooney-Rivlin: {C1: 1.863e5, C2: 9.79e3}"
    )
    bc: BoundaryConditionsInput
    elements: Optional[list[list[int]]] = None
    use_warm_start: bool = True
    solver: str = "petsc"
    petsc_opts: Optional[dict] = None
    tol: float = 1e-6
    max_iter: int = 30
    model_config_override: Optional[dict] = Field(
        None, description="Override Transolver architecture config: "
        "{n_layers: 3, n_hidden: 64, n_heads: 4, slice_num: 32, ref: 8}"
    )


# ── Response ─────────────────────────────────────────────────────────────

class SolveResponse(BaseModel):
    """Solver output with displacement field and convergence info."""
    displacement: list[list[float]] = Field(
        ..., description="Displacement per node [[ux, uy], ...]. Shape: (n_nodes, 2)"
    )
    iterations: int = Field(..., description="Number of Newton-Raphson iterations")
    converged: bool
    residual_history: list[float] = Field(
        ..., description="NR residual norm per iteration"
    )
    model_id: str
    warm_started: bool = Field(
        ..., description="Whether ML warm start was used"
    )


# ── Model info ───────────────────────────────────────────────────────────
class ModelInfo(BaseModel):
    id: str = Field(example="linear_elastic_2d")
    material: str = Field(example="linear_elastic")
    mat_params: dict = Field(example={"E": 1.0, "nu": 0.3})
    nodes: int = Field(example=972)
    topology: str = Field(example="Plate with circular hole (quarter symmetry)")
    domain: str = Field(example="Unit square [0,1]x[0,1] with hole at center")
    bc_description: str = Field(example="Symmetry: ux=0 at x=0.5, uy=0 at y=0.5. Traction: ±T on edges")
    bc_details: dict = Field(example={
        "symmetry": [{"axis": "x", "coord": 0.5}, {"axis": "y", "coord": 0.5}],
        "outer_traction": {"traction": 1.0}
    })
    screenshots: list[str] = Field(example=[
        "http://localhost:8000/screenshots/linear_elastic.png"
    ])
    description: str = Field(example="2D plane stress, 972 nodes, trained on FEM batch data")