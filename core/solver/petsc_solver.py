"""
PETSc linear solver wrapper.

Converts scipy sparse matrix + numpy vectors to PETSc binary format,
calls a small C/PETSc executable via subprocess, and reads the
solution back into numpy.

Fallback to scipy.sparse.linalg.spsolve if PETSc is not available.

File location: elasticity_api/solver/petsc_solver.py
"""

import os
import struct
import subprocess
import tempfile
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# ── PETSc binary I/O ────────────────────────────────────────────────────

def _write_petsc_vec(filepath, vec):
    """Write a numpy vector in PETSc binary Vec format."""
    n = len(vec)
    with open(filepath, 'wb') as f:
        # Header: classid (1211214) + length
        f.write(struct.pack('>i', 1211214))
        f.write(struct.pack('>i', n))
        # Data: big-endian doubles
        for v in vec:
            f.write(struct.pack('>d', float(v)))


def _read_petsc_vec(filepath):
    """Read a PETSc binary Vec file back into numpy."""
    with open(filepath, 'rb') as f:
        classid = struct.unpack('>i', f.read(4))[0]
        assert classid == 1211214, f"Not a PETSc Vec file (classid={classid})"
        n = struct.unpack('>i', f.read(4))[0]
        data = struct.unpack(f'>{n}d', f.read(8 * n))
    return np.array(data)


def _write_petsc_mat(filepath, K):
    """Write a scipy CSR matrix in PETSc binary Mat (AIJ) format."""
    K = csr_matrix(K)
    n_rows, n_cols = K.shape
    nnz = K.nnz

    with open(filepath, 'wb') as f:
        # Header: classid (1211216), rows, cols, total nnz
        f.write(struct.pack('>i', 1211216))
        f.write(struct.pack('>i', n_rows))
        f.write(struct.pack('>i', n_cols))
        f.write(struct.pack('>i', nnz))

        # Number of nonzeros per row
        for i in range(n_rows):
            row_nnz = K.indptr[i + 1] - K.indptr[i]
            f.write(struct.pack('>i', row_nnz))

        # Column indices (0-based)
        for j in K.indices:
            f.write(struct.pack('>i', int(j)))

        # Values
        for v in K.data:
            f.write(struct.pack('>d', float(v)))


# ── PETSc C solver source ───────────────────────────────────────────────

_PETSC_C_SOURCE = r"""
#include <petsc.h>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);

    // Read matrix
    Mat A;
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "K.petsc", FILE_MODE_READ, &viewer);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetFromOptions(A);
    MatLoad(A, viewer);
    PetscViewerDestroy(&viewer);

    // Read RHS
    Vec b, x;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "rhs.petsc", FILE_MODE_READ, &viewer);
    VecCreate(PETSC_COMM_WORLD, &b);
    VecLoad(b, viewer);
    PetscViewerDestroy(&viewer);

    // Read initial guess (if exists)
    VecDuplicate(b, &x);
    FILE *fp = fopen("x0.petsc", "r");
    if (fp) {
        fclose(fp);
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, "x0.petsc", FILE_MODE_READ, &viewer);
        VecLoad(x, viewer);
        PetscViewerDestroy(&viewer);
    }

    // Solve
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSolve(ksp, b, x);

    // Write solution
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, "sol.petsc", FILE_MODE_WRITE, &viewer);
    VecView(x, viewer);
    PetscViewerDestroy(&viewer);

    // Print convergence info
    PetscInt its;
    PetscReal rnorm;
    KSPGetIterationNumber(ksp, &its);
    KSPGetResidualNorm(ksp, &rnorm);
    PetscPrintf(PETSC_COMM_WORLD, "KSP iterations: %d, residual norm: %e\n", its, rnorm);

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}
"""


def _ensure_petsc_solver(petsc_dir=None):
    """
    Compile the PETSc C solver if not already built.
    Returns path to the executable, or None if PETSc not found.
    """
    solver_dir = os.path.join(os.path.dirname(__file__), '_petsc_build')
    exe_path = os.path.join(solver_dir, 'petsc_solve')

    if os.path.isfile(exe_path):
        return exe_path

    # Try to find PETSc
    petsc_dir = petsc_dir or os.environ.get('PETSC_DIR')
    petsc_arch = os.environ.get('PETSC_ARCH', '')

    if not petsc_dir or not os.path.isdir(petsc_dir):
        return None

    os.makedirs(solver_dir, exist_ok=True)
    src_path = os.path.join(solver_dir, 'petsc_solve.c')

    with open(src_path, 'w') as f:
        f.write(_PETSC_C_SOURCE)

    # Compile using PETSc makefile system
    include_dir = os.path.join(petsc_dir, petsc_arch, 'include') if petsc_arch else os.path.join(petsc_dir, 'include')
    lib_dir = os.path.join(petsc_dir, petsc_arch, 'lib') if petsc_arch else os.path.join(petsc_dir, 'lib')

    cmd = (
        f"cc -o {exe_path} {src_path} "
        f"-I{os.path.join(petsc_dir, 'include')} "
        f"-I{include_dir} "
        f"-L{lib_dir} -lpetsc -lm"
    )

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return exe_path
    except subprocess.CalledProcessError:
        return None


# ── Main solve interface ─────────────────────────────────────────────────

def petsc_solve(K, rhs, x0=None, solver_type='gmres', precond='ilu',
                rtol=1e-8, atol=1e-12, max_iter=1000, petsc_dir=None,
                verbose=False):
    """
    Solve K @ x = rhs using PETSc (subprocess) with fallback to scipy.

    Parameters
    ----------
    K : scipy sparse matrix
        System matrix (n x n).
    rhs : (n,) array
        Right-hand side vector.
    x0 : (n,) array or None
        Initial guess (warm start). None → zero.
    solver_type : str
        PETSc KSP type: 'cg', 'gmres', 'minres', 'bcgs', etc.
    precond : str
        PETSc PC type: 'ilu', 'icc', 'jacobi', 'gamg' (AMG), 'none'.
    rtol, atol : float
        Relative and absolute tolerances.
    max_iter : int
        Maximum Krylov iterations.
    petsc_dir : str or None
        Path to PETSc installation. Uses $PETSC_DIR if None.
    verbose : bool
        Print PETSc solver output.

    Returns
    -------
    x : (n,) array
        Solution vector.
    info : dict
        {'method': 'petsc'|'scipy', 'converged': bool, ...}
    """
    exe = _ensure_petsc_solver(petsc_dir)

    if exe is None:
        # Fallback to scipy
        if verbose:
            print("PETSc not available, falling back to scipy.sparse.linalg.spsolve")
        x = spsolve(csr_matrix(K), rhs)
        return x, {'method': 'scipy', 'converged': True}

    # PETSc solve via subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_petsc_mat(os.path.join(tmpdir, 'K.petsc'), K)
        _write_petsc_vec(os.path.join(tmpdir, 'rhs.petsc'), rhs)

        if x0 is not None:
            _write_petsc_vec(os.path.join(tmpdir, 'x0.petsc'), x0)

        cmd = [
            exe,
            f'-ksp_type', solver_type,
            f'-pc_type', precond,
            f'-ksp_rtol', str(rtol),
            f'-ksp_atol', str(atol),
            f'-ksp_max_it', str(max_iter),
        ]

        result = subprocess.run(
            cmd, cwd=tmpdir, capture_output=True, text=True
        )

        if verbose and result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0:
            if verbose:
                print(f"PETSc failed: {result.stderr}")
            # Fallback
            x = spsolve(csr_matrix(K), rhs)
            return x, {'method': 'scipy_fallback', 'converged': True,
                        'petsc_error': result.stderr}

        sol_path = os.path.join(tmpdir, 'sol.petsc')
        x = _read_petsc_vec(sol_path)

    return x, {'method': 'petsc', 'converged': True,
               'solver': solver_type, 'precond': precond}


def scipy_solve(K, rhs, x0=None, **kwargs):
    """
    Direct scipy solve (no PETSc). Always available.

    Parameters
    ----------
    K : scipy sparse matrix
    rhs : (n,) array
    x0 : ignored (spsolve is direct)

    Returns
    -------
    x : (n,) array
    info : dict
    """
    x = spsolve(csr_matrix(K), rhs)
    return x, {'method': 'scipy', 'converged': True}
