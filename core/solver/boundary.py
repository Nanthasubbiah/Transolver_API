"""
Boundary conditions manager for 2D FEM.

Stores Dirichlet (fixed DOFs), Neumann (traction edges), and symmetry
conditions. The assembler and solver query this object during setup.

Usage
-----
    bc = BoundaryConditions(nodes)
    bc.add_symmetry(axis='x', coord=0.5, tol=0.02)   # ux=0 at x=0.5
    bc.add_symmetry(axis='y', coord=0.5, tol=0.02)   # uy=0 at y=0.5
    bc.add_dirichlet(node_ids=[0, 1], component='x', value=0.0)
    bc.add_traction(node_ids, tx=1e4, ty=0.0)         # outward pressure

    fixed_dofs, free_dofs = bc.get_dofs()
    f_ext = bc.build_f_ext(elements)
"""

import numpy as np


class BoundaryConditions:
    """
    Container for boundary conditions on a 2D mesh.

    Parameters
    ----------
    nodes : (n_nodes, 2) array
        Nodal coordinates.
    """

    def __init__(self, nodes):
        self.nodes = np.asarray(nodes)
        self.n_nodes = len(nodes)
        self.n_dof = 2 * self.n_nodes

        self._fixed = {}       # {dof_index: prescribed_value}
        self._tractions = []   # list of {'nodes': set, 'traction': (tx, ty)}

    # ── Dirichlet ────────────────────────────────────────────────────────

    def add_dirichlet(self, node_ids, component, value=0.0):
        """
        Fix a displacement component at specified nodes.

        Parameters
        ----------
        node_ids : array-like of int
            Node indices to constrain.
        component : 'x' or 'y'
            Which displacement component to fix.
        value : float
            Prescribed displacement value (default 0).
        """
        offset = 0 if component == 'x' else 1
        for n in node_ids:
            self._fixed[2 * n + offset] = value

    def fix_isolated_nodes(self, elements):
        """
        Fix all DOFs of nodes not referenced by any element.
        Prevents singular stiffness matrix.

        Parameters
        ----------
        elements : (n_elem, 3) int array
        """
        used = set(np.unique(elements))
        for n in range(self.n_nodes):
            if n not in used:
                self._fixed[2 * n] = 0.0
                self._fixed[2 * n + 1] = 0.0

    # ── Symmetry (shorthand) ─────────────────────────────────────────────

    def add_symmetry(self, axis, coord, tol=0.02):
        """
        Apply symmetry BC: fix the normal displacement component
        for all nodes near the symmetry plane.

        Parameters
        ----------
        axis : 'x' or 'y'
            Symmetry plane normal direction.
            'x' → fixes ux=0 for nodes near x=coord.
            'y' → fixes uy=0 for nodes near y=coord.
        coord : float
            Location of the symmetry plane.
        tol : float
            Tolerance for detecting nodes on the plane.
        """
        if axis == 'x':
            ids = np.where(np.abs(self.nodes[:, 0] - coord) < tol)[0]
            self.add_dirichlet(ids, 'x', value=0.0)
        elif axis == 'y':
            ids = np.where(np.abs(self.nodes[:, 1] - coord) < tol)[0]
            self.add_dirichlet(ids, 'y', value=0.0)
        else:
            raise ValueError(f"axis must be 'x' or 'y', got '{axis}'")

    # ── Neumann (traction) ───────────────────────────────────────────────

    def add_traction(self, node_ids, tx=0.0, ty=0.0):
        """
        Apply traction (force per unit length) on boundary edges
        connecting the specified nodes.

        Parameters
        ----------
        node_ids : array-like of int
            Nodes on this boundary segment.
        tx, ty : float
            Traction components (force/length in x and y).
        """
        self._tractions.append({
            'nodes': set(node_ids),
            'traction': np.array([tx, ty])
        })

    def add_outer_traction(self, traction, tol=0.02):
        """
        Convenience: apply outward-normal traction on all 4 outer edges
        of a rectangular domain. Matches the unit-cell BC from the dataset.

        Parameters
        ----------
        traction : float
            Magnitude of outward traction T.
        tol : float
            Tolerance for detecting boundary nodes.
        """
        x = self.nodes[:, 0]
        y = self.nodes[:, 1]
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        top   = np.where(y > ymax - tol)[0]
        bot   = np.where(y < ymin + tol)[0]
        right = np.where(x > xmax - tol)[0]
        left  = np.where(x < xmin + tol)[0]

        self.add_traction(top,   tx=0.0,       ty=traction)
        self.add_traction(bot,   tx=0.0,       ty=-traction)
        self.add_traction(right, tx=traction,   ty=0.0)
        self.add_traction(left,  tx=-traction,  ty=0.0)

    # ── Query interface ──────────────────────────────────────────────────

    def get_dofs(self):
        """
        Returns
        -------
        fixed_dofs : sorted int array
        free_dofs : sorted int array
        """
        fixed = np.array(sorted(self._fixed.keys()), dtype=int)
        free = np.setdiff1d(np.arange(self.n_dof), fixed)
        return fixed, free

    def get_prescribed_values(self):
        """
        Returns
        -------
        dict : {dof_index: prescribed_value}
        """
        return dict(self._fixed)

    def build_f_ext(self, elements):
        """
        Build the external force vector from all Neumann BCs.

        Parameters
        ----------
        elements : (n_elem, 3) int array

        Returns
        -------
        f_ext : (n_dof,) array
        """
        f_ext = np.zeros(self.n_dof)
        for bc in self._tractions:
            bc_nodes = bc['nodes']
            tx, ty = bc['traction']
            for elem in elements:
                for i in range(3):
                    n1, n2 = elem[i], elem[(i + 1) % 3]
                    if n1 in bc_nodes and n2 in bc_nodes:
                        L = np.linalg.norm(self.nodes[n1] - self.nodes[n2])
                        f_ext[2 * n1]     += tx * L / 2
                        f_ext[2 * n1 + 1] += ty * L / 2
                        f_ext[2 * n2]     += tx * L / 2
                        f_ext[2 * n2 + 1] += ty * L / 2
        return f_ext
