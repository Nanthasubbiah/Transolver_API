"""
ML warm start: predict initial displacement u0 using a trained Transolver
model, then feed it to the NR solver to reduce iterations.

Pipeline: nodes → Transolver → u_pred → solver.solve(warm_start_u=u_pred)

File location: elasticity_api/ml/warm_start.py
"""

import os
import numpy as np
from model.Transolver_Irregular_Mesh import Model
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WarmStarter:
    """
    Loads a trained Transolver model and UnitTransformer normalizer,
    predicts displacement field from node coordinates.

    Parameters
    ----------
    checkpoint_path : str
        Path to model .pt file.
    normalizer_data : torch.Tensor or None
        Training displacement tensor (ntrain, n_nodes, 2) used to fit
        UnitTransformer. If None, no denormalization is applied.
    model_config : dict
        Model hyperparameters matching training config:
        n_hidden, n_layers, n_heads, mlp_ratio, dropout,
        slice_num, ref, unified_pos.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, checkpoint_path, normalizer_data=None,
                 model_config=None, device='cpu'):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ML warm start. "
                              "Run: pip install torch")

        self.device = device
        self.config = model_config or {}

        # Build model
        self.model = self._build_model()
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Normalizer
        self.normalizer = None
        if normalizer_data is not None:
            from normalizer import UnitTransformer
            self.normalizer = UnitTransformer(normalizer_data)
            if device == 'cuda':
                self.normalizer.cuda()

    def _build_model(self):
        """Build Transolver model from config."""
        try:
            from Transolver_Irregular_Mesh import Model
        except ImportError:
            raise ImportError(
                "Transolver_Irregular_Mesh.py must be importable. "
                "Add its directory to sys.path or PYTHONPATH."
            )

        cfg = self.config
        model = Model(
            space_dim=2,
            n_layers=cfg.get('n_layers', 3),
            n_hidden=cfg.get('n_hidden', 64),
            dropout=cfg.get('dropout', 0.0),
            n_head=cfg.get('n_heads', 4),
            Time_Input=False,
            mlp_ratio=cfg.get('mlp_ratio', 1),
            fun_dim=0,
            out_dim=2,
            slice_num=cfg.get('slice_num', 32),
            ref=cfg.get('ref', 8),
            unified_pos=cfg.get('unified_pos', 0),
        )

        return model.to(self.device)

    def predict(self, nodes):
        """
        Predict displacement field from node coordinates.

        Parameters
        ----------
        nodes : (n_nodes, 2) array
            Nodal coordinates.

        Returns
        -------
        u_pred : (n_nodes, 2) array
            Predicted displacement [ux, uy] per node.
        """
        xy = torch.tensor(nodes, dtype=torch.float32).unsqueeze(0)  # (1, N, 2)
        xy = xy.to(self.device)

        with torch.no_grad():
            pred = self.model(xy, None)  # (1, N, 2)
            if self.normalizer is not None:
                pred = self.normalizer.decode(pred)
            u_pred = pred.squeeze(0).cpu().numpy()

        return u_pred


def load_warm_starter(checkpoint_path, train_disp_path=None,
                      model_config=None, device='cpu'):
    """
    Convenience function to create a WarmStarter.

    Parameters
    ----------
    checkpoint_path : str
        Path to .pt checkpoint.
    train_disp_path : str or None
        Path to training displacement .npy file for normalizer fitting.
        Expected shape: (n_nodes, 2, n_samples), transposed to (n_train, n_nodes, 2).
    model_config : dict or None
    device : str

    Returns
    -------
    WarmStarter instance.
    """
    if train_disp_path is None:
        print("⚠ No normalizer — assuming physics-loss model with physical-unit outputs")
    normalizer_data = None
    if train_disp_path and os.path.isfile(train_disp_path):
        disp = np.load(train_disp_path)  # (972, 2, 2000)
        disp_t = torch.tensor(disp, dtype=torch.float32).permute(2, 0, 1)
        ntrain = model_config.get('ntrain', 1800) if model_config else 1800
        normalizer_data = disp_t[:ntrain]

    return WarmStarter(checkpoint_path, normalizer_data, model_config, device)
