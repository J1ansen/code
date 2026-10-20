"""Official duplicate-filtered Chameleon, isolated from WikipediaNetwork caches."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

SOURCE_COMMIT = "a431395582e929d88271309716bea4fe24ce6318"
SOURCE_URL = (
    "https://raw.githubusercontent.com/yandex-research/heterophilous-graphs/"
    f"{SOURCE_COMMIT}/data/chameleon_filtered.npz"
)


def load_filtered_chameleon(root: Path) -> Data:
    path = Path(root) / "FilteredWikipedia/chameleon_filtered/raw/chameleon_filtered.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Download {SOURCE_URL} explicitly to {path}")
    with np.load(path, allow_pickle=False) as arrays:
        x = torch.tensor(arrays["node_features"], dtype=torch.float32)
        y = torch.tensor(arrays["node_labels"], dtype=torch.long)
        edges = torch.tensor(arrays["edges"], dtype=torch.long)
        if x.ndim != 2 or y.shape != (len(x),) or edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("Invalid filtered Chameleon array shapes")
        if not torch.isfinite(x).all() or (edges.numel() and (edges.min() < 0 or edges.max() >= len(x))):
            raise ValueError("Invalid features or edge indices")
        data = Data(x=x, y=y, edge_index=to_undirected(edges.t().contiguous(), num_nodes=len(x)))
        # Retain official masks for provenance; the experiment uses generated splits.
        masks = []
        for name in ("train", "val", "test"):
            mask = torch.tensor(arrays[f"{name}_masks"], dtype=torch.bool).t().contiguous()
            if mask.ndim != 2 or mask.shape[0] != len(x):
                raise ValueError("Invalid official split masks")
            setattr(data, f"{name}_mask", mask)
            masks.append(mask)
        if ((masks[0].int() + masks[1].int() + masks[2].int()) > 1).any():
            raise ValueError("Overlapping official split masks")
    return data
