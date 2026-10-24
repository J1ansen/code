"""Official duplicate-filtered Wikipedia graphs in an isolated cache."""

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
    return load_filtered_wikipedia(root, "chameleon_filtered")


def load_filtered_wikipedia(root: Path, name: str, *, download_if_missing: bool = False) -> Data:
    if name not in {"chameleon_filtered", "squirrel_filtered"}:
        raise ValueError(f"Unknown filtered Wikipedia dataset: {name}")
    path = Path(root) / f"FilteredWikipedia/{name}/raw/{name}.npz"
    if not path.is_file():
        url = SOURCE_URL.replace("chameleon_filtered", name)
        if not download_if_missing:
            raise FileNotFoundError(f"Download {url} explicitly to {path}")
        from urllib.request import urlopen
        import os
        import tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url, timeout=120) as response:
            content = response.read()
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
            tmp.write(content)
            temporary = Path(tmp.name)
        try:
            with np.load(temporary, allow_pickle=False) as arrays:
                if not {"node_features", "node_labels", "edges", "train_masks", "val_masks", "test_masks"}.issubset(arrays.files):
                    raise ValueError("Invalid filtered dataset archive")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
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
