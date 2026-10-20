"""Node dataset loading with explicit cache checks.

The baseline reuses existing PyG cache directories and must not silently
download datasets during experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import (
    Actor,
    HeterophilousGraphDataset,
    Planetoid,
    WebKB,
    WikiCS,
    WikipediaNetwork,
)


DATASET_ALIASES: dict[str, str] = {
    "actor": "Actor",
    "Actor": "Actor",
    "chameleon": "chameleon",
    "Chameleon": "chameleon",
    "chameleon_filtered": "chameleon_filtered",
    "ChameleonFiltered": "chameleon_filtered",
    "squirrel": "squirrel",
    "Squirrel": "squirrel",
    "texas": "Texas",
    "Texas": "Texas",
    "wisconsin": "Wisconsin",
    "Wisconsin": "Wisconsin",
    "cora": "Cora",
    "Cora": "Cora",
    "citeseer": "CiteSeer",
    "CiteSeer": "CiteSeer",
    "pubmed": "PubMed",
    "PubMed": "PubMed",
    "wikics": "WikiCS",
    "wiki_cs": "WikiCS",
    "WikiCS": "WikiCS",
    "minesweeper": "minesweeper",
    "Minesweeper": "minesweeper",
    "amazon_ratings": "amazon_ratings",
    "amazon-ratings": "amazon_ratings",
    "amazon-rating": "amazon_ratings",
    "AmazonRatings": "amazon_ratings",
    "Amazon-Ratings": "amazon_ratings",
    "Amazon-Rating": "amazon_ratings",
    "roman_empire": "roman_empire",
    "roman-empire": "roman_empire",
    "RomanEmpire": "roman_empire",
    "Roman-empire": "roman_empire",
    "Roman-Empire": "roman_empire",
    "tolokers": "tolokers",
    "Tolokers": "tolokers",
    "questions": "questions",
    "Questions": "questions",
}


@dataclass(frozen=True)
class LoadedNodeDataset:
    name: str
    data: Data
    num_features: int
    num_classes: int


def resolve_dataset_name(name: str) -> str:
    key = name.strip()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(
        f"Unsupported dataset {name!r}. Supported names: "
        "Actor, chameleon, squirrel, Texas, Wisconsin, minesweeper, "
        "amazon_ratings, roman_empire, "
        "tolokers, questions, Cora, CiteSeer, PubMed, WikiCS."
    )


def _require_processed(path: Path, *, download_if_missing: bool) -> None:
    if path.is_dir() and any(path.iterdir()):
        return
    if download_if_missing:
        return
    raise FileNotFoundError(
        f"Processed dataset cache not found: {path}. "
        "Set download_if_missing=true only when intentional."
    )


def _select_dataset_root(
    candidates: list[tuple[Path, Path]], *, download_if_missing: bool
) -> Path:
    """Choose the first existing PyG cache layout and return its dataset root."""

    for dataset_root, processed_path in candidates:
        if processed_path.is_dir() and any(processed_path.iterdir()):
            return dataset_root
    dataset_root, processed_path = candidates[0]
    _require_processed(processed_path, download_if_missing=download_if_missing)
    return dataset_root


def load_node_dataset(
    name: str,
    root: str | Path,
    *,
    download_if_missing: bool = False,
    normalize_features: bool | None = None,
) -> LoadedNodeDataset:
    """Load a supported node dataset from an existing PyG cache."""

    canonical = resolve_dataset_name(name)
    root_path = Path(root).expanduser()
    if normalize_features is None:
        normalize_features = canonical != "WikiCS"
    transform = T.NormalizeFeatures() if bool(normalize_features) else None

    if canonical == "chameleon_filtered":
        from data.filtered_wikipedia import load_filtered_chameleon

        data = load_filtered_chameleon(root_path)
        if transform is not None:
            data = transform(data)
        return LoadedNodeDataset(
            name=canonical, data=data,
            num_features=int(data.num_features),
            num_classes=int(data.y.unique().numel()),
        )
    if canonical in {"Cora", "CiteSeer", "PubMed"}:
        dataset_root = _select_dataset_root(
            [
                (root_path / "Planetoid", root_path / "Planetoid" / canonical / "processed"),
                (root_path, root_path / canonical / "processed"),
            ],
            download_if_missing=download_if_missing,
        )
        dataset = Planetoid(root=str(dataset_root), name=canonical, transform=transform)
    elif canonical == "Actor":
        _require_processed(root_path / "Actor" / "processed", download_if_missing=download_if_missing)
        dataset = Actor(root=str(root_path / "Actor"), transform=transform)
    elif canonical in {"chameleon", "squirrel"}:
        dataset_root = _select_dataset_root(
            [
                (
                    root_path / "WikipediaNetwork",
                    root_path / "WikipediaNetwork" / canonical / "geom_gcn" / "processed",
                ),
                (root_path, root_path / canonical / "geom_gcn" / "processed"),
            ],
            download_if_missing=download_if_missing,
        )
        dataset = WikipediaNetwork(
            root=str(dataset_root),
            name=canonical,
            transform=transform,
        )
    elif canonical in {"Texas", "Wisconsin"}:
        dataset_root = _select_dataset_root(
            [
                (
                    root_path / "WebKB",
                    root_path / "WebKB" / canonical / "processed",
                ),
                (root_path, root_path / canonical / "processed"),
            ],
            download_if_missing=download_if_missing,
        )
        dataset = WebKB(
            root=str(dataset_root),
            name=canonical,
            transform=transform,
        )
    elif canonical == "WikiCS":
        _require_processed(root_path / "WikiCS" / "processed", download_if_missing=download_if_missing)
        # Match GP2F: use the original WikiCS GloVe features without row
        # normalization while retaining this repo's dedicated cache directory.
        # GP2F pins PyG 2.7, where the omitted flag defaults to an undirected
        # WikiCS graph. Keep that behavior explicit so future PyG changes do
        # not alter the experiment protocol.
        dataset = WikiCS(
            root=str(root_path / "WikiCS"),
            transform=transform,
            is_undirected=True,
        )
    elif canonical in {"minesweeper", "amazon_ratings", "roman_empire", "tolokers", "questions"}:
        dataset_root = _select_dataset_root(
            [
                (
                    root_path / "HeterophilousGraphDataset",
                    root_path / "HeterophilousGraphDataset" / canonical / "processed",
                ),
                (root_path, root_path / canonical / "processed"),
            ],
            download_if_missing=download_if_missing,
        )
        dataset = HeterophilousGraphDataset(
            root=str(dataset_root),
            name=canonical,
            transform=transform,
        )
    else:
        raise AssertionError(f"Unhandled canonical dataset: {canonical}")

    data = dataset[0]
    return LoadedNodeDataset(
        name=canonical,
        data=data,
        num_features=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
    )
