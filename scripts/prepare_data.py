"""Explicitly download the main-experiment datasets into the local cache."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data import load_node_dataset
parser = argparse.ArgumentParser()
parser.add_argument("--datasets", nargs="+", default=["Cora","CiteSeer","PubMed","WikiCS","Actor","Chameleon","Minesweeper"])
args = parser.parse_args()
for name in args.datasets:
    dataset = load_node_dataset(name, ROOT/"datasets", download_if_missing=True)
    print(name, dataset.data.num_nodes, dataset.num_features, dataset.num_classes)
