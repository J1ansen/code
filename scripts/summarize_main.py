"""Summarize the declared checkpoint selections, never oracle test maxima."""
import json
from pathlib import Path
import statistics
import argparse
import yaml
ROOT = Path(__file__).resolve().parents[1]
entries=json.loads((ROOT/"configs/main/protocol.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--strategy",choices=["GRACE","DGI","GraphMAE"],default="GRACE")
parser.add_argument("--dataset",choices=["all",*[entry["dataset"] for entry in entries]],default="all")
args = parser.parse_args()
for entry in entries:
    if args.dataset != "all" and args.dataset != entry["dataset"]:
        continue
    output = ROOT/entry["output_dir"]
    if args.strategy != "GRACE":
        output = ROOT/"outputs/main_validation"/args.strategy.lower()/entry["dataset"]
    paths=sorted(output.rglob("summary.json"))
    if not paths:
        continue
    if len(paths)!=1:
        raise ValueError(f"Multiple runs for {entry['dataset']}; select an output folder explicitly")
    runs=json.loads(paths[0].read_text())["runs"]
    config_paths = list(paths[0].parent.glob("seed_*/trial_*/config.yaml"))
    if len(runs) != entry["trials"] or len(config_paths) != entry["trials"]:
        raise ValueError(f"Expected {entry['trials']} completed trials for {entry['dataset']}")
    for path in config_paths:
        cfg = yaml.safe_load(path.read_text())
        if cfg["training"]["early_stop_metric"] != (entry.get("selection_metric") or entry["validation_metric"]):
            raise ValueError(f"Checkpoint selection mismatch: {path}")
        if cfg.get("release", {}).get("protocol") != entry.get("protocol", "validation_selected_v1"):
            raise ValueError(f"Run does not declare the current protocol: {path}")
        if cfg["pretrained"].get("strategy","GRACE") != args.strategy:
            raise ValueError(f"Pretraining strategy mismatch: {path}")
    values=[100*r[entry["report_selection"]][entry["report_metric"]] for r in runs]
    print(f"{entry['dataset']}: {statistics.mean(values):.2f} +/- {statistics.stdev(values):.2f} ({len(values)} trials)")
