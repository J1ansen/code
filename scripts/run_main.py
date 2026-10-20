"""Run the validation-selected main-experiment configurations."""
import argparse
from pathlib import Path
import subprocess
import sys
import yaml
ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["all","cora","citeseer","pubmed","wikics","actor","chameleon","minesweeper"], default="all")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--strategy", choices=["GRACE","DGI","GraphMAE"], default="GRACE")
args = parser.parse_args()
names = ["cora","citeseer","pubmed","wikics","actor","chameleon","minesweeper"] if args.dataset=="all" else [args.dataset]
targets = [yaml.safe_load((ROOT/f"configs/main/{name}.yaml").read_text())["experiment"]["target_dataset"] for name in names]
subprocess.run([sys.executable,"scripts/prepare_data.py","--datasets",*dict.fromkeys(["Cora",*targets])],cwd=ROOT,check=True)
for name in names:
    config_path = ROOT / f"configs/main/{name}.yaml"
    if args.smoke or args.strategy != "GRACE":
        config = yaml.safe_load(config_path.read_text())
        key = args.strategy.lower()
        config["pretrained"].update(strategy=args.strategy,checkpoint_path=f"checkpoints/cora_{key}.pt",
            pretraining_config_path=f"pretrain_GNN/cora_{key}.yaml")
        if args.strategy != "GRACE":
            config["training"]["output_dir"] = f"outputs/main_validation/{key}/{name}"
        if args.smoke:
            config["pretrained"].update(checkpoint_path=f"outputs/smoke_pretraining/cora_{key}.pt",
                pretraining_overrides={"epochs":2,"patience":0})
        config_path = ROOT / "outputs/resolved_configs" / key / ("smoke" if args.smoke else "formal") / f"{name}.yaml"
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    command = [sys.executable,"experiments/run_main_experiment.py","--config",str(config_path)]
    if args.smoke:
        command += ["--epochs","1","--trials_per_seed","1","--output_dir",f"outputs/smoke_validation/{args.strategy.lower()}/{name}"]
    subprocess.run(command, cwd=ROOT, check=True)
