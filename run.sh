#!/usr/bin/env bash
set -euo pipefail

# Third-party components: GP2F dual-branch adaptation, GRAPHITE-inspired feature
# nodes, PyTorch and PyTorch Geometric. Preserve upstream notices on redistribution.
# No new license is asserted here for third-party components.
# GRACE follows Deep Graph Contrastive Representation Learning (Zhu et al., 2020)
# and GP2F's view generation, with self-loop preservation and view normalization.
# DGI follows Deep Graph Infomax (Velickovic et al., 2019), with the corrected
# graph-wide readout. GraphMAE follows Self-Supervised Masked Graph Autoencoders
# (Hou et al., 2022), with GAT layers, mask tokens, latent re-masking and SCE loss.

# Formal adaptation: 10 trials; validation-accuracy selection except validation
# Macro-F1 for Minesweeper. Squirrel trains learnable Adapters for all 160 epochs
# without early stopping. Source pretraining uses self-supervised training loss.
# Historical records in results/main/ predate this protocol. New weights are
# trained from source; historical table values are not new reproduction results.

usage() {
    cat <<'USAGE'
Usage: bash run.sh [options]

  --strategy GRACE|DGI|GraphMAE|all   Pretraining objective (default: GRACE)
  --dataset all|cora|citeseer|pubmed|wikics|actor|chameleon|squirrel|minesweeper
                                     Target dataset (default: all)
  --install                          Install requirements into the chosen Python
  --smoke                            Two pretraining epochs and one adaptation epoch
  --summary-only                     Summarize completed formal runs without training
  -h, --help                         Show this message

Examples:
  bash run.sh --install --strategy GRACE --dataset all
  bash run.sh --strategy DGI --dataset all
  bash run.sh --strategy GraphMAE --dataset cora --smoke
  bash run.sh --strategy all --summary-only

Set PYTHON_BIN to select a Python interpreter, e.g. PYTHON_BIN=.venv/bin/python.
Datasets are prepared automatically. Missing source checkpoints are pretrained
and cached separately per strategy. Smoke checkpoints never enter the formal cache.
USAGE
}

strategy="GRACE"
dataset="all"
install=0
smoke=0
summary_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy|--dataset)
            if [[ $# -lt 2 ]]; then
                printf 'Missing value for %s\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == "--strategy" ]]; then strategy="$2"; else dataset="$2"; fi
            shift 2
            ;;
        --install) install=1; shift ;;
        --smoke) smoke=1; shift ;;
        --summary-only) summary_only=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$strategy" in
    GRACE|DGI|GraphMAE|all) ;;
    *) printf 'Unsupported strategy: %s\n' "$strategy" >&2; exit 2 ;;
esac
case "$dataset" in
    all|cora|citeseer|pubmed|wikics|actor|chameleon|squirrel|minesweeper) ;;
    *) printf 'Unsupported dataset: %s\n' "$dataset" >&2; exit 2 ;;
esac
if [[ "$smoke" -eq 1 && "$summary_only" -eq 1 ]]; then
    printf '%s\n' '--smoke and --summary-only cannot be combined.' >&2
    exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    printf 'Python not found: %s. Set PYTHON_BIN to your interpreter.\n' "$python_bin" >&2
    exit 2
fi
# Resolve a relative interpreter before changing the working directory.
python_bin="$(command -v "$python_bin")"
if [[ "$python_bin" != /* ]]; then python_bin="$PWD/$python_bin"; fi
cd "$repo_dir"

if [[ "$install" -eq 1 ]]; then
    "$python_bin" -m pip install -r requirements.txt
fi
strategies=("$strategy")
if [[ "$strategy" == "all" ]]; then strategies=(GRACE DGI GraphMAE); fi
for selected in "${strategies[@]}"; do
    if [[ "$summary_only" -eq 0 ]]; then
        command=("$python_bin" scripts/run_main.py --strategy "$selected" --dataset "$dataset")
        if [[ "$smoke" -eq 1 ]]; then command+=(--smoke); fi
        "${command[@]}"
    fi
    if [[ "$smoke" -eq 0 ]]; then
        "$python_bin" scripts/summarize_main.py --strategy "$selected" --dataset "$dataset"
    fi
done
