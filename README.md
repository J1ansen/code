# Running the Code

Run the following commands from the repository root.

```bash
# Install dependencies and run all datasets with GRACE pretraining.
bash run.sh --install --strategy GRACE --dataset all

# Run with DGI or GraphMAE pretraining.
bash run.sh --strategy DGI --dataset all
bash run.sh --strategy GraphMAE --dataset all

# Quick check on Cora.
bash run.sh --strategy GRACE --dataset cora --smoke
```

Datasets are prepared automatically. Missing source checkpoints are pretrained
automatically and reused for subsequent runs.

Replace `all` with `cora`, `citeseer`, `pubmed`, `wikics`, `actor`, `chameleon`,
or `minesweeper` to run a single dataset. Set `PYTHON_BIN` to choose a Python
interpreter, for example:

```bash
PYTHON_BIN=/path/to/python bash run.sh --strategy GRACE --dataset cora
```

For all options, run `bash run.sh --help`.
