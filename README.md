# Install dependencies and run all datasets with GRACE pretraining.
bash run.sh --install --strategy GRACE --dataset all

# Run with DGI or GraphMAE pretraining.
bash run.sh --strategy DGI --dataset all
bash run.sh --strategy GraphMAE --dataset all
