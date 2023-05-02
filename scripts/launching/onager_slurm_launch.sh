cd ../../

onager launch \
    --backend slurm \
    --jobname prisoners_dilemma \
    --mem 2 \
    --cpus 2 \
    --duration 0-03:00:00 \
    --venv venv \
#    --tasks-per-node 4 \
