cd ../../

onager launch \
    --backend gridengine \
    --jobname tmaze_corridor_sweep_hallway_discrep \
    --mem 1 \
    --cpus 2 \
    --duration 0-03:00:00 \
    --venv venv \
    -q '*@@mblade12'\
