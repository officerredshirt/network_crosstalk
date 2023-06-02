#!/usr/bin/env bash

for d in cluster_2023-05-22*; do
    echo "Generating plots for folder $d..."
    python3 ./src/gen_plots.py $d
    echo "Done."
done
