#!/usr/bin/env bash

for i in {100,150,250}; do
	for j in {3,5,8}; do
		DIRNAME="gene${i}_maxclust${j}"
		if ! test -d "$DIRNAME"; then
			mkdir "$DIRNAME"
		fi

		if ! test -f "$DIRNAME/sm.sh"; then
			cp sm.sh "$DIRNAME/sm.sh"
		fi

		if ! test -f "$DIRNAME/Snakefile"; then
			cp Snakefile "$DIRNAME/Snakefile"
			sed -i "s/^DIR = \"\/scratch\/mindylp\/\"/DIR = \"\/scratch\/mindylp\/${DIRNAME}\/\"/" "$DIRNAME/Snakefile"
		fi

		if ! test -d "$DIRNAME/src"; then
			cp -r src "$DIRNAME"
			sed -i "s/GENES_PER_CLUSTER = [0-9]\+/GENES_PER_CLUSTER = 10/" "$DIRNAME/src/params.py"
			sed -i "s/N_PF = [0-9]\+/N_PF = $(($i/10))/" "$DIRNAME/src/params.py"
			sed -i "s/ignore_off_during_optimization = True/ignore_off_during_optimization = False/" "$DIRNAME/src/params.py"
			sed -i "s/MIN_EXPRESSION = [0-9\.]\+/MIN_EXPRESSION = 0.09/" "$DIRNAME/src/params.py"
			sed -i "s/NUM_TARGETS = [0-9]\+/NUM_TARGETS = 100/" "$DIRNAME/src/params.py"
			sed -i "s/MIN_CLUSTERS_ACTIVE = [0-9]\+/MIN_CLUSTERS_ACTIVE = ${j}/" "$DIRNAME/src/params.py"
			sed -i "s/MAX_CLUSTERS_ACTIVE = [0-9]\+/MAX_CLUSTERS_ACTIVE = ${j}/" "$DIRNAME/src/params.py"
		fi
	done
done
