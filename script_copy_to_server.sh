#!/usr/bin/env bash

SERVER_DIR="/g/crocker/mindylp/network_crosstalk/"
#SERVER_DIR="FAKE_SERVER_DIR/"

cluster_dirs=$(find -type d -name "cluster*")
server_cluster_dirs=$(find ${SERVER_DIR} -type d -name "cluster*" -printf "%f\n")

for dir in $cluster_dirs; do
	dest_basename="$(basename $dir)"
	perfect_match_exists=false

	dir_matches=$(find ${SERVER_DIR} -type d -name "*$dest_basename*")
	if ! [[ "$dir_matches" == "" ]]; then
		for dir_match in $dir_matches; do
			if cmp -s "${dir}/res/local_db.db" "${dir_match}/res/local_db.db"; then
				echo "$dest_basename is identical to ${dir_match}; skipping..."
				perfect_match_exists=true
				break
			fi
		done

		if ! $perfect_match_exists; then
			x=1
			while [[ "$dir_matches" == *"${dest_basename}-${x}"* ]]; do
				((x++))
			done
			dest_basename="${dest_basename}-${x}"
		fi
	fi

	if ! $perfect_match_exists; then
		DEST_DIR="${SERVER_DIR}${dest_basename}"
		echo "Copying $dir to $DEST_DIR..."
		mkdir $DEST_DIR
		cp -r $dir/* $DEST_DIR
	fi
done
