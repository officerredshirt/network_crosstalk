#!/usr/bin/env bash

CURDIR=`pwd`
cd src/
python3 combine_df_to_csv.py ../cluster_2023-04*/res/ ../cluster_2023-05*/res/
cd $CURDIR
