#!/usr/bin/env bash

CURDIR=`pwd`
cd src/
python3 combine_df_to_hdf.py ../cluster_2023-04*/res/ ../cluster_2023-05*/res/ ../cluster_2023-06*/res/ ../cluster_2023-07*/res/
cd $CURDIR
