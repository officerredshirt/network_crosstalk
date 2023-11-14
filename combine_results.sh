#!/usr/bin/env bash

CURDIR=`pwd`
cd src/
python3 combine_df_to_hdf.py ../cluster_2023-07*/res/ ../cluster_2023-08*/res/ ../cluster_2023-10*/res/
cd $CURDIR
