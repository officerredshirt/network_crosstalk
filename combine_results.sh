#!/usr/bin/env bash

CURDIR=`pwd`
cd src/
python3 combine_df_to_pq.py ../cluster_res/cluster_2023-07*/res/ ../cluster_res/cluster_2023-08*/res/ ../cluster_res/cluster_2023-10*/res/ ../cluster_res/cluster_2023-11*/res/ ../cluster_res/cluster_2023-12*/res/ ../cluster_res/cluster_2024-01*/res/ ../cluster_res/cluster_2024-03*/res/ ../cluster_res/cluster_2024-05*/res/
cd $CURDIR
