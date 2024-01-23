#!/usr/bin/env bash

CURDIR=`pwd`
cd src/
python3 combine_df_to_pq.py ../cluster_2023-07*/res/ ../cluster_2023-08*/res/ ../cluster_2023-10*/res/ ../cluster_2023-11*/res/ ../cluster_2023-12*/res/ ../cluster_2024-01*/res/
cd $CURDIR
