#!/usr/bin/env python3

import pprint
import params
import os, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import dill
import itertools
import manage_db
import pandas as pd

# Merges dictionaries with shared keys into single dictionary with same
# keys and values as list of values across all merged dictionaries.
# Helper function for "convert_to_dataframe"
def merge_dicts(list_of_dicts,repfac):
    new_dict = {}
    for d in list_of_dicts:
        for key in d.keys():
            new_dict.setdefault(key,[])
            for j in range(repfac):
                new_dict[key].append(d[key])
    return new_dict


# Convert information from a database for a single parameter set to
# a pandas DataFrame.
def convert_to_dataframe(db_filename):
    xtalk = merge_dicts(manage_db.get_formatted(db_filename,"xtalk"),1)
    df = pd.DataFrame(xtalk)
    nentries = len(df)

    parameters = manage_db.get_formatted(db_filename,"parameters")
    if len(parameters) > 1:
        print("error: can only convert to dataframe for databases where all networks share same parameters")
        sys.exit()
    df_parameters = pd.DataFrame(merge_dicts(parameters,nentries))
    df_networks = pd.DataFrame(merge_dicts(manage_db.get_formatted(db_filename,"networks"),nentries))

    df = df.join(df_parameters)
    df = df.join(df_networks)
    return df


# Convert databases to pandas DataFrames and concatenate.
def combine_databases(db_filenames):
    df = []
    for db_filename in db_filenames:
        if len(df) == 0:
            df = convert_to_dataframe(db_filename)
        else:
            df = pd.concat([df,convert_to_dataframe(db_filename)])
    return df


# HOW DOES CHROMATIN ADVANTAGE OVER TF SCALE WITH GENOME SIZE?
# - boxplot patterning error/gene vs. "genome size" at different specificities

def error_by_gene(df):
    return df["fun"].div(df.M_GENE,axis=0)

# TODO: separate functions for each type of desired plot, wrapper function to overlay them
def boxplot_groupby(filename,df,cols,f):
    gb = df.groupby(cols)
    gb_f = gb.apply(f)
    gb_f = [list(gb_f[key]) for key in gb.groups.keys()]
    
    plt.rcParams.update({'font.size':24})
    fig, ax = plt.subplots(figsize=(48,24))
    ax.boxplot(gb_f,labels=gb.groups.keys())
    plt.savefig(filename)


# TODO: rewrite to use DataFrames
# idealized curve: given a concentration of noncogate factors,
# what specific layer 2 concentration would give exactly the
# target expression level? (ignoring layer 1 binding)
def calc_optimal_cS_for_fixed_other_cNS(db_filename):

    def objective_fn(c_NS,c_S,target):
        return tf_pr_bound(c_NS+c_S,c_S) - target

    for ii, target in enumerate(res["target_pattern"]):
        if res["layer2_cS_for_fixed_other_cNS"][ii] == None:
            layer2_cS_given_noncog = np.zeros(len(res["target_pattern"][0]))
            for ii_gene, target_level in enumerate(target):
                cur_C_NS = np.sum(res["optimized_input"][ii]) - res["optimized_input"][ii][ii_gene]
                layer2_cS_given_noncog[ii_gene] = scipy.optimize.fsolve(lambda x: objective_fn(cur_C_NS,x,target),res["optimized_input"][ii][ii_gene])
        cur.execute(f"UPDATE xtalk SET layer2_cS_for_fixed_other_cNS = {layer2_cS_given_noncog.tobytes()} WHERE network_rowid = {res['network_rowid'][ii]}")
