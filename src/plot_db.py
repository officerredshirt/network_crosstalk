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
    df_filename = pd.DataFrame(merge_dicts([{"filename":db_filename}],nentries))

    df = df.join(df_parameters)
    df = df.join(df_networks)
    df = df.join(df_filename)
    df["modulating_concentrations"] = np.nan
    df["modulating_concentrations"] = df["modulating_concentrations"].astype(object)
    return df


# Convert databases to DataFrames and concatenate.  If DataFrame df is
# provided, appends databases to df.
def combine_databases(db_filenames,df=[]):
    for db_filename in db_filenames:
        if len(df) == 0:
            df = convert_to_dataframe(db_filename)
        elif any([db_filename in x for x in df["filename"]]):
            print(f"{db_filename} already in dataframe--skipping...")
        else:
            df = pd.concat([df,convert_to_dataframe(db_filename)])
    df.reset_index(drop=True,inplace=True)
    return df


# HOW DOES CHROMATIN ADVANTAGE OVER TF SCALE WITH GENOME SIZE?
# - boxplot patterning error/gene vs. "genome size" at different specificities

def row_calc_patterning_error(df):
    d = df["output_expression"] - df["target_pattern"]
    return d@d


def patterning_error(df):
    return df.apply(row_calc_patterning_error,axis=1).apply(np.log)
    

def xtalk_by_gene(df):
    d = df["fun"].div(df.M_GENE,axis=0)
    return d.apply(np.log)


def ratio_xtalk_chromatin_tf_by_pair(df):
    tf = df.loc[df["tf_first_layer"] == 1]
    pf = df.loc[df["tf_first_layer"] == 0]

    ratios = []
    matched_tf_rows = []
    for ix_pf, row_pf in pf.iterrows():
        for ix_tf, row_tf in tf.iterrows():
            if not ix_tf in matched_tf_rows:
                if np.array_equal(row_pf["target_pattern"],row_tf["target_pattern"]):
                    ratios.append(np.log(row_pf["fun"] / row_tf["fun"]))
                    matched_tf_rows.append(ix_tf)
                    break
    return ratios


def ratio_patterning_noncognate_by_pair(df):
    nb = df.loc[df["minimize_noncognate_binding"] == 1]
    pt = df.loc[df["minimize_noncognate_binding"] == 0]

    ratios = []
    matched_nb_rows = []
    for ix_pt, row_pt in pt.iterrows():
        for ix_nb, row_nb in nb.iterrows():
            if not ix_nb in matched_nb_rows:
                if np.array_equal(row_pt["target_pattern"],row_nb["target_pattern"]):
                    # factor of 2 b/c max patterning error term is 1 per gene and noncognate is 2 per gene
                    ratios.append(np.log(2 * row_pt["fun"] / row_nb["fun"]))
                    matched_nb_rows.append(ix_nb)
                    break
    return ratios


def set_default_font_sizes(fontsize):

    SMALL_SIZE = round(fontsize*0.8)
    MEDIUM_SIZE = fontsize
    BIGGER_SIZE = round(fontsize*1.2) 

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_label_from_sublabels(key,sublabs,include_super=False,default_super=[]):
    if not isinstance(key,tuple):
        key = tuple([key])

    if not isinstance(sublabs,list):
        sublabs = [sublabs]

    # construct label
    lab = []
    for jj, minikey in enumerate(key):
        if (len(sublabs) > jj) and sublabs[jj]:
            lab.append(sublabs[jj][minikey])
        elif include_super:
            lab.append(f"{default_super[jj]} = {minikey}")
        else:
            lab.append(f"{minikey}")
    lab = ", ".join(lab)

    return lab


def subplots_groupby(df,supercol,filename,title,plotfn,*args,
                     subtitle_include_supercol = True,fontsize=24,
                     custom_subtitles = [],
                     figsize = [],subplot_dim = [],**kwargs):
    if type(supercol) == str:
        supercol = [supercol]
    gb = df.groupby(supercol,group_keys=True)
    
    if len(subplot_dim) == 0:
        sq = int(np.ceil(np.sqrt(gb.ngroups)))
        subplot_dim = (sq,sq)

        if len(figsize) == 0:
            figsize = (24,24,)
    elif len(figsize) == 0:
        figsize = (24*subplot_dim[1],24*subplot_dim[0],)

    set_default_font_sizes(fontsize)

    fig, ax = plt.subplots(subplot_dim[0],subplot_dim[1],figsize=figsize)
    ax = np.array(ax).flatten()
    for ii, key in enumerate(gb.groups.keys()):
        keytuple = key
        if not (keytuple is tuple):
            keytuple = tuple([keytuple])

        # construct subtitle
        subtitle = get_label_from_sublabels(key,custom_subtitles,subtitle_include_supercol,supercol)

        plotfn(gb.get_group(key),*args,ax=ax[ii],title=subtitle,**kwargs)
    fig.suptitle(title,wrap=True)

    plt.savefig(filename)
    plt.close()


def boxplot_groupby(df,cols,f,title="",filename="",ax=[],axlabel=[]):
    gb = df.groupby(cols,group_keys=True)
    gb_f = gb.apply(f)
    gb_f = [list(gb_f[key]) for key in gb.groups.keys()]
    
    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    bp = ax.boxplot(gb_f,patch_artist=True)

    if not axlabel:
        axlabel = f"{tuple(cols)}"
    ax.set_xlabel(axlabel,wrap=True)
    ax.set_xticklabels(gb.groups.keys(),rotation=45,ha="right")
    plt.subplots_adjust(bottom=0.15)

    def color_patches(f):
        bpcolors = gb.apply(f).to_list()
        for cur_bp in bp:
            for patch, color in zip(bp["boxes"],bpcolors):
                patch.set_facecolor(color)

    if "tf_first_layer" in cols:
        def by_tf(x):
            if any(x["tf_first_layer"].to_list()) == 1:
                return "lightgreen"
            else:
                return "lightblue"
        color_patches(by_tf)
    elif "MAX_CLUSTERS_ACTIVE" in cols:
        color_by_cluster_dict = {3:"pink",5:"lightgreen",8:"lightblue"}

        def by_cluster(x):
            for key, value in color_by_cluster_dict.items():
                if any(x["MAX_CLUSTERS_ACTIVE"] == key):
                       return value
        color_patches(by_cluster)


    if not title == "":
        ax.set_title(title)

    if not filename == "":
        plt.rcParams.update({'font.size':24})
        plt.savefig(filename)
        

def scatter_target_expression_groupby(df,cols,title="",filename="",ax=[],leglabel=[]):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        actual_expression = np.array(gr["output_expression"].to_list()).flatten()

        ax.plot([0,1],[0,1],color="gray",linewidth=1)
        if not leglabel:
            labtext = f"{cols} = {gr.name}"
        else:
            labtext = leglabel[gr.name]

        ax.plot(target_pattern_vals,actual_expression,'o',ms=5,alpha=0.2,label=labtext)

    gb.apply(scatter_one)

    ax.set_xlabel("target expression level")
    ax.set_ylabel("actual expression level")
    ax.set_ylim(0,1)
    lg = ax.legend(loc="upper left",markerscale=10)

    for lgh in lg.get_lines():
        lgh.set_alpha(1)
        lgh.set_marker('.')

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def scatter_patterning_residuals_groupby(df,cols,title="",filename="",ax=(),fontsize=24):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        actual_expression = np.array(gr["output_expression"].to_list()).flatten()

        ax.scatter(target_pattern_vals,actual_expression - target_pattern_vals,s=5,alpha=0.2,label=f"{cols} = {gr.name}")
        plt.rcParams.update({'font.size':fontsize})
        plt.rc("legend",fontsize=np.round(fontsize*0.75))

    gb.apply(scatter_one)

    ax.set_xlabel("target expression level")
    ax.set_ylabel("target expression level - actual expression level")
    ax.legend(loc="lower left")

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)

def scatter_error_fraction_groupby(df,cols,title="",filename="",ax=(),fontsize=24):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        error_frac = np.array(gr["output_error"].to_list())
        error_frac = [x[:,2] for x in error_frac]
        error_frac = np.array(error_frac).flatten()

        ax.scatter(target_pattern_vals,error_frac,s=5,alpha=0.2,label=f"{cols} = {gr.name}")
        plt.rcParams.update({'font.size':fontsize})
        plt.rc("legend",fontsize=np.round(fontsize*0.75))

    gb.apply(scatter_one)

    ax.set_xlabel("target expression level")
    ax.set_ylabel("total error fraction")
    ax.legend(loc="lower left")

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def scatter_modulating_concentrations(df,title="",filename="",ax=[]):
    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    tf_pr_bound = dill_load_as_dict(df,"tf_pr_bound.out")

    target_pattern_vals = np.array(df["target_pattern"].to_list()).flatten()
    optimized_input_vals = np.array(df[["optimized_input","N_PF"]].apply(lambda x: x["optimized_input"][x["N_PF"]:],axis=1).to_list()).flatten()
    modulating_concentration_vals = np.array(df["modulating_concentrations"].to_list()).flatten()

    # induction curve
    for cur_tf_pr_bound in tf_pr_bound.values():
        tf_sweep = np.linspace(0,2000,5000)
        layer2_induction_no_xtalk = np.array(list(map(cur_tf_pr_bound,tf_sweep,tf_sweep)))
        ax.plot(layer2_induction_no_xtalk,tf_sweep,color="black",linewidth=2)

    ax.scatter(target_pattern_vals,optimized_input_vals,color="blue",s=5,alpha=0.1,
               label="globally optimized concentration")
    ax.scatter(target_pattern_vals,modulating_concentration_vals,color="green",s=5,alpha=0.1,
               label="locally optimized concentration")
    ax.set_xlabel("target expression level")
    ax.set_ylabel("concentration")
    ax.set_ylim(0,min(200,max([max(optimized_input_vals),max(modulating_concentration_vals)])))
    
    if not title == "":
        ax.set_title(title,wrap=True)

    lg = ax.legend(loc="upper left",markerscale=10)

    for lgh in lg.legendHandles:
        lgh.set_alpha(1)

    if not filename == "":
        plt.rcParams.update({'font.size':24})
        plt.savefig(filename)


def dill_load_as_dict(df,filename):
    db_folders = df["filename"].apply(os.path.dirname)
    db_folders = db_folders.unique()
    return dict(zip(db_folders,list(map(lambda x: dill.load(open(os.path.join(x,filename),"rb")),db_folders))))

def tf_vs_kpr_error_rate(df,folder):
    tf_chrom_equiv_pr_bound = dill_load_as_dict(df,"tf_chrom_equiv_pr_bound.out")
    tf_chrom_equiv_error_rate = dill_load_as_dict(df,"tf_chrom_equiv_error_rate.out")
    kpr_pr_open = dill_load_as_dict(df,"kpr_pr_open.out")
    kpr_error_rate = dill_load_as_dict(df,"kpr_opening_error_rate.out")

    C_NS = np.mean(df["optimized_input"].apply(sum))
    c_S = np.logspace(-5,5,1000)
    plt.rcParams.update({'font.size':24})
    for key in kpr_pr_open.keys():
        fig, ax = plt.subplots(figsize=(24,24))
        ax.plot(tf_chrom_equiv_pr_bound[key](C_NS+c_S,c_S),np.log(tf_chrom_equiv_error_rate[key](C_NS+c_S,c_S)),label="TF")
        ax.plot(kpr_pr_open[key](C_NS+c_S,c_S),np.log(kpr_error_rate[key](C_NS+c_S,c_S)),label="chromatin")
        ax.legend()

        ax.set_xlabel("probability on/open")
        ax.set_ylabel("log error fraction")
        
        plt.savefig(os.path.join(folder,f"error_plot_{os.path.split(os.path.split(key)[0])[1]}.png"))
        plt.close(fig)
        break
    return



# idealized curve: given a concentration of noncogate factors,
# what specific layer 2 concentration would give exactly the
# target expression level? (fix layer 1 factors)
def calc_modulating_concentrations(df):
    def objective_fn(f,c_NS,c_S,target):
        return f(c_NS+c_S,c_S) - target

    tf_chrom_equiv_pr_bound = dill_load_as_dict(df,"tf_chrom_equiv_pr_bound.out")
    kpr_pr_open = dill_load_as_dict(df,"kpr_pr_open.out")
    tf_pr_bound = dill_load_as_dict(df,"tf_pr_bound.out")

    def calc_one_row(row):
        if np.isnan(row["modulating_concentrations"]).any():
            db_folder = os.path.dirname(row.filename)

            modulating_concentrations = np.zeros(len(row["target_pattern"]))
            layer1_concentrations = row["optimized_input"][:row["N_PF"]]
            tf_concentrations = row["optimized_input"][row["N_PF"]:]
            for ii_gene, target_level in enumerate(row["target_pattern"]):
                if row["tf_first_layer"]: # b/c crosslayer crosstalk permitted
                    cur_noncognate_concentrations = np.sum(row["optimized_input"]) - tf_concentrations[ii_gene]
                    noncognate_for_layer1 = cur_noncognate_concentrations
                    layer1_pr_on = tf_chrom_equiv_pr_bound[db_folder]
                else:
                    layer1_pr_on = kpr_pr_open[db_folder]
                    cur_noncognate_concentrations = np.sum(tf_concentrations) - tf_concentrations[ii_gene]
                    noncognate_for_layer1 = np.sum(layer1_concentrations)

                layer1_probabilities = np.array(row["R"]).dot(layer1_pr_on(noncognate_for_layer1,layer1_concentrations))
                target_corrected_for_layer1 = np.divide(row["target_pattern"],layer1_probabilities)

                modulating_concentrations[ii_gene] = scipy.optimize.fsolve(lambda x: objective_fn(tf_pr_bound[db_folder],cur_noncognate_concentrations,x,target_corrected_for_layer1[ii_gene]),tf_concentrations[ii_gene])
            row["modulating_concentrations"] = np.array(modulating_concentrations)
        return row

    return df.apply(calc_one_row,axis=1)
