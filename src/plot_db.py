#!/usr/bin/env python3

import pprint
import params
import os, sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import dill
import itertools
import manage_db
import pandas as pd
from pandarallel import pandarallel
import xarray

pandarallel.initialize()

TICK_FONT_RATIO = 3/4
LEG_FONT_RATIO = 3/4

def get_varname_to_value_dict(df):
    varname_dict = {"ratio_KNS_KS":"intrinsic specificity",#"$K_{NS}/K_S$",
                    "K_NS":"$K_{NS}$",
                    "K_S":"$K_S$",
                    "M_GENE":"number of genes",
                    "MAX_CLUSTERS_ACTIVE":"number of active clusters"}
    
    varname_to_value = {}
    for var in varname_dict.keys():
        possible_values = set(df[var])
        key_val_pairs = list(zip(itertools.repeat(var),possible_values))
        labels_per_key = [f"{varname_dict[x[0]]} = {int(x[1])}" for x in key_val_pairs]
        varname_to_value = varname_to_value | dict(zip(key_val_pairs,labels_per_key))

    boolean_vars = {("minimize_noncognate_binding",0):"optimize expression levels",
                    ("minimize_noncognate_binding",1):"optimize binding profiles",
                    ("tf_first_layer",0):"chromatin",
                    ("tf_first_layer",1):"free DNA",
                    "tf_first_layer":"TF first layer",
                    ("target_independent_of_clusters",0):"aligned",#"OFF genes aligned\nwith clusters",
                    ("target_independent_of_clusters",1):"shuffled",#"OFF genes unaligned\nwith clusters",
                    ("ignore_off_during_optimization",0):"globally optimal",
                    ("ignore_off_during_optimization",1):"optimal for ON genes",
                    ("layer2_repressors",0):"activators only",
                    ("layer2_repressors",1):"with repressors"}

    varname_to_value = varname_to_value | boolean_vars | varname_dict

    return varname_to_value

def get_varname_to_color_dict():
    #return {"free DNA":np.array([174,169,223])/255,"chromatin":np.array([171,255,184])/255}
    return {"free DNA":np.array([230,0,73])/255,"chromatin":np.array([11,180,255])/255}

def to_tuple(x):
    if not type(x) == tuple:
        return tuple([x])
    return x


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
    df["error_metric_post_modulation"] = np.nan
    df["error_metric_post_modulation"] = df["error_metric_post_modulation"].astype(object)
    if "ratio_KNS_KS" not in df.columns:
        df["ratio_KNS_KS"] = df["K_NS"]/df["K_S"]
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
            new_df = convert_to_dataframe(db_filename)
            if not set(["ignore_off_during_optimization","target_independent_of_clusters"]).issubset(new_df.columns):
                new_df["ignore_off_during_optimization"] = int(0)
                new_df["target_independent_of_clusters"] = int(0)
            df = pd.concat([df,new_df])
    df.reset_index(drop=True,inplace=True)
    df["N_PF"] = df["N_PF"].astype(pd.Int64Dtype())
    df["N_TF"] = df["N_TF"].astype(pd.Int64Dtype())
    return df


# HOW DOES CHROMATIN ADVANTAGE OVER TF SCALE WITH GENOME SIZE?
# - boxplot patterning error/gene vs. "genome size" at different specificities

def row_calc_patterning_error(df):
    d = df["output_expression"] - df["target_pattern"]
    return d@d


def patterning_error(df):
    return df.apply(row_calc_patterning_error,axis=1)
    

def xtalk_by_gene(df):
    d = df["fun"].div(df.M_GENE,axis=0)
    return d.apply(np.log)


def rms_patterning_error(df):
    #d = df["fun"].div(df.M_GENE,axis=0)
    d = patterning_error(df)
    d = d.div(df["M_GENE"],axis=0)
    return d.apply(np.sqrt)

def curve_collapse(df,exponent):
    d = rms_patterning_error(df)
    fraction_on = df["MAX_CLUSTERS_ACTIVE"].div(df["N_CLUSTERS"],axis=0)
    denominator = np.power(fraction_on,exponent*np.ones(len(df)))
    d = d.div(denominator,axis=0)

    return d


def cumulative_expression_err_from_high_genes(df,thresh):
    def per_row(row):
        on_ix = row["target_pattern"] > thresh

        on_target = np.array(manage_db.logical_ix(row["target_pattern"],on_ix))

        on_expression = np.array(manage_db.logical_ix(row["output_expression"],on_ix))
        
        on_err = on_expression - on_target
        on_err = on_err@on_err
    
        return on_err
    return df.parallel_apply(per_row,axis=1)


def cumulative_expression_err_from_OFF_genes(df):
    def per_row(row):
        off_ix = row["target_pattern"] == 0

        off_target = np.array(manage_db.logical_ix(row["target_pattern"],off_ix))

        off_expression = np.array(manage_db.logical_ix(row["output_expression"],off_ix))
        
        off_err = off_expression - off_target
        off_err = off_err@off_err
    
        return off_err
    return df.parallel_apply(per_row,axis=1)


def percent_expression_err_from_ON_vs_OFF_genes(df):
    def per_row(row):
        on_ix = row["target_pattern"] > 0
        off_ix = row["target_pattern"] == 0

        on_target = np.array(manage_db.logical_ix(row["target_pattern"],on_ix))
        off_target = np.array(manage_db.logical_ix(row["target_pattern"],off_ix))

        on_expression = np.array(manage_db.logical_ix(row["output_expression"],on_ix))
        off_expression = np.array(manage_db.logical_ix(row["output_expression"],off_ix))

        on_err = on_expression - on_target
        on_err = on_err@on_err
        
        off_err = off_expression - off_target
        off_err = off_err@off_err
    
        return (off_err)/(on_err+off_err)
    return df.parallel_apply(per_row,axis=1)


def get_regulator_concentrations(df,layer):
    def per_row(row):
        if layer == 1:
            row["layer_conc"] = row["optimized_input"][0:row["N_PF"]]
        elif layer == 2:
            row["layer_conc"] = row["optimized_input"][row["N_PF"]:]
        else:
            print(f"unrecognized layer {layer}")
            sys.exit()
        return row

    return df.parallel_apply(per_row,axis=1)["layer_conc"]


def effective_dynamic_range_per_row(row):
        on_ix = row["target_pattern"] > 0
        on_expression = np.array(manage_db.logical_ix(row["output_expression"],on_ix))
        return max(on_expression) - min(on_expression)

def effective_dynamic_range(df):
    # TAKE 1: defined as max(ON expression) - min(ON expression)
    return df.parallel_apply(effective_dynamic_range_per_row,axis=1)


def ratio_effective_dynamic_range_by_pair(df):
    tf = df.loc[df["tf_first_layer"] == 1]
    pf = df.loc[df["tf_first_layer"] == 0]

    ratios = []
    matched_tf_rows = []
    for ix_pf, row_pf in pf.iterrows():
        for ix_tf, row_tf in tf.iterrows():
            if not ix_tf in matched_tf_rows:
                if np.array_equal(row_pf["target_pattern"],row_tf["target_pattern"]):
                    pf_dr = effective_dynamic_range_per_row(row_pf)
                    tf_dr = effective_dynamic_range_per_row(row_tf)
                    ratios.append(pf_dr/tf_dr)
                    matched_tf_rows.append(ix_tf)
                    break
    return ratios


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


def ratio_rms_xtalk_chromatin_tf_by_pair(df):
    tf = df.loc[df["tf_first_layer"] == 1]
    pf = df.loc[df["tf_first_layer"] == 0]

    ratios = []
    matched_tf_rows = []
    for ix_pf, row_pf in pf.iterrows():
        for ix_tf, row_tf in tf.iterrows():
            if not ix_tf in matched_tf_rows:
                if row_pf[["filename","target_pattern"]].equals(row_tf[["filename","target_pattern"]]):
                    #ratios.append(0.5*np.log(row_pf["fun"] / row_tf["fun"])) #0.5 b/c RMS
                    ratios.append(np.sqrt(row_pf["fun"] / row_tf["fun"]))
                    matched_tf_rows.append(ix_tf)
                    break
    return ratios


# Note this function uses a single error metric, the expression error,
# even if the optimization was performed over noncognate binding error.
def ratio_rms_error_patterning_noncognate_by_pair(df):
    nb = df.loc[df["minimize_noncognate_binding"] == 1]
    pt = df.loc[df["minimize_noncognate_binding"] == 0]

    ratios = []
    matched_nb_rows = []
    for ix_pt, row_pt in pt.iterrows():
        for ix_nb, row_nb in nb.iterrows():
            if not ix_nb in matched_nb_rows:
                if np.array_equal(row_pt["target_pattern"],row_nb["target_pattern"]):
                    # calculate expression error for noncognate binding
                    d = row_nb["output_expression"] - row_nb["target_pattern"]
                    d1 = row_pt["output_expression"] - row_pt["target_pattern"]
                    #ratios.append(0.5*np.log(row_pt["fun"] / d@d)) #0.5 b/c RMS
                    ratios.append(np.sqrt(row_pt["fun"] / (d@d)))
                    matched_nb_rows.append(ix_nb)
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


def get_label(varnames,vals,varnames_dict):
    if varnames_dict:
        keys = list(zip(varnames,vals))
        lab = []
        for key in keys:
            lab.append(varnames_dict[key])
        lab = ", ".join(lab)
    else:
        lab = f"{varnames} = {vals}"
    return lab


def subplots_groupby(df,supercol,filename,title,plotfn,*args,
                     fontsize = 24, ax = [], subtitles = [],
                     varnames_dict = [],figsize = [],subplot_dim = [],**kwargs):
    new_fig = True

    if type(supercol) == str:
        supercol = [supercol]
    gb = df.groupby(supercol,group_keys=True)
    
    if len(ax) > 0:
        new_fig = False
        if len(subplot_dim) > 0:
            print("ax is specified--ignoring subplot_dim...")
        if len(figsize) > 0:
            print("ax is specified--ignoring figsize...")
        if len(title) > 0:
            print("ax is specified--ignoring title...")
        if len(filename) > 0:
            print("ax is specified--ignoring filename...")
    
    if len(subplot_dim) == 0:
        sq = int(np.ceil(np.sqrt(gb.ngroups)))
        subplot_dim = (sq,sq)

        if len(figsize) == 0:
            figsize = (24,24,)
    elif len(figsize) == 0:
        figsize = (24*subplot_dim[1],24*subplot_dim[0],)

    set_default_font_sizes(fontsize)

    if new_fig:
        fig, ax = plt.subplots(subplot_dim[0],subplot_dim[1],figsize=figsize)

    ax = np.array(ax).flatten()
    for ii, key in enumerate(gb.groups.keys()):
        cur_group_label = get_label(supercol,to_tuple(key),varnames_dict)

        mastercolor_dict = get_varname_to_color_dict()
        if cur_group_label in mastercolor_dict.keys():
            mastercolor = mastercolor_dict[cur_group_label]
        else:
            mastercolor = 0.7*np.array([1,1,1])

        if not subtitles:
            subtitle = cur_group_label
        else:
            subtitle = subtitles[ii]

        plotfn(gb.get_group(key),*args,varnames_dict=varnames_dict,ax=ax[ii],title=subtitle,
               mastercolor=mastercolor,fontsize=fontsize,**kwargs)

    if new_fig:
        fig.suptitle(title,wrap=True)

        plt.savefig(filename)
        plt.close()


def symbolscatter_groupby(df,cols,f,title="",filename="",varnames_dict=[],
                          ax=[],ylabel="mean",fontsize=24,
                          legloc="upper right",axlabel=[],logxax=True,logyax=False,
                          suppress_leg=False,linewidth=3,markersize=15,take_ratio=False,
                          color="black",force_color=False,
                          xticks=None,yticks=None,legncol=1,**kwargs):
    if not ((len(cols) > 1) & (len(cols) < 4)):
        print("symbolscatter_groupby requires 2 or 3 cols")
        sys.exit()

    try:
        df["temp_barchart_fn"] = f(df)
        vals_test = (df.groupby(cols)["temp_barchart_fn"].mean()).to_frame()
        #vals_std = (df.groupby(cols)["temp_barchart_fn"].sem()).to_frame()
    except:
        gb = df.groupby(cols,group_keys=True)
        gb_f = gb.apply(f)
        vals_test = gb_f.apply(lambda x: np.mean(x)).to_frame()
        vals_test = vals_test.rename({0:"temp_barchart_fn"},axis="columns")
        #vals_std = gb_f.apply(lambda x: np.std(x)/sqrt(len(x)-1)).to_frame()
        #vals_std = vals_std.rename({0:"temp_barchart_fn"},axis="columns")

    vals_test = vals_test.reset_index()
    #vals_std = vals_std.reset_index()
    
    last_col_ix = len(cols)-1
    if take_ratio:
        def gr_take_ratio(gr):
            numerator = gr.loc[gr[cols[last_col_ix]] != 0]
            denominator = gr.loc[gr[cols[last_col_ix]] == 0]
            if (not len(numerator["temp_barchart_fn"]) == 1) | (not len(denominator["temp_barchart_fn"]) == 1):
                print("error: need 2 items to take ratio")
                sys.exit()
            return numerator["temp_barchart_fn"].values / denominator["temp_barchart_fn"].values

        new_gb = vals_test.groupby(cols[0:last_col_ix])
        vals_test = new_gb.apply(gr_take_ratio).reset_index(name="temp_barchart_fn")

    if len(vals_test.columns) == 2:
        ax.plot(vals_test[cols[0]].values,vals_test["temp_barchart_fn"].values,color=color,
                linewidth=linewidth,marker="h",markersize=markersize)
        #ax.plot(ratios[cols[0]].values,ratios["ratio"].values,color=color,
                #linewidth=linewidth,marker="h",markersize=markersize)
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ["o","D"]
        color_dict = get_varname_to_color_dict()


        for ii, prop in enumerate(set(vals_test[cols[1]])):
            label = get_label([cols[1]],to_tuple(prop),varnames_dict)
            if label in color_dict.keys():
                cur_color = color_dict[label]
            else:
                cur_color = colors[ii]
            if force_color:
                cur_color = color
            cur_vals = vals_test.loc[vals_test[cols[1]] == prop]
            #cur_err = vals_std.loc[vals_std[cols[1]] == prop]
            ax.plot(cur_vals[cols[0]],cur_vals["temp_barchart_fn"],label=label,color=cur_color,
                    linewidth=linewidth,marker=markers[ii],markersize=markersize)
            #ax.errorbar(cur_vals[cols[0]],cur_vals["temp_barchart_fn"],yerr=cur_err["temp_barchart_fn"],
                        #ecolor='k',elinewidth=3,label=label,color=color,
                        #linewidth=5,marker=markers[ii],markersize=20)

    ax.set_ylabel(ylabel,wrap=True,fontsize=fontsize)
    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))

    if logxax:
        ax.set_xscale('log')
        ax.set_xlim([1e2,1e4])

    if logyax:
        ax.set_yscale('log')
        ax.set_ylim([1,100])

    if xticks is not None:
        ax.set_xticks(xticks)

    if yticks is not None:
        ax.set_yticks(yticks)

    if not axlabel:
        axlabel = varnames_dict[cols[0]]
    ax.set_xlabel(axlabel,wrap=True,fontsize=fontsize)

    if not suppress_leg:
        lg = ax.legend(loc=legloc,ncol=legncol,frameon=False,fontsize=round(LEG_FONT_RATIO*fontsize))

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    if not title == "":
        ax.set_title(title)

    if not filename == "":
        plt.rcParams.update({'font.size':24})
        plt.savefig(filename)


def barchart_groupby(df,cols,f,title="",filename="",varnames_dict=[],ax=[],ylabel="mean",
                     legloc="upper right",axlabel=[],mastercolor=[],legncol=1,fontsize=24):
    if not len(cols) == 2:
        print("barchart_groupby requires len(cols) == 2")
        sys.exit()

    try:
        df["temp_barchart_fn"] = f(df)
        vals_test = (df.groupby(cols)["temp_barchart_fn"].mean()).to_frame()
    except:
        gb = df.groupby(cols,group_keys=True)
        gb_f = gb.apply(f)
        vals_test = gb_f.apply(lambda x: np.mean(x)).to_frame()
        vals_test = vals_test.rename({0:"temp_barchart_fn"},axis="columns")

    ncol0 = len(vals_test.index.unique(level=cols[0]))
    ncol1 = len(vals_test.index.unique(level=cols[1]))
    
    dd = {}
    for ix in vals_test.index:
        dd.setdefault(ix[1],[])
        dd[ix[1]] = dd[ix[1]] + [(vals_test.loc[ix]["temp_barchart_fn"])]

    axticklabs = []
    for col0_ix in vals_test.index.unique(level=cols[0]):
        axticklabs.append(get_label(cols,to_tuple(col0_ix),varnames_dict))

    barcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_dict = get_varname_to_color_dict()

    labelloc = np.arange(ncol0)
    width = 0.25
    multiplier = 0
    for ii, (att, vals) in enumerate(dd.items()):
        label = get_label([cols[1]],to_tuple(att),varnames_dict)
        if label in color_dict.keys():
            color = color_dict[label]
        else:
            color = barcolors[ii]
        offset = width*multiplier
        ax.bar(labelloc+offset,vals,width,label=label,color=color)
        multiplier += 1

    ax.set_ylabel(ylabel,wrap=True)

    try:
        ix = axticklabs[0].index("=")
        ticklab_prefixes = [x[0:ix] for x in axticklabs]
        
        if len(set(ticklab_prefixes)) == 1:
            axlabel = ticklab_prefixes[0][:-1]
            axticklabs = [x[ix+2:] for x in axticklabs]
    except:
        pass

    if not axlabel:
        axlabel = f"{tuple(cols)}"
    ax.set_xlabel(axlabel,wrap=True)

    ax.set_xticks(labelloc + width*((ncol1-1)/2),axticklabs)
    lg = ax.legend(loc=legloc,ncol=legncol,fontsize=round(LEG_FONT_RATIO*fontsize))

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    if not title == "":
        ax.set_title(title)

    if not filename == "":
        plt.rcParams.update({'font.size':24})
        plt.savefig(filename)


def get_color_from_label(label,mastercolor):
    color_dict = get_varname_to_color_dict()
    if label in color_dict.keys():
        color = color_dict[label]
    else:
        color = mastercolor
    return color


def colorscatter_2d_groupby(df,cols,f,title="",filename="",varnames_dict=[],ax=[],
                            mastercolor=[1,1,1],sizenorm_lims=[],size_lims=[500,2000],
                            ylabel=[],fontsize=24,draw_lines=False,markeralpha=0.6,
                            suppress_leg=False,linewidth=2,normalize=False,
                            logfit=False,**kwargs):
    gb = df.groupby(cols[0:2],group_keys=True)

    if not len(cols) == 4:
        print("ratio_colorscatter_2d_groupby requires len(cols) == 4")
        sys.exit()

    # WARNING: hacky solution for selecting colors
    ncol1 = len(set(df[cols[1]]))
    if ncol1 > 2:
        print("second column must store boolean values")
        sys.exit()

    #color_multiplier = np.linspace(1,0.2,ncol1)
    marker_dict = {0:"o",1:"X"}
    def scatter_one(gr,sizenorm_lims):
        try:
            gr["temp_barchart_fn"] = f(gr)
            vals_test = (gr.groupby(cols[2:])["temp_barchart_fn"].mean()).to_frame()
        except:
            gb = gr.groupby(cols[2:],group_keys=True)
            gb_f = gb.apply(f)
            vals_test = gb_f.apply(lambda x: np.mean(x)).to_frame()
            vals_test = vals_test.rename({0:"temp_barchart_fn"},axis="columns")

        vals_test = vals_test.reset_index()
        if normalize:
            vals_test["temp_barchart_fn"] = vals_test["temp_barchart_fn"] / \
                    np.min(vals_test["temp_barchart_fn"])
        sizes = np.array(vals_test[cols[2]])
        if len(sizenorm_lims) == 0:
            sizenorm_lims = [np.min(sizes),np.max(sizes)]

        if len(set(sizes)) > 1:
            vals_test = vals_test.reset_index()
            sizes = (sizes - sizenorm_lims[0])/(sizenorm_lims[1] - sizenorm_lims[0])
            sizes = sizes*(size_lims[1] - size_lims[0]) + size_lims[0]
        else:
            sizes = (size_lims[1] + size_lims[0])/2.0

        label = get_label([cols[0]],to_tuple(gr.name[0]),varnames_dict)
        color = get_color_from_label(label,mastercolor)
        #color = color*color_multiplier[gr.name[1]]
        ax.scatter(vals_test[cols[3]],vals_test["temp_barchart_fn"],s=sizes,color=color,
                   marker=marker_dict[gr.name[1]],clip_on=False,alpha=markeralpha,edgecolors="k",
                   label=get_label([cols[1]],to_tuple(gr.name[1]),varnames_dict))
        if draw_lines:
            line_gb = vals_test.groupby(cols[2])
            def plot_one(gr):
                ax.plot(gr[cols[3]],gr["temp_barchart_fn"],color=color,linewidth=linewidth,
                        marker="none")
            line_gb.apply(plot_one)
            #ax.plot(vals_test[cols[3]],vals_test["temp_barchart_fn"],color=color,linewidth=linewidth,
                    #marker="none",label=get_label([cols[1]],to_tuple(gr.name[1]),varnames_dict))
        if logfit:
            m, b = np.polyfit(np.log(vals_test[cols[3]]),np.log(vals_test["temp_barchart_fn"]),1)
            ax.plot(vals_test[cols[3]],np.exp(b)*np.power(vals_test[cols[3]],m),linewidth=5,color="k")
            print(f"m = {m}, exp(b) = {np.exp(b)}")

    gb.apply(lambda x: scatter_one(x,sizenorm_lims))

    ax.set_xlabel(varnames_dict[cols[3]],fontsize=fontsize)
    ax.set_xscale("log")
    ax.set_ylabel(ylabel,fontsize=fontsize)
    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))

    if not suppress_leg:
        f = lambda m,c: plt.plot([],[],marker=m,color=c,ls="none")[0]
        markerlabs = [get_label([cols[1]],to_tuple(ii),varnames_dict) for ii in set(df[cols[1]])]
        colorlabs = [get_label([cols[0]],to_tuple(ii),varnames_dict) for ii in set(df[cols[0]])]

        handles = [f(marker_dict[ii],"k") for ii in set(df[cols[1]])]
        handles += [mpl.lines.Line2D([0],[0],color=get_color_from_label(x,mastercolor),lw=linewidth)
                    for x in colorlabs]

        ax.legend(handles,markerlabs + colorlabs,loc="upper right",frameon=False,
                  markerscale=3.5,handlelength=1)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    if not title == "":
        ax.set_title(title,fontsize=fontsize)

    if not filename == "":
        plt.rcParams.update({'font.size':fontsize})


def colorplot_2d_groupby(df,cols,f,title="",filename="",varnames_dict=[],ax=[],
                         mastercolor=[1,1,1],colorbar_lims=[],**kwargs):
    if not len(cols) == 2:
        print("ratio_colorplot_2d_groupby requires len(cols) == 2")
        sys.exit()

    print("Calculating...")
    try:
        df["temp_barchart_fn"] = f(df)
        vals_test = (df.groupby(cols)["temp_barchart_fn"].mean()).to_frame()
    except:
        gb = df.groupby(cols,group_keys=True)
        gb_f = gb.apply(f)
        vals_test = gb_f.apply(lambda x: np.mean(x)).to_frame()
        vals_test = vals_test.rename({0:"temp_barchart_fn"},axis="columns")

    H = np.array(vals_test.to_xarray().to_array())[0]
    l0 = vals_test.index.unique(level=cols[0])
    l1 = vals_test.index.unique(level=cols[1])

    if len(colorbar_lims) == 0:
        colorbar_lims = [np.min(sizes),np.max(sizes)]

    print("Plotting...")
    custom_cm = mpl.colors.LinearSegmentedColormap.from_list("custom_cm",[[0,0,0],mastercolor])
    ax.imshow(H,interpolation="none",origin="lower",extent=[min(l1),max(l1),min(l0),max(l0)],
              aspect=(max(l1)-min(l1))/(max(l0)-min(l0)),cmap=custom_cm,vmin=colorbar_lims[0],
              vmax=colorbar_lims[1])

    for (j,i), label in np.ndenumerate(H):
        j = ((j+0.5)/len(l0))*(max(l0)-min(l0))+min(l0)
        i = ((i+0.5)/len(l1))*(max(l1)-min(l1))+min(l1)
        ax.text(i,j,f"{label:.3f}",ha="center",va="center",fontweight="bold",color="w")

    yticklocs = ((np.arange(0,H.shape[0])+0.5)/len(l0))*(max(l0)-min(l0))+min(l0)
    xticklocs = ((np.arange(0,H.shape[1])+0.5)/len(l1))*(max(l1)-min(l1))+min(l1)

    ax.set_xticks(xticklocs)
    ax.set_xticklabels(l1)

    ax.set_yticks(yticklocs)
    ax.set_yticklabels(l0)

    ax.set_xlabel(varnames_dict[cols[1]])
    ax.set_ylabel(varnames_dict[cols[0]])

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    if not title == "":
        ax.set_title(title)

    if not filename == "":
        plt.rcParams.update({'font.size':24})


def rms_scatter_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[],ylabel="",
                        legloc="upper right",axlabel=[],mastercolor=[],legncol=1,
                        bbox_to_anchor=None,**kwargs):
    if not len(cols) == 2:
        print("rms_scatter_groupby requires len(cols) == 2")
        sys.exit()

    if not ax:
        fig, ax = plt.subplots(figsize=(12,12))

    df["rms"] = rms_patterning_error(df)
    df["percent_off"] = percent_expression_err_from_ON_vs_OFF_genes(df)
    rms_vals = np.array(df["rms"].to_list())
    percent_off_vals = np.array(df["percent_off"].to_list())

    df["rms_on_vals"] = np.multiply(rms_vals,1-percent_off_vals)
    df["rms_off_vals"] = np.multiply(rms_vals,percent_off_vals)

    gb = df.groupby(cols[1])
    df["gbix"] = gb.ngroup()

    ncol0 = len(set(df[cols[0]]))
    ncol1 = len(set(df[cols[1]]))

    color_dict = get_varname_to_color_dict()

    width = 0.25
    multiplier = 0
    axticklabs = []
    labelloc = np.arange(ncol0)#+ncol0*width
    for ii in set(df["gbix"]):
        cur_df = df.loc[df["gbix"] == ii]
        color_label = get_label([cols[1]],to_tuple(cur_df.tail(1)[cols[1]].values[0]),varnames_dict)
        if color_label in color_dict.keys():
            color = color_dict[color_label]
        else:
            color = "black"

        subgrp = cur_df.groupby(cols[0])
        cur_df["sgbix"] = subgrp.ngroup()

        offset = width*multiplier
        def scatter_gr(gr):
            label = get_label([cols[0]],to_tuple(gr.name),varnames_dict)
            axticklabs.append(label)

            inner_multiplier = gr["sgbix"].unique()[0]

            rms_on_vals = np.array(gr["rms_on_vals"].to_list())
            rms_off_vals = np.array(gr["rms_off_vals"].to_list())

            ax.scatter(labelloc[inner_multiplier]+offset+np.random.uniform(low=-0.5*width,high=0.5*width,
                size=len(rms_on_vals)),rms_on_vals+rms_off_vals,marker="o",color=color*0.5)
            ax.scatter(labelloc[inner_multiplier]+offset+np.random.uniform(low=-0.5*width,high=0.5*width,
                size=len(rms_off_vals)),rms_off_vals,marker="x",color=color*0.1)

        subgrp.apply(scatter_gr)

        multiplier += 1

    #axticklabs = axticklabs[0:ncol0]
    #ax.set_xticks(labelloc + width*((ncol0-1)/2),axticklabs)

def rms_barchart_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[],ylabel="mean",
                     legloc="upper right",axlabel=[],mastercolor=[],legncol=1,suppress_leg=False,
                     bbox_to_anchor=None,fontsize=24,total_error=False,**kwargs):
    if not len(cols) == 2:
        print("barchart_groupby requires len(cols) == 2")
        sys.exit()

    if total_error:
        df["rms"] = patterning_error(df)
    else:
        df["rms"] = rms_patterning_error(df)
    vals_rms = (df.groupby(cols)["rms"].mean()).to_frame()
    df["percent_off"] = percent_expression_err_from_ON_vs_OFF_genes(df)
    vals_percent_off = (df.groupby(cols)["percent_off"].mean()).to_frame()

    ncol0 = len(vals_rms.index.unique(level=cols[0]))
    ncol1 = len(vals_rms.index.unique(level=cols[1]))

    dd = {}
    for ix in vals_rms.index:
        dd.setdefault(ix[1],([],[]))
        dd[ix[1]] = (dd[ix[1]][0] + [(vals_rms.loc[ix]["rms"])],
                     dd[ix[1]][1] + [(vals_rms.loc[ix]["rms"])*vals_percent_off.loc[ix]["percent_off"]])

    axticklabs = []
    for col0_ix in vals_rms.index.unique(level=cols[0]):
        axticklabs.append(get_label(cols,to_tuple(col0_ix),varnames_dict))

    barcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_dict = get_varname_to_color_dict()

    labelloc = np.arange(ncol0)
    width = 0.25
    multiplier = 0
    bottom = np.zeros(len(dd[ix[1]]))
    for ii, (att, vals) in enumerate(dd.items()):
        label = get_label([cols[1]],to_tuple(att),varnames_dict)
        if label in color_dict.keys():
            color = color_dict[label]
        else:
            color = barcolors[ii]
        offset = width*multiplier
        ax.bar(labelloc+offset,vals[0],width,label=label,color=color,edgecolor=color)
        #ax.bar(labelloc+offset,vals[1],width,color=(0,0,0),alpha=0.5,label="")
        ax.bar(labelloc+offset,vals[1],width,color="none",edgecolor="black",hatch="///")
        multiplier += 1

    ax.set_ylabel(ylabel,wrap=True,fontsize=fontsize)
    plt.setp(ax.get_yticklabels()[1::2],visible=False)

    try:
        ix = axticklabs[0].index("=")
        ticklab_prefixes = [x[0:ix] for x in axticklabs]
        
        if len(set(ticklab_prefixes)) == 1:
            if not axlabel:
                axlabel = ticklab_prefixes[0][:-1]
            axticklabs = [x[ix+2:] for x in axticklabs]
    except:
        pass

    if not axlabel:
        axlabel = f"{tuple(cols)}"
    ax.set_xlabel(axlabel,wrap=True)

    ax.set_xticks(labelloc + width*((ncol1-1)/2),axticklabs)
    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))
    if bbox_to_anchor is None:
        bbox_to_anchor = (0,0,1.0,1.0)
    if not suppress_leg:
        lg = ax.legend(loc=legloc,ncol=legncol,bbox_to_anchor=bbox_to_anchor,frameon=False,
                       fontsize=round(LEG_FONT_RATIO*fontsize))
    #lg = ax.legend(loc=legloc,ncol=legncol,frameon=False)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    if not title == "":
        ax.set_title(title,wrap=True,x=0.05,y=0.9,fontweight="bold",ha="left",fontsize=fontsize)

    if not filename == "":
        plt.rcParams.update({'font.size':24})
        plt.savefig(filename)


def boxplot_groupby(df,cols,f,title="",filename="",varnames_dict=[],ax=[],ylabel=" ",axlabel=[],
                    axticklabrotation=45):
    gb = df.groupby(cols,group_keys=True)
    gb_f = gb.apply(f)
    gb_f = [list(gb_f[key]) for key in gb.groups.keys()]
    
    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    bp = ax.boxplot(gb_f,patch_artist=True)

    axticklabs = [None]*len(gb.groups.keys())
    for ii, key in enumerate(gb.groups.keys()):
        axticklabs[ii] = get_label(cols,to_tuple(key),varnames_dict)

    try:
        ix = axticklabs[0].index("=")
        ticklab_prefixes = [x[0:ix] for x in axticklabs]
        
        if len(set(ticklab_prefixes)) == 1:
            if not axlabel:
                axlabel = ticklab_prefixes[0][:-1]
                axticklabs = [x[ix+2:] for x in axticklabs]
    except:
        pass

    ax.set_xticklabels(axticklabs,rotation=axticklabrotation,ha="center")

    if not axlabel:
        axlabel = f"{tuple(cols)}"
    ax.set_xlabel(axlabel,wrap=True)
    ax.set_ylabel(ylabel,wrap=True)
    
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
        

def bar_error_fraction_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[]):
    if len(cols) == 1:
        gb_cols = cols[0]
    else:
        gb_cols = cols
    gb = df.groupby(gb_cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    layer1_error_frac = np.zeros(len(gb.groups.keys()))
    layer2_error_frac = np.zeros(len(gb.groups.keys()))
    total_error_frac = np.zeros(len(gb.groups.keys()))
    labtext = []
    for ii, (name,gr) in enumerate(gb):
        cur_on_ix = np.array(gr["target_pattern"].to_list()).flatten() > 0
        cur_error_frac = gr["output_error"].to_list()
        cur_layer1_error_frac = np.array([x[:,0] for x in cur_error_frac]).flatten()
        cur_layer2_error_frac = np.array([x[:,1] for x in cur_error_frac]).flatten()
        cur_total_error_frac = np.array([x[:,2] for x in cur_error_frac]).flatten()
        layer1_error_frac[ii] = np.mean(manage_db.logical_ix(cur_layer1_error_frac,cur_on_ix))
        layer2_error_frac[ii] = np.mean(manage_db.logical_ix(cur_layer2_error_frac,cur_on_ix))
        total_error_frac[ii] = np.mean(manage_db.logical_ix(cur_total_error_frac,cur_on_ix))

        labtext.append(get_label(cols,to_tuple(name),varnames_dict))

    error_frac = {"Layer 1": layer1_error_frac, "Layer 2": layer2_error_frac, "total": total_error_frac}

    labelloc = np.arange(len(labtext))
    width = 0.25
    multiplier = 0
    for att, vals in error_frac.items():
        offset = width*multiplier
        rects = ax.bar(labelloc+offset,vals,width,label=att)
        multiplier += 1

    ax.set_ylabel("mean error fraction across ON genes")
    ax.set_xticks(labelloc+width,labtext)
    lg = ax.legend(loc="upper left")

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def regulator_concentration_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[],
                                    **kwargs):
    gb = df.groupby(cols,group_keys=True)

    #nbin = 1000

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    def plot_one(gr):
        pf_concentrations = get_regulator_concentrations(gr,1)
        tf_concentrations = get_regulator_concentrations(gr,2)

        pf_sum = pf_concentrations.parallel_apply(np.sum,axis=0)
        tf_sum = tf_concentrations.parallel_apply(np.sum,axis=0)
        total_by_pattern = pf_sum + tf_sum

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        #cnt, edges = np.histogram(total_by_pattern,density=False,bins=len(total_by_pattern))
        #ax.step(edges[:-1],cnt.cumsum()/sum(cnt),label=f"{labtext} total",drawstyle="steps")

        rms = rms_patterning_error(gr)
        ax.scatter(rms,total_by_pattern,label=labtext)

    gb.apply(plot_one)

    ax.set_xlabel("global expression error")
    ax.set_ylabel("total regulator concentration")
    #lg = ax.legend(loc="lower right")
    lg = ax.legend(loc="upper right")

    for lgh in lg.get_lines():
        lgh.set_alpha(1)
        lgh.set_marker('.')

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def scatter_error_fraction_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[],
                                   fontsize=24,mastercolor=[1,1,1],**kwargs):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    ncol1 = len(set(df[cols[0]]))
    cur_color_list = np.linspace(0.2,1,ncol1).reshape(ncol1,1) * np.multiply(mastercolor,np.ones((1,3)))
    colordict = {}
    for ii, lab in enumerate(gb.groups.keys()):
        colordict[lab] = cur_color_list[ii]

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        cur_on_ix = target_pattern_vals > 0
        cur_total_error_frac = gr["output_error"].to_list()
        cur_total_error_frac = np.array([x[:,2] for x in cur_total_error_frac]).flatten()
        cur_total_error_frac = manage_db.logical_ix(cur_total_error_frac,cur_on_ix)

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        bins = np.linspace(0,1,100)
        ax.ecdf(cur_total_error_frac,label=labtext,color=colordict[gr.name],linewidth=5)
        #ax.hist(cur_total_error_frac,bins=bins,alpha=0.5,label=labtext,color=colordict[gr.name])
        #ax.plot(target_pattern_vals,cur_total_error_frac,'o',ms=5,alpha=0.2,label=labtext,
                #color=colordict[gr.name])
        #target_pattern_vals_on = manage_db.logical_ix(target_pattern_vals,target_pattern_vals > 0)
        #ax.set_xlim(np.min(target_pattern_vals_on),np.max(target_pattern_vals_on))

    gb.apply(scatter_one)

    #ax.set_xlabel("target expression level")
    #ax.set_ylabel("total nontarget contribution")
    ax.set_xlabel("total nontarget contribution")
    ax.set_xlim(0,1)
    #ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0.5,1])
    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))

    cur_cmap = mpl.colors.ListedColormap(cur_color_list)
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cur_cmap),ax=ax,location='top')
    cb.ax.get_xaxis().set_ticks([])
    for j, lab in enumerate(gb.groups.keys()):
        cur_label = f"{lab:.0f}"
        cb.ax.text((2*j+1)/10.0,0.45,cur_label,ha='center',va='center',color='white',fontweight='bold')
    cb.ax.get_xaxis().labelpad = 15
    try:
        cb.ax.set_xlabel(varnames_dict[cols[0]],fontsize=fontsize)
    except Exception as e:
        print(e)
        pass

    if not title == "":
        ax.set_title(title,wrap=True,x=0.95,y=0.9,fontweight="bold",ha="right",fontsize=fontsize)
    if not filename == "":
        plt.savefig(filename)


def scatter_expression_factor_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[]):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    tf_pr_bound = dill_load_as_dict(df,"tf_pr_bound.out")
    def expression_from_layer(row):
        filename = os.path.dirname(row["filename"])
        layer2_opt_conc = row["optimized_input"][row["N_PF"]:]
        if row["crosslayer_crosstalk"]:
            C = np.sum(row["optimized_input"])
        else:
            C = np.sum(layer2_opt_conc)
        row["from_layer2"] = np.array(list(map(lambda x: tf_pr_bound[filename](C,x),layer2_opt_conc)))
        row["from_layer1"] = np.divide(row["output_expression"],row["from_layer2"])
        return row

    def scatter_one(gr):
        #target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        #from_layer1 = gr[["optimized_input","N_PF","crosslayer_crosstalk", \
        #        "output_expression","filename"]].parallel_apply(lambda x: expression_from_layer(x)[0],axis=1)
        #from_layer1 = np.array(from_layer1.to_list()).flatten()
        from_layer = gr[["optimized_input","N_PF","crosslayer_crosstalk", \
                "output_expression","filename"]].parallel_apply(expression_from_layer,axis=1)

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        #ax.plot(target_pattern_vals,from_layer1,'o',ms=5,alpha=0.2,label=labtext)
        from_layer1 = np.array(from_layer["from_layer1"].to_list()).flatten()
        from_layer2 = np.array(from_layer["from_layer2"].to_list()).flatten()
        ax.plot(from_layer1,from_layer2,'o',ms=5,alpha=0.2,label=labtext)

    gb.apply(scatter_one)

    #ax.set_xlabel("target expression level")
    ax.set_xlabel("probability Layer 1 open/bound")
    ax.set_ylabel("probability Layer 2 bound")
    ax.set_xlim([0.5,1])
    ax.set_ylim([0,1])
    lg = ax.legend(loc="upper left",markerscale=10)

    for lgh in lg.get_lines():
        lgh.set_alpha(1)
        lgh.set_marker('.')

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def scatter_target_expression_groupby(df,cols,title="",filename="",varnames_dict=[],ax=[],mastercolor=[],
                                      fontsize=24,colorbar_leg=True,gray_first_level=False,**kwargs):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    ncol1 = len(set(df[cols[0]]))
    cur_color_list = np.linspace(0.2,1,ncol1).reshape(ncol1,1) * np.multiply(mastercolor,np.ones((1,3)))
    if gray_first_level:
        cur_color_list[0,:] = 0.8*np.array([1,1,1])
    colordict = {}
    for ii, lab in enumerate(gb.groups.keys()):
        colordict[lab] = cur_color_list[ii]

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        actual_expression = np.array(gr["output_expression"].to_list()).flatten()

        ax.plot([0,1],[0,1],color="gray",linewidth=1)
        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        ax.plot(target_pattern_vals,actual_expression,'o',ms=5,alpha=0.2,label=labtext,
                color=colordict[gr.name])
        ax.plot(0,np.mean(manage_db.logical_ix(actual_expression,target_pattern_vals == 0)),
                   color=colordict[gr.name],marker='X',markersize=20,clip_on=False,zorder=10)

    gb.apply(scatter_one)

    ax.set_xlabel("target expression level",fontsize=fontsize)
    ax.set_ylabel("actual expression level",fontsize=fontsize)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0.5,1])
    ax.set_box_aspect(1)

    cur_cmap = mpl.colors.ListedColormap(cur_color_list)
    if colorbar_leg:
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cur_cmap),ax=ax,location='top')
        cb.ax.get_xaxis().set_ticks([])
    for j, lab in enumerate(gb.groups.keys()):
        cur_label = f"{lab:.0f}"
        if colorbar_leg:
            cb.ax.text((2*j+1)/10.0,0.45,cur_label,ha='center',va='center',color='white',fontweight='bold')

    try:
        cb.ax.get_xaxis().labelpad = 15
        cb.ax.set_xlabel(varnames_dict[cols[0]],fontsize=fontsize)
    except Exception as e:
        print(e)
        pass

    if not colorbar_leg:
        #lg = ax.legend(loc="upper center",markerscale=5,frameon=False,
                       #bbox_to_anchor=(0.5,1.17))
        lg = ax.legend(loc="lower right",markerscale=5)
        for lgh in lg.get_lines():
            lgh.set_alpha(1)
            lgh.set_marker('.')
        #cb.remove()

    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))

    if not title == "":
        ax.set_title(title,wrap=True,x=0.05,y=0.9,fontweight='bold',ha="left",fontsize=fontsize)
    if not filename == "":
        plt.savefig(filename)


def scatter_repressor_activator(df,cols,title="",filename="",ax=(),fontsize=24,varnames_dict=[],**kwargs):
    df = df.loc[df["layer2_repressors"] == True]
    if len(df) == 0:
        print("error: no entries found with layer2_repressors = True")
        sys.exit()

    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    def get_concentrations_per_row(row):
        cur_conc = np.array(row["optimized_input"])
        return cur_conc[row["N_PF"]:]

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        concentration_vals = np.array(gr[["N_PF","optimized_input"]].parallel_apply(get_concentrations_per_row,axis=1).to_list())
        
        activator_vals = np.array(list(map(lambda x: x[0:int(len(x)/2)],concentration_vals))).flatten()
        repressor_vals = np.array(list(map(lambda x: x[int(len(x)/2):],concentration_vals))).flatten()

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        #ax.scatter(target_pattern_vals,activator_vals/(activator_vals+repressor_vals),s=10,label=labtext)
        ax.scatter(target_pattern_vals,activator_vals,s=20,color=np.array([230,170,70])/255,label="activator")
        ax.scatter(target_pattern_vals,repressor_vals,s=20,color=np.array([10,200,100])/255,label="repressor")
        plt.rcParams.update({'font.size':fontsize})
        plt.rc("legend",fontsize=fontsize)

    gb.apply(scatter_one)

    ax.set_xlim([0,1])
    ax.set_xlabel("target expression level",fontsize=fontsize)
    ax.set_ylabel("target regulator concentration",fontsize=fontsize)
    ax.set_xticks([0,0.5,1])
    ax.set_box_aspect(1)

    lg = ax.legend(loc="upper left",markerscale=5)
    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)

def scatter_patterning_residuals_groupby(df,cols,title="",filename="",ax=(),fontsize=24,varnames_dict=[]):
    gb = df.groupby(cols,group_keys=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        actual_expression = np.array(gr["output_expression"].to_list()).flatten()

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        ax.scatter(target_pattern_vals,actual_expression - target_pattern_vals,s=5,alpha=0.2,label=labtext)
        plt.rcParams.update({'font.size':fontsize})
        plt.rc("legend",fontsize=fontsize)

    gb.apply(scatter_one)

    ax.set_xlabel("target expression level")
    ax.set_ylabel("target expression level - actual expression level")
    ax.legend(loc="lower left")

    if not title == "":
        ax.set_title(title,wrap=True)
    if not filename == "":
        plt.savefig(filename)


def scatter_error_increase_by_modulating_concentration_groupby(df,cols,title="",filename="",ax=(),varnames_dict=[],
                                                               fontsize=24,layer2=False,mastercolor=[]):
    gb = df.groupby(cols,group_keys = True)

    tf_error_rate = dill_load_as_dict(df,"tf_error_rate.out")

    def second_layer_error(row):
        filename = os.path.dirname(row["filename"])
        layer2_opt_conc = row["optimized_input"][row["N_PF"]:]
        C = np.sum(layer2_opt_conc)
        cum_err = np.zeros(len(row["modulating_concentrations"]))
        opt_cum_err = np.sum(list(map(lambda x: tf_error_rate[filename](C,x),layer2_opt_conc)))
        for ii, modconc in enumerate(row["modulating_concentrations"]):
            cur_conc = layer2_opt_conc.copy()
            cur_conc[ii] = modconc
            cum_err[ii] = np.sum(list(map(lambda x: tf_error_rate[filename](C - layer2_opt_conc[ii] + \
                    modconc, x), cur_conc)))
        return cum_err-opt_cum_err

    if not ax:
        fig, ax = plt.subplots(figsize=(12*len(gb),24))

    def scatter_one(gr):
        target_pattern_vals = np.array(gr["target_pattern"].to_list()).flatten()
        modulating_concentration_vals = np.array(gr["modulating_concentrations"].to_list()).flatten()
        modulating_concentration_vals[target_pattern_vals == 0] = None
        optimized_input_vals = np.array(gr[["optimized_input","N_PF"]].apply(lambda x: x["optimized_input"][x["N_PF"]:],axis=1).to_list()).flatten()
        concentration_change = np.divide(modulating_concentration_vals,optimized_input_vals)
        if not layer2:
            error_increase = np.array(gr[["fun","error_metric_post_modulation"]].parallel_apply( \
                    lambda x: x["error_metric_post_modulation"]-x["fun"],axis=1).to_list()).flatten()
        else:
            error_increase = gr[["filename","optimized_input", \
                    "N_PF","modulating_concentrations"]].parallel_apply(second_layer_error,axis=1)
            error_increase = np.array(error_increase.to_list()).flatten()

        labtext = get_label(cols,to_tuple(gr.name),varnames_dict)

        ax.plot(concentration_change,error_increase,'o',ms=5,alpha=0.2,label=labtext)
        ax.set_ylim(0,0.015)

    gb.apply(scatter_one)

    ax.set_xlabel("relative modulating concentration")
    ax.set_xlim(1,min(2,ax.get_xlim()[1]))

    if not layer2:
        ax.set_ylabel("change in total error")
    else:
        ax.set_ylabel("change in layer 2 cumulative error rate")

    lg = ax.legend(loc="upper left",markerscale=5)

    for lgh in lg.get_lines():
        lgh.set_alpha(1)
        lgh.set_marker('.')
    
    if not title == "":
        ax.set_title(title,wrap=True)

    if not filename == "":
        plt.rcParams.update({'font.size':fontsize})
        plt.savefig(filename)


def scatter_modulating_concentrations(df,title="",filename="",ax=[],varnames_dict=[],
                                      fontsize=24,mastercolor=[],**kwargs):
    if not ax:
        fig, ax = plt.subplots(figsize=(24,24))

    tf_pr_bound = dill_load_as_dict(df,"tf_pr_bound.out")

    target_pattern_vals = np.array(df["target_pattern"].to_list()).flatten()
    optimized_input_vals = np.array(df[["optimized_input","N_PF"]].apply(lambda x: x["optimized_input"][x["N_PF"]:],axis=1).to_list()).flatten()
    modulating_concentration_vals = np.array(df["modulating_concentrations"].to_list()).flatten()
    modulating_concentration_vals[target_pattern_vals == 0] = None

    # induction curve
    for cur_tf_pr_bound in tf_pr_bound.values():
        tf_sweep = np.logspace(-1,3,500)#np.linspace(0,2000,5000)
        layer2_induction_no_xtalk = np.array(list(map(cur_tf_pr_bound,tf_sweep,tf_sweep)))
        ax.plot(tf_sweep,layer2_induction_no_xtalk,color="black",linewidth=2,label="induction curve")

    ax.scatter(optimized_input_vals,target_pattern_vals,color=mastercolor,s=10,alpha=0.5,
               label="globally optimized")
    ax.scatter(modulating_concentration_vals,target_pattern_vals,color=0.5*mastercolor,s=10,alpha=0.5,
               label="optimized target TF")

    ax.set_xlabel("target TF concentration",fontsize=fontsize)
    ax.set_ylabel("target expression level",fontsize=fontsize)
    #ax.set_xlim(0,min(100,max([max(optimized_input_vals),max(modulating_concentration_vals)])))
    ax.set_xscale('log')
    ax.set_xlim([1e-1,1e3])
    #ax.set_ylim([0.5,1])

    ax.tick_params(axis="both",labelsize=round(TICK_FONT_RATIO*fontsize))
    
    if not title == "":
        ax.set_title(title,wrap=True,x=0.05,y=0.90,fontweight='bold',ha="left",fontsize=fontsize)

    lg = ax.legend(markerscale=5,handlelength=1,fontsize=round(LEG_FONT_RATIO*fontsize),#frameon=False,
                   loc="lower right")

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

    print("Calculating modulating concentrations...")
    def calc_one_row(row):
        if not row["layer2_repressors"]:
            if np.isnan(row["modulating_concentrations"]).any():
                db_folder = os.path.dirname(row.filename)
                print(".",end="",flush=True)

                crosstalk_metric = manage_db.get_crosstalk_metric_from_row(row)

                modulating_concentrations = np.zeros(len(row["target_pattern"]))
                error_metric_post_modulation = np.zeros(len(row["target_pattern"]))
                layer1_concentrations = row["optimized_input"][:int(row["N_PF"])]
                tf_concentrations = row["optimized_input"][int(row["N_PF"]):]
                for ii_gene, target_level in enumerate(row["target_pattern"]):
                    if True:#target_level > 0:
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

                        tf_concentrations_with_ii_modulated = tf_concentrations.copy()
                        tf_concentrations_with_ii_modulated[ii_gene] = modulating_concentrations[ii_gene]
                        error_metric_post_modulation[ii_gene] = crosstalk_metric(row["target_pattern"], \
                                layer1_concentrations,tf_concentrations_with_ii_modulated)

                row["modulating_concentrations"] = np.array(modulating_concentrations)
                row["error_metric_post_modulation"] = np.array(error_metric_post_modulation)
        return row

    return df.parallel_apply(calc_one_row,axis=1)
