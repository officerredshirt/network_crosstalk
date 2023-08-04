from numpy import *

import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import pandas
import numpy as np
import warnings

HIGHLY_EXPRESSING_THRESHOLD = 0.8

def main(argv):
    parser = argparse.ArgumentParser(
            prog = "df_plots",
            description = "",
            epilog = "")
    parser.add_argument("resfile")

    args = parser.parse_args()
    resfile = args.resfile

    if os.path.exists(resfile):
        df = pandas.read_hdf(resfile,"df")
    else:
        print(f"error: {resfile} does not exist")
        sys.exit()

    #warnings.filterwarnings("ignore",category=RuntimeWarning)

    prefixes = ['patterning','noncognate_binding']
    tf_prefix = ["chromatin","TF"]

    var_names_dict = {"ratio_KNS_KS":"$K_{NS}/K_S$",
                      "M_GENE":"number genes",
                      "minimize_noncognate_binding":"minimize noncognate binding"}

    #plot_db.tf_vs_kpr_error_rate(df,"../plots/fig/")

    maxclust = 8
    m_gene = 250

    df = df.loc[df["layer1_static"] == False]

    # FIGURE 1
    # main result: chromatin outperforms TF in terms of expression error
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig1_boxplot_chromatin_tf",
                             f"RMS global expression error",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.rms_xtalk,
                             axlabel=" ",
                             axticks={0:"chromatin",1:"TF only"},
                             supercol_labels = var_names_dict)

    # actual vs. target expression
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             f"../plots/fig/fig1_scatter_target_expression_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"target expression",
                             plot_db.scatter_target_expression_groupby,
                             ["ratio_KNS_KS"],fontsize=36,
                             subtitle_include_supercol = True,
                             custom_subtitles={0:"chromatin",1:"TF only"},
                             leglabel_vars=var_names_dict)

    # breakdown of contributions to global expression error by target expression level
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig1_scatter_patterning_residuals_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"global expression error",
                             plot_db.scatter_patterning_residuals_groupby,
                             ["tf_first_layer"],subplot_dim=(1,4),fontsize=36,
                             supercol_labels=var_names_dict)

    # aggregate error over ON vs. OFF genes
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig1_boxplot_aggregate_error_percentage",
                             f"percent global expression error from OFF genes",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.percent_expression_err_from_ON_vs_OFF_genes,
                             supercol_labels=var_names_dict,
                             axlabel=" ",
                             axticks={0:"chromatin",1:"TF only"})

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig1_boxplot_aggregate_error_off",
                             f"cumulative global expression error from OFF genes",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.cumulative_expression_err_from_OFF_genes,
                             supercol_labels=var_names_dict,
                             axlabel=" ",
                             axticks={0:"chromatin",1:"TF only"})

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig1_boxplot_aggregate_error_high",
                             f"cumulative global expression error from highly expressed genes (> {HIGHLY_EXPRESSING_THRESHOLD})",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             lambda x: plot_db.cumulative_expression_err_from_high_genes(x,HIGHLY_EXPRESSING_THRESHOLD),
                             supercol_labels=var_names_dict,
                             axlabel=" ",
                             axticks={0:"chromatin",1:"TF only"})


    # effective dynamic range
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig1_boxplot_effective_dynamic_range",
                             f"effective dynamic range (max(ON) - min(ON))",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.effective_dynamic_range,
                             supercol_labels=var_names_dict,
                             axlabel=" ",
                             axticks={0:"chromatin",1:"TF only"})

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             f"../plots/fig/fig1_boxplot_effective_dynamic_range_ratio",
                             f"ratio of effective dynamic range (max(ON) - min(ON))",
                             plot_db.boxplot_groupby,
                             ["ratio_KNS_KS"],
                             plot_db.ratio_effective_dynamic_range_by_pair,
                             supercol_labels=var_names_dict,
                             axlabel=" ")
    


    l2dict = {True:"layer2",False:""}
    for l2 in [True,False]:
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["minimize_noncognate_binding","ratio_KNS_KS"],
                                 f"../plots/fig/scatter_error{l2dict[l2]}_by_relativemodulating{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                 f"error by modulating",
                                 plot_db.scatter_error_increase_by_modulating_concentration_groupby,
                                 ["tf_first_layer"],subplot_dim=(3,3),fontsize=52,
                                 custom_subtitles = [{0:"patterning error",1:"noncognate binding error"},{}],
                                 supercol_labels=var_names_dict,
                                 leglabel={0:"chromatin",1:"TF only"},layer2=l2)



if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
