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
import matplotlib.pyplot as plt

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


    #plot_db.tf_vs_kpr_error_rate(df,"../plots/fig/")

    maxclust = 8
    m_gene = 250

    df = df.loc[(df["layer1_static"] == False) & (df["ratio_KNS_KS"] > 100) &
                (df["MIN_EXPRESSION"] < 0.3)]

    varnames_dict = plot_db.get_varname_to_value_dict(df)

    # FIGURE 2
    """
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             ["M_GENE"],
                             f"../plots/fig/fig2_bar_chromatin_tf_RMS_error",
                             f"RMS global expression error",
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.rms_xtalk,
                             varnames_dict=varnames_dict)

    # main result: chromatin outperforms TF in terms of expression error
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig2_boxplot_chromatin_tf_RMS_error",
                             f"RMS global expression error",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.rms_xtalk,
                             varnames_dict=varnames_dict)

    # actual vs. target expression
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             f"../plots/fig/fig2_scatter_target_expression_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"target expression",
                             plot_db.scatter_target_expression_groupby,
                             ["ratio_KNS_KS"],subplot_dim=(1,2),fontsize=36,
                             varnames_dict=varnames_dict)


    # breakdown of contributions to global expression error by target expression level
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig2_scatter_patterning_residuals_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"global expression error",
                             plot_db.scatter_patterning_residuals_groupby,
                             ["tf_first_layer"],subplot_dim=(1,4),fontsize=48,
                             varnames_dict=varnames_dict)

    # aggregate error over ON vs. OFF genes
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig2_boxplot_aggregate_error_percentage",
                             f"percent global expression error from OFF genes",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.percent_expression_err_from_ON_vs_OFF_genes,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig2_boxplot_aggregate_error_off",
                             f"cumulative global expression error from OFF genes",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.cumulative_expression_err_from_OFF_genes,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig2_boxplot_aggregate_error_high",
                             f"cumulative global expression error from highly expressed genes (> {HIGHLY_EXPRESSING_THRESHOLD})",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             lambda x: plot_db.cumulative_expression_err_from_high_genes(x,HIGHLY_EXPRESSING_THRESHOLD),
                             varnames_dict=varnames_dict)


    # effective dynamic range
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig2_boxplot_effective_dynamic_range",
                             f"effective dynamic range (max(ON) - min(ON))",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.effective_dynamic_range,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             f"../plots/fig/fig2_boxplot_effective_dynamic_range_ratio",
                             f"ratio of effective dynamic range (max(ON) - min(ON))",
                             plot_db.boxplot_groupby,
                             ["ratio_KNS_KS"],
                             plot_db.ratio_effective_dynamic_range_by_pair,
                             axlabel=[],
                             varnames_dict=varnames_dict)
    
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             f"../plots/fig/fig2_boxplot_rms_error_ratio",
                             f"ratio of RMS global expression error",
                             plot_db.boxplot_groupby,
                             ["ratio_KNS_KS"],
                             plot_db.ratio_rms_xtalk_chromatin_tf_by_pair,
                             axlabel=[],
                             varnames_dict=varnames_dict)
    
    l2dict = {True:"layer2",False:""}
    for l2 in [True,False]:
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["minimize_noncognate_binding","ratio_KNS_KS"],
                                 f"../plots/fig/scatter_error{l2dict[l2]}_by_relativemodulating{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                 f"error by modulating",
                                 plot_db.scatter_error_increase_by_modulating_concentration_groupby,
                                 ["tf_first_layer"],subplot_dim=(2,3),fontsize=52,
                                 varnames_dict=varnames_dict,layer2=l2)
    

    # FIGURE 3

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             f"../plots/fig/fig3_scatter_error_fraction_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"total error fraction",
                             plot_db.scatter_error_fraction_groupby,
                             ["ratio_KNS_KS"],subplot_dim=(1,2),fontsize=36,
                             varnames_dict=varnames_dict)
    

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig3_bar_error_fraction_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             "",
                             plot_db.bar_error_fraction_groupby,
                             ["tf_first_layer"],subplot_dim=(1,3),fontsize=36,
                             varnames_dict=varnames_dict)
    
    
    tf_dict = {0:"chromatin",1:"free_DNA"}
    for tf in [0,1]:
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["tf_first_layer"] == tf)],
                                 ["minimize_noncognate_binding","ratio_KNS_KS"],
                                 f"../plots/fig/scatter_modulating_{tf_dict[tf]}_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                 f"induction curves vs. modulating concentrations ({tf_dict[tf].replace('_',' ')})",
                                 plot_db.scatter_modulating_concentrations,
                                 subplot_dim=(2,3),fontsize=52,
                                 varnames_dict=varnames_dict)
    
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig3_scatter_expression_layer_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"expression by layer",
                             plot_db.scatter_expression_factor_groupby,
                             ["tf_first_layer"],subplot_dim=(1,3),fontsize=36,
                             varnames_dict=varnames_dict)
    
    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["ratio_KNS_KS"],
                             f"../plots/fig/fig3_regulator_concentrations_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"",
                             plot_db.regulator_concentration_groupby,
                             ["tf_first_layer"],subplot_dim=(1,3),fontsize=36,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 1000.0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             f"../plots/fig/fig3_scatter_target_expression_vs_noncognate_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"target expression",
                             plot_db.scatter_target_expression_groupby,
                             ["minimize_noncognate_binding"],subplot_dim=(1,2),fontsize=36,
                             varnames_dict=varnames_dict)
    
    plot_db.subplots_groupby(df.loc[(df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "ratio_KNS_KS",
                             f"../plots/fig/fig3_boxplot_expression_error_ratio_ncb",
                             f"fold-change in RMS global expression error when co-opt nontarget binding",
                             plot_db.boxplot_groupby,
                             ["tf_first_layer"],
                             plot_db.ratio_rms_error_patterning_noncognate_by_pair,
                             subplot_dim=(1,3),fontsize=36,
                             varnames_dict=varnames_dict)
    """

    # --- COMPILED FIGURES --- #
    fntsz = 30
    plt.rcParams["font.size"] = f"{fntsz}"

    # FIGURE 2
    fig, ax =  plt.subplots(3,3,figsize=(36,36))

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             ["M_GENE"],
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.rms_xtalk,ax=[ax[0][0]],
                             subtitles=["",""],
                             fontsize=fntsz,ylabel="RMS global expression error",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             ["M_GENE"],
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.cumulative_expression_err_from_OFF_genes,ax=[ax[0][1]],
                             subtitles=["",""],
                             fontsize=fntsz,ylabel="cumulative expression error from OFF genes",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.percent_expression_err_from_ON_vs_OFF_genes,
                             subtitles=[""],fontsize=fntsz,ax=[ax[0][2]],
                             ylabel="percent expression error from OFF genes",
                             legloc="upper left",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             [],[],
                             plot_db.scatter_target_expression_groupby,
                             ["ratio_KNS_KS"],
                             fontsize=fntsz,ax=ax[1][0:2],
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.effective_dynamic_range,
                             ax=[ax[1][2]],legloc="upper left",#legloc="lower right",
                             subtitles=[""],fontsize=fntsz,
                             ylabel="effective dynamic range",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["minimize_noncognate_binding"] == 0) &
                                    (df["ratio_KNS_KS"] == 1000.0)],
                             ["tf_first_layer","ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_modulating_concentrations,
                             ax=ax[2][0:2],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["minimize_noncognate_binding"] == 0) &
                                    (df["ratio_KNS_KS"] == 1000.0)],
                             ["ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_error_increase_by_modulating_concentration_groupby,
                             ["tf_first_layer"],
                             ax=[ax[2][2]],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    fig.tight_layout()
    plt.savefig("../plots/fig/fig2.png")
    plt.close()
    

    # FIGURE 3
    fig, ax =  plt.subplots(2,3,figsize=(36,24))

    plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 1000.0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer","ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_target_expression_groupby,
                             ["minimize_noncognate_binding"],
                             ax=ax[0][0:2],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene)],
                             "M_GENE",
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.ratio_rms_error_patterning_noncognate_by_pair,
                             subtitles=["fold-change in RMS expression error \nwhen co-opt nontarget binding"],
                             ax=[ax[0][2]],fontsize=fntsz,
                             ylabel="fold-change",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             [],[],
                             plot_db.scatter_error_fraction_groupby,
                             ["ratio_KNS_KS"],
                             ax=ax[1][0:2],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    fig.tight_layout()
    plt.savefig("../plots/fig/fig3.png")
    plt.close()
    

    

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
