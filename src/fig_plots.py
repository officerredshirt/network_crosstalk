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

    df["N_PF"] = df["N_PF"].astype(pandas.Int64Dtype())
    df["N_TF"] = df["N_PF"].astype(pandas.Int64Dtype())
    df = df.loc[(df["layer1_static"] == False) & (df["ratio_KNS_KS"] > 100) &
               (df["MIN_EXPRESSION"] < 0.3)]
    
    df_normal = df.loc[(df["ignore_off_during_optimization"] == False) &
                (df["target_independent_of_clusters"] == False)]

    varnames_dict = plot_db.get_varname_to_value_dict(df)

    fntsz = 30
    plt.rcParams["font.size"] = f"{fntsz}"


    # ----- FIGURE 2 ----- #
    fig, ax =  plt.subplots(3,3,figsize=(36,36))

    # ROW 1
    plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df_normal["M_GENE"] == m_gene)],
                             ["M_GENE"],
                             [],[],
                             plot_db.rms_barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             ax=[ax[0][0]],
                             subtitles=["",""],
                             fontsize=fntsz,ylabel="RMS global expression error",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 500) & 
                                    (df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["target_independent_of_clusters"] == 0)],
                             ["ratio_KNS_KS"],
                             [],[],
                             plot_db.rms_barchart_groupby,
                             ["tf_first_layer","ignore_off_during_optimization"],
                             ax=[ax[0][1]],axlabel=" ",barcolors=["g","r"],
                             legloc="upper right",
                             fontsize=fntsz,ylabel="RMS global expression error",
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 500) & 
                                    (df["minimize_noncognate_binding"] == 0) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df["M_GENE"] == m_gene) &
                                    (df["ignore_off_during_optimization"] == 0)],
                             ["ratio_KNS_KS"],
                             [],[],
                             plot_db.rms_barchart_groupby,
                             ["tf_first_layer","target_independent_of_clusters"],
                             ax=[ax[0][2]],axlabel=" ",barcolors=["g","r"],
                             legloc="upper left",
                             fontsize=fntsz,ylabel="RMS global expression error",
                             varnames_dict=varnames_dict)

    # ROW 2
    plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["M_GENE"] == m_gene) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer"],
                             [],[],
                             plot_db.scatter_target_expression_groupby,
                             ["ratio_KNS_KS"],
                             fontsize=fntsz,ax=ax[1][0:2],
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df_normal["M_GENE"] == m_gene)],
                             "M_GENE",
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.effective_dynamic_range,
                             ax=[ax[1][2]],legloc="upper left",
                             subtitles=[""],fontsize=fntsz,
                             ylabel="effective dynamic range",
                             varnames_dict=varnames_dict)

    # ROW 3
    plot_db.subplots_groupby(df_normal.loc[(df_normal["M_GENE"] == m_gene) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["ratio_KNS_KS"] == 1000.0)],
                             ["tf_first_layer","ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_modulating_concentrations,
                             ax=ax[2][0:2],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df_normal.loc[(df_normal["M_GENE"] == m_gene) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["ratio_KNS_KS"] == 1000.0)],
                             ["ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_error_increase_by_modulating_concentration_groupby,
                             ["tf_first_layer"],
                             ax=[ax[2][2]],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    fig.tight_layout()
    plt.savefig("../plots/fig/fig2.png")
    plt.close()
    

    # ----- FIGURE 3 ----- #
    fig, ax =  plt.subplots(3,3,figsize=(36,36))

    # ROW 1
    plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000.0) &
                                    (df_normal["M_GENE"] == m_gene) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["tf_first_layer","ratio_KNS_KS"],
                             [],[],
                             plot_db.scatter_target_expression_groupby,
                             ["minimize_noncognate_binding"],
                             ax=ax[0][0:2],fontsize=fntsz,
                             varnames_dict=varnames_dict)

    plot_db.subplots_groupby(df_normal.loc[(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                    (df_normal["M_GENE"] == m_gene)],
                             "M_GENE",
                             [],[],
                             plot_db.barchart_groupby,
                             ["ratio_KNS_KS","tf_first_layer"],
                             plot_db.ratio_rms_error_patterning_noncognate_by_pair,
                             subtitles=["fold-change in RMS expression error \nwhen co-opt nontarget binding"],
                             ax=[ax[0][2]],fontsize=fntsz,
                             ylabel="fold-change",
                             varnames_dict=varnames_dict)

    # ROW 2
    plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                    (df_normal["M_GENE"] == m_gene) &
                                    (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
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
