from numpy import *

import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import pandas

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

    BOXPLOT = True
    SCATTER_MODULATING = False
    SCATTER_TARGET_EXPRESSION = False
    PRESENTATION_PLOTS = False

    prefixes = ['patterning','noncognate_binding']
    tf_prefix = ["chromatin","TF"]

    plot_db.tf_vs_kpr_error_rate(df,"../plots/")

    if BOXPLOT:
        """
        for mnb in [0,1]:
            plot_db.subplots_groupby(df.loc[df["minimize_noncognate_binding"] == mnb],
                                     "K_NS",
                                     f"../plots/ratio_{prefixes[mnb]}_error.png",
                                     f"log ratio of {prefixes[mnb].replace('_',' ')} error in chromatin to equivalent TF networks",
                                     plot_db.boxplot_groupby,
                                     ["M_GENE","MAX_CLUSTERS_ACTIVE"],
                                     plot_db.ratio_xtalk_chromatin_tf_by_pair)
            plot_db.subplots_groupby(df.loc[df["minimize_noncognate_binding"] == mnb],
                                     "K_NS",
                                     f"../plots/{prefixes[mnb]}_by_gene.png",
                                     f"log {prefixes[mnb].replace('_',' ')} error per gene",
                                     plot_db.boxplot_groupby,
                                     ["M_GENE","MAX_CLUSTERS_ACTIVE","tf_first_layer"],
                                     plot_db.xtalk_by_gene)
        
        for mnb in [0,1]:
            plot_db.subplots_groupby(df.loc[df["minimize_noncognate_binding"] == mnb],
                                     "MAX_CLUSTERS_ACTIVE",
                                     f"../plots/ratio_{prefixes[mnb]}_error_compare_KNS.png",
                                     f"log ratio of {prefixes[mnb].replace('_',' ')} error in chromatin to equivalent TF networks",
                                     plot_db.boxplot_groupby,["K_NS"],
                                     plot_db.ratio_xtalk_chromatin_tf_by_pair)
            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == 8)],
                                     "M_GENE",
                                     f"../plots/{prefixes[mnb]}_by_gene_compare_KNS.png",
                                     f"log {prefixes[mnb].replace('_',' ')} error per gene",
                                     plot_db.boxplot_groupby,["K_NS","tf_first_layer"],
                                     plot_db.xtalk_by_gene)
        for tf in [0,1]:
            plot_db.subplots_groupby(df.loc[df["tf_first_layer"] == tf],
                                     "K_NS",
                                     f"../plots/ratio_{tf_prefix[tf]}_patterning_noncognate_error.png",
                                     f"log ratio of patterning to noncognate error in {tf_prefix[tf].replace('_',' ')} networks",
                                     plot_db.boxplot_groupby,
                                     ["M_GENE","MAX_CLUSTERS_ACTIVE"],
                                     plot_db.ratio_patterning_noncognate_by_pair)
        for max_cluster in [3,5,8]:
            plot_db.subplots_groupby(df.loc[df["MAX_CLUSTERS_ACTIVE"] == max_cluster],
                                     "M_GENE",
                                     f"../plots/ratio_patterning_noncognate_error_maxclust{max_cluster}.png",
                                     f"log ratio of patterning to noncognate error for max {max_cluster} active clusters",
                                     plot_db.boxplot_groupby,
                                     ["K_NS","tf_first_layer"],
                                     plot_db.ratio_patterning_noncognate_by_pair)
            """
        #for m_gene in [100, 150, 250]:
        plot_db.subplots_groupby(df,#df.loc[(df["M_GENE"] == m_gene)],
                                 ["K_NS","MAX_CLUSTERS_ACTIVE"],
                                 f"../plots/for_gasper/patterning_error.png",#_M_GENE{m_gene}.png",
                                 f"log patterning error",
                                 plot_db.boxplot_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","M_GENE"],
                                 plot_db.patterning_error,
                                 subplot_dim=(3,3),fontsize=52,
                                 axlabel="TF-only (true/false), \nminimize noncognate binding (true/false), \n# genes")

    if SCATTER_MODULATING:
        cols = ["M_GENE","MAX_CLUSTERS_ACTIVE"]
        for mnb in [0,1]:
            for tf_first_layer in [0,1]:
                for k_ns in [1000,10000,100000]:
                    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                                    (df["tf_first_layer"] == tf_first_layer) & (df["K_NS"] == k_ns)],
                                             cols,
                                             f"../plots/{prefixes[mnb]}_modulating_concentrations_{tf_prefix[tf_first_layer]}_K_NS{k_ns}.png",
                                             f"{prefixes[mnb].replace('_',' ')} modulating concentrations, {tf_prefix[tf_first_layer]}, K_NS = {k_ns} ({cols})",
                                             plot_db.scatter_modulating_concentrations)

    if SCATTER_TARGET_EXPRESSION:
        cols = ["MAX_CLUSTERS_ACTIVE"]
        m_gene_list = [100,150,250]
        for mnb in [0,1]:
            for tf_first_layer in [0,1]:
                for m_gene in m_gene_list:
                    plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                                    (df["tf_first_layer"] == tf_first_layer) &
                                                    (df["M_GENE"] == m_gene)],
                                                 cols,
                                                 f"../plots/{prefixes[mnb]}_scatter_target_expression_{tf_prefix[tf_first_layer]}_M_GENE{m_gene}.png",
                                                 f"{prefixes[mnb].replace('_',' ')} target expression, {tf_prefix[tf_first_layer]}, M_GENE = {m_gene} ({cols})",
                                                 plot_db.scatter_target_expression_groupby,["K_NS"])

    if PRESENTATION_PLOTS:
        cols = ["tf_first_layer"]
        m_gene = 250
        k_ns = 100000
        maxclust = 8
        """
        for mnb in [0,1]:
            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["M_GENE"] == m_gene) &
                                            (df["K_NS"] == k_ns) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                     cols,
                                     f"../plots/unitsem/{prefixes[mnb]}_scatter_target_expression_K_NS{k_ns}_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                     f"{prefixes[mnb].replace('_',' ' )} target expression",
                                     plot_db.scatter_target_expression_groupby,
                                     ["K_NS"],fontsize=36,
                                     subtitle_include_supercol = False)

            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["M_GENE"] == m_gene) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                     cols,
                                     f"../plots/unitsem/{prefixes[mnb]}_scatter_target_expression_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                     f"{prefixes[mnb].replace('_',' ' )} target expression",
                                     plot_db.scatter_target_expression_groupby,
                                     ["K_NS"],fontsize=36,
                                     subtitle_include_supercol = False)

            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["M_GENE"] == m_gene) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                     ["K_NS"],
                                     f"../plots/unitsem/{prefixes[mnb]}_scatter_patterning_residuals_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                     f"{prefixes[mnb].replace('_',' ' )} error",
                                     plot_db.scatter_patterning_residuals_groupby,
                                     ["tf_first_layer"],subplot_dim=(1,3),fontsize=36,
                                     subtitle_include_supercol = True)

            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["M_GENE"] == m_gene) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                     ["K_NS"],
                                     f"../plots/unitsem/{prefixes[mnb]}_scatter_error_fraction_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                     f"{prefixes[mnb].replace('_',' ' )} error",
                                     plot_db.scatter_error_fraction_groupby,
                                     ["tf_first_layer"],subplot_dim=(1,3),fontsize=36,
                                     subtitle_include_supercol = True)

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer","K_NS"],
                                 f"../plots/unitsem/scatter_target_expression_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                 f"target expression",
                                 plot_db.scatter_target_expression_groupby,
                                 ["minimize_noncognate_binding"],subplot_dim=(2,3),fontsize=52,
                                 custom_subtitles = [{0:"chromatin",1:"TF only"},{}],
                                 leglabel={0:"patterning error",1:"noncognate binding error"})

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 "K_NS",
                                 f"../plots/unitsem/patterning_error_M_GENE{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                                 f"log patterning error",
                                 plot_db.boxplot_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding"],
                                 plot_db.patterning_error,
                                 subplot_dim=(1,3),fontsize=52,
                                 axlabel="TF-only (true/false), \nminimize noncognate binding (true/false)")

        """

        for mnb in [0,1]:
            plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                            (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                            (df["M_GENE"] == m_gene)],
                                     ["tf_first_layer","K_NS"],
                                     f"../plots/unitsem/{prefixes[mnb]}_modulating_concentrations.png",
                                     f"{prefixes[mnb].replace('_',' ')} modulating concentrations",
                                     plot_db.scatter_modulating_concentrations,
                                     subplot_dim=(2,3),fontsize=52,
                                     custom_subtitles = [{0:"chromatin",1:"TF only"},{}])


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
