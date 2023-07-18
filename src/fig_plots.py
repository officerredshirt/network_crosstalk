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

    prefixes = ['patterning','noncognate_binding']
    tf_prefix = ["chromatin","TF"]

    #plot_db.tf_vs_kpr_error_rate(df,"../plots/fig/")

    maxclust = 8
    m_gene = 250

    """
    for mnb in [0,1]:
        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == mnb) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene)],
                                 ["tf_first_layer","K_NS"],
                                 f"../plots/fig/{prefixes[mnb]}_error_by_modulating.png",
                                 f"{prefixes[mnb].replace('_',' ')}",
                                 plot_db.scatter_error_increase_by_modulating_concentration,
                                 subplot_dim=(2,3),fontsize=52,
                                 custom_subtitles = [{0:"chromatin",1:"TF only"},{}])
    """
    plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                    (df["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                             ["minimize_noncognate_binding","ratio_KNS_KS"],
                             f"../plots/fig/scatter_error_by_modulating{m_gene}_MAX_CLUSTERS_ACTIVE{maxclust}.png",
                             f"error by modulating",
                             plot_db.scatter_error_increase_by_modulating_concentration_groupby,
                             ["tf_first_layer"],subplot_dim=(2,3),fontsize=52,
                             custom_subtitles = [{0:"patterning error",1:"noncognate binding error"},{}],
                             leglabel={0:"chromatin",1:"TF only"})


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
