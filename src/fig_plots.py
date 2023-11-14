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
GEN_FIGURE_2 = False
GEN_FIGURE_3 = False
GEN_TESTING = True

pandas.options.mode.chained_assignment = None


# FINAL CHECK: MAKE SURE ARE PLOTTING CORRECT ERROR METRIC IN ALL INSTANCES
# (i.e., check calls that use df["fun"] vs. those that calculate patterning
# error explicitly)

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

    fntsz = 36
    insetfntsz = 28
    insetsz = 0.37
    plt.rcParams["font.size"] = f"{fntsz}"


    # ----- FIGURE 2 ----- #
    if GEN_FIGURE_2:
        fig, ax =  plt.subplots(3,2,figsize=(28,44))

        # ROW 1
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["ratio_KNS_KS"],
                                 fontsize=fntsz,ax=ax[0][0:2],
                                 varnames_dict=varnames_dict)

        # ROW 2
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 ax=[ax[1][0]],
                                 subtitles=["",""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 legloc="best",bbox_to_anchor=[0.48,0,0.5,0.47],
                                 varnames_dict=varnames_dict)
        ax[1][0].set_box_aspect(1)
        # inset
        ax_inset = ax[1][0].inset_axes((0.57,0.60,insetsz,insetsz))
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax_inset],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="fold-change",
                                 yticks=[0.1,0.3,0.5],
                                 varnames_dict=varnames_dict)
        ax_inset.set_box_aspect(1)
        

        # PUT IN SUPPLEMENT
        """
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 500) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["target_independent_of_clusters"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["ignore_off_during_optimization","tf_first_layer"],
                                 ax=[ax[0][1]],axlabel=" ",
                                 legloc="upper left",
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        """

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 ax=[ax[1][1]],legloc="upper left",
                                 subtitles=[""],fontsize=fntsz,
                                 ylabel="dynamic range",
                                 varnames_dict=varnames_dict)
        ax[1][1].set_box_aspect(1)
        # inset
        ax_inset = ax[1][1].inset_axes((0.57,0.10,insetsz,insetsz))
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 ax=[ax_inset],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="fold-change",
                                 varnames_dict=varnames_dict)
        ax_inset.set_box_aspect(1)

        
        # ROW 3
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 500) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[ax[2][0]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        ax[2][0].set_box_aspect(1)
        """
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
        """

        #fig.tight_layout()
        plt.savefig("../plots/fig/fig2.png")
        plt.close()
    

    # ----- FIGURE 3 ----- #
    if GEN_FIGURE_3:
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
                                 subtitles=["fold-change in global expression error \nwhen co-opt nontarget binding"],
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


    if GEN_TESTING:
        # ----- TESTING ----- #
        #fig, ax =  plt.subplots(4,4,figsize=(48,48))
        fig, ax =  plt.subplots(4,4,figsize=(48,48))
        #plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
        #                                       (df_normal["minimize_noncognate_binding"] == 0)],
        #                         "ratio_KNS_KS",
        #                         [],[],
        #                         plot_db.colorplot_2d_groupby,
        #                         ["MAX_CLUSTERS_ACTIVE","M_GENE"],
        #                         lambda x: np.reciprocal(plot_db.ratio_rms_xtalk_chromatin_tf_by_pair(x)),
        #                         subtitles=["fold-improvement in global expression error \non using chromatin"], 
        #                         ax=[ax[0][2]],fontsize=fntsz,
        #                         varnames_dict=varnames_dict)
        """
        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.colorplot_2d_groupby,
                                 ["MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,
                                 ax=ax[0][0:2],fontsize=fntsz,
                                 varnames_dict=varnames_dict,
                                 colorbar_lims=[0,0.05])

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.colorplot_2d_groupby,
                                 ["MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.effective_dynamic_range,
                                 ax=ax[0][2:],fontsize=fntsz,
                                 varnames_dict=varnames_dict,
                                 colorbar_lims=[0.6,0.8])

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["ratio_KNS_KS"] == 1000)],
                                 ["MAX_CLUSTERS_ACTIVE"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["M_GENE","tf_first_layer"],
                                 ax=ax[1][0:],
                                 legloc="right",
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax[2][0]],fontsize=fntsz,
                                 xlabel="global expression error",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.effective_dynamic_range,
                                 ax=[ax[2][1]],fontsize=fntsz,
                                 xlabel="dynamic range",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000)],
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 [],[],
                                 plot_db.colorplot_2d_groupby,
                                 ["MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.ratio_rms_error_patterning_noncognate_by_pair,
                                 ax=ax[2][0:2],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == 500) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["target_independent_of_clusters"] == 0)],
                                 ["ignore_off_during_optimization"],
                                 [],[],
                                 plot_db.regulator_concentration_groupby,
                                 ["tf_first_layer"],
                                 ax=ax[3][0:],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        """

        plot_db.subplots_groupby(df_normal.loc[(df_normal["tf_first_layer"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","minimize_noncognate_binding"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax[3][2]],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,#linewidth=2,markersize=10,
                                 yticks=[0.1,0.3,0.5],
                                 varnames_dict=varnames_dict)
    
        fig.tight_layout()
        plt.savefig("../plots/fig/test.png")
        plt.close()
    

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
