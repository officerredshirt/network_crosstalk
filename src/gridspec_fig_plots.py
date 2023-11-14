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
import matplotlib.gridspec as gs
from matplotlib.ticker import FormatStrFormatter

HIGHLY_EXPRESSING_THRESHOLD = 0.8
RATIO_FOR_SINGLE_EXAMPLES = 1000
GEN_FIGURE_2 = False
GEN_FIGURE_3 = False
GEN_FIGURE_4 = True
GEN_SUPPLEMENTAL = False

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
    plt.rcParams["font.size"] = f"{fntsz}"


    # ----- FIGURE 2 ----- #
    if GEN_FIGURE_2:
        fig = plt.figure(figsize=(28,36),layout="tight")

        outer = gs.GridSpec(3,1,height_ratios=[1,1,1])
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0])
        inner1 = gs.GridSpecFromSubplotSpec(1,3,subplot_spec = outer[1],width_ratios=[1,1,0.75],
                                            wspace=0.25)
        inner2 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[2],width_ratios=[0.25,1])

        ratios = gs.GridSpecFromSubplotSpec(2,1,subplot_spec = inner1[2],hspace=0.05)
        scattermod = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = inner2[1],wspace=0.05)

        axd = {"A":plt.subplot(inner0[0]),
               "B":plt.subplot(inner0[1]),
               "C":plt.subplot(inner1[0]),
               "D":plt.subplot(ratios[0]),
               "E":plt.subplot(inner1[1]),
               "F":plt.subplot(ratios[1]),
               "G":plt.subplot(inner2[0]),
               "H":plt.subplot(scattermod[0]),
               "I":plt.subplot(scattermod[1])}

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["ratio_KNS_KS"],
                                 fontsize=fntsz,ax=[axd["A"],axd["B"]],
                                 varnames_dict=varnames_dict)
        drmin = 0.26
        drmax = 0.75
        drxpos = 0.97
        arrowprops = dict(arrowstyle="<->",linewidth=2,mutation_scale=40)
        lineprops = dict(arrowstyle="-",linewidth=2,mutation_scale=40,
                               edgecolor=[0.5,0.5,0.5])
        axd["A"].annotate("",xy=(drxpos,drmin),xytext=(drxpos,drmax),
                          arrowprops=arrowprops)
        axd["A"].annotate("",xy=(drmin+0.01,drmin),xytext=(drxpos,drmin),
                          arrowprops=lineprops)
        axd["A"].annotate("",xy=(0.9,drmax),xytext=(drxpos,drmax),
                          arrowprops=lineprops)
        axd["A"].text(drxpos-0.015,drmin+(drmax-drmin)/2,"dynamic range",
                      va="center",ha="right",rotation=90)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 ax=[axd["C"]],
                                 subtitles=["",""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 legloc="upper right",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["D"]],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="fold-change in error",
                                 yticks=[0.1,0.3,0.5],
                                 varnames_dict=varnames_dict)
        
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 ax=[axd["E"]],legloc="lower right",
                                 subtitles=[""],fontsize=fntsz,
                                 ylabel="dynamic range",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 ax=[axd["F"]],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="fold-change in dynamic range",
                                 yticks=[1,1.1,1.2,1.3],
                                 varnames_dict=varnames_dict)

        axd["C"].set_box_aspect(1.2)
        axd["D"].set_box_aspect(1)
        axd["D"].set_xlabel(" ")
        plt.setp(axd["D"].get_xticklabels(),visible=False)
        axd["E"].set_box_aspect(1.2)
        axd["F"].set_box_aspect(1)
        
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        axd["G"].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        axd["G"].text(0.38,0.75,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      transform=axd["G"].transAxes,va="center",ha="center")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_modulating_concentrations,
                                 ax=[axd["H"],axd["I"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        axd["H"].set_xticks([1e0,1e1,1e2])
        axd["I"].set_ylabel("")
        axd["I"].set_xticks([1e0,1e1,1e2])
        plt.setp(axd["I"].get_yticklabels(),visible=False)

        plt.gcf().text(0.01,0.950,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.01,0.635,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.37,0.635,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.72,0.647,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.01,0.317,"E",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.26,0.317,"F",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig2.png")
        plt.close()
    
    # ----- FIGURE 3 ----- #
    if GEN_FIGURE_3:
        fig = plt.figure(figsize=(28,36),layout="tight")

        outer = gs.GridSpec(3,1,height_ratios=[1,1,0.8])
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0])
        inner1 = gs.GridSpecFromSubplotSpec(1,3,subplot_spec = outer[1])
        inner2 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[2],wspace=0.05)

        axd = {"A":plt.subplot(inner0[0]),
               "B":plt.subplot(inner0[1]),
               "C":plt.subplot(inner1[0]),
               "D":plt.subplot(inner1[1]),
               "E":plt.subplot(inner1[2]),
               "F":plt.subplot(inner2[0]),
               "G":plt.subplot(inner2[1])}


        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["minimize_noncognate_binding"],
                                 ax=[axd["C"],axd["D"]],fontsize=fntsz,
                                 colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        box1 = axd["C"].get_position()
        box2 = axd["D"].get_position()
        axd["C"].text(1.1,1.2,f"intrinsic specificity = {RATIO_FOR_SINGLE_EXAMPLES}",fontsize=fntsz,
                      ha="center",va="center")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer","minimize_noncognate_binding"],
                                 plot_db.rms_patterning_error,
                                 subtitles=[""],
                                 ax=[axd["E"]],fontsize=fntsz,take_ratio=True,
                                 ylabel="fold-change in error upon co-opting",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_error_fraction_groupby,
                                 ["ratio_KNS_KS"],
                                 ax=[axd["A"],axd["B"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                               (df_normal["M_GENE"] == m_gene)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["F"]],fontsize=fntsz,draw_lines=True,markeralpha=1,
                                 size_lims=[500,500],
                                 subtitles=[""],ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        xticks = [1e2,1e3,1e4]
        axd["F"].set_xticks(xticks)
        axd["F"].plot([1e2,1e4],[0.018,0.018],"gray",linewidth=2,linestyle="dashed")
        axd["F"].set_xlim(xticks[0],xticks[-1])

        plt.gcf().text(0.01,0.96,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.01,0.62,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.66,0.625,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.01,0.275,"D",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig3.png")
        plt.close()

    # ----- FIGURE 4 ----- #
    if GEN_FIGURE_4:
        fig, ax = plt.subplots(2,3,figsize=(42,42),layout="tight")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax[0][0]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,
                                 ylabel="global expression error",
                                 varnames_dict=varnames_dict)

        collapse_exponent_free_DNA = 1.3
        collapse_exponent_chromatin = 2.6
        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_free_DNA),
                                 ax=[ax[0][1]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,
                                 ylabel=f"RMSE / (# ON genes)^{collapse_exponent_free_DNA}",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_chromatin),
                                 ax=[ax[0][2]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,
                                 ylabel=f"RMSE / (# ON genes)^{collapse_exponent_chromatin}",
                                 varnames_dict=varnames_dict)
        #ax[0][2].set_ylim(0,5e-13)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["tf_first_layer"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","minimize_noncognate_binding"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax[1][0]],suppress_leg=False,
                                 subtitles=[""],fontsize=fntsz,#linewidth=2,markersize=10,
                                 varnames_dict=varnames_dict)
        ax[1][0].set_ylabel("patterning error")


        plot_db.subplots_groupby(df_normal.loc[(df_normal["tf_first_layer"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","minimize_noncognate_binding"],
                                 lambda x: np.log(plot_db.rms_patterning_error(x)),
                                 ax=[ax[1][1]],suppress_leg=False,
                                 subtitles=[""],fontsize=fntsz,#linewidth=2,markersize=10,
                                 varnames_dict=varnames_dict)
        ax[1][1].set_ylabel("patterning error")

        plt.savefig("../plots/fig/fig4.png")
        plt.close()



        if GEN_SUPPLEMENTAL:
            plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 1) &
                                            (df_normal["M_GENE"] == m_gene) &
                                            (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                     ["tf_first_layer"],
                                     [],[],
                                     plot_db.scatter_error_fraction_groupby,
                                     ["ratio_KNS_KS"],
                                     ax=[axd["F"],axd["G"]],fontsize=fntsz,
                                     varnames_dict=varnames_dict)
            # TODO: colorscatter for dynamic range?

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
