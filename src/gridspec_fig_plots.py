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
import matplotlib.image as mpimg
from matplotlib import ticker
from matplotlib.lines import Line2D

HIGHLY_EXPRESSING_THRESHOLD = 0.8
RATIO_FOR_SINGLE_EXAMPLES = 1000
GEN_FIGURE_2 = False
GEN_FIGURE_3 = False
GEN_FIGURE_4 = False
GEN_FIGURE_5 = False
GEN_SUPPLEMENTAL = False
GEN_TEST = True

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
        extension = os.path.splitext(resfile)[1]
        if extension == ".hdf":
            df = pandas.read_hdf(resfile,"df")
        elif extension == ".pq":
            df = pandas.read_parquet(resfile)
        else:
            print(f"unsupported file extension {extension}")
            sys.exit()
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
    df["N_TF"] = df["N_TF"].astype(pandas.Int64Dtype())
    df = df.loc[(df["layer1_static"] == False) & (df["ratio_KNS_KS"] > 100) &
               (df["MIN_EXPRESSION"] < 0.3)]
    
    df_normal = df.loc[(df["ignore_off_during_optimization"] == False) &
                (df["target_independent_of_clusters"] == False) &
                (df["layer2_repressors"] == False) &
                (df["MIN_EXPRESSION"] > 0.01)]

    varnames_dict = plot_db.get_varname_to_value_dict(df)

    fntsz = 36
    insetfntsz = 28
    insetsz = 0.4
    plt.rcParams["font.size"] = f"{fntsz}"


    # ----- FIGURE 2 ----- #
    if GEN_FIGURE_2:
        fig = plt.figure(figsize=(30,20),layout="tight")

        outer = gs.GridSpec(2,1,height_ratios=[1,0.66])
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0],width_ratios=[2,1])
        scattertarget = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=inner0[0],wspace=0.1)
        inner1 = gs.GridSpecFromSubplotSpec(1,3,subplot_spec = outer[1],width_ratios=[1,1,2.2],wspace=0.25)
        scattermod = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=inner1[2],wspace=0.1)

        axd = {"A":plt.subplot(scattertarget[0]),
               "B":plt.subplot(scattertarget[1]),
               "C":plt.subplot(inner0[1]),
               "E":plt.subplot(inner1[0]),
               "G":plt.subplot(inner1[1]),
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
        axd["B"].set_ylabel("")
        plt.setp(axd["B"].get_yticklabels(),visible=False)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 ax=[axd["C"]],
                                 subtitles=["",""],
                                 fontsize=fntsz,ylabel="GEE",
                                 legloc="best",bbox_to_anchor=[0.48,0,0.47,0.47],
                                 varnames_dict=varnames_dict)

        ax_inset = axd["C"].inset_axes((0.53,0.58,insetsz,insetsz))
        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.rms_patterning_error,
                                 ax=[ax_inset],suppress_leg=True,
                                 subtitles=[""],fontsize=insetfntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="fold-reduction",logyax=True,
                                 #yticks=[0.1,0.3,0.5],
                                 varnames_dict=varnames_dict)
        ax_inset.set_box_aspect(1)
        
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
        axd["E"].plot([1e2,1e4],[0.81,0.81],linewidth=2,color="gray",linestyle="dashed")

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 0) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 0) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["G"].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        axd["G"].text(0.3,0.56,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
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

        axd["H"].text(0.26,0.62,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      transform=axd["H"].transAxes,va="center",ha="center")
        axd["I"].text(0.26,0.62,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      transform=axd["I"].transAxes,va="center",ha="center")

        plt.gcf().text(0.012,0.930,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.652,0.950,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.012,0.390,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.26,0.390,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.522,0.390,"E",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig2.png")
        plt.close()
    

    # ----- FIGURE 3 ----- #
    if GEN_FIGURE_3:
        fig = plt.figure(figsize=(22,16),layout="tight")

        #outer = gs.GridSpec(2,1,height_ratios=[0.5,1])
        #littler_plots = gs.GridSpecFromSubplotSpec(1,2,width_ratios=[1.2,1],subplot_spec=outer[0])
        #schematic = gs.GridSpecFromSubplotSpec(1,1,height_ratios=[1],subplot_spec=littler_plots[0])
        #scatter_actual = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = littler_plots[1],hspace=0.05)
        #boxy_plots = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1])

        outer = gs.GridSpec(2,1,height_ratios=[0.7,1])
        littler_plots = gs.GridSpecFromSubplotSpec(1,2,width_ratios=[1,1],subplot_spec=outer[0],
                                                   wspace=0.3)
        schematic = gs.GridSpecFromSubplotSpec(1,1,height_ratios=[1],subplot_spec=littler_plots[0])
        boxy_plots = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[1],width_ratios=[1,1],
                                                wspace=0.22)
        boxy_left = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=boxy_plots[0],height_ratios=[3,1])
        scatter_actual = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = boxy_left[0],hspace=0.05)

        axd = {"A":plt.subplot(littler_plots[1]),
               "B":plt.subplot(scatter_actual[0]),
               "C":plt.subplot(scatter_actual[1]),
               "D":plt.subplot(boxy_plots[1]),
               "E":plt.subplot(schematic[0])}

        nontarget_contribution_schematic = mpimg.imread("../plots/fig/nontarget_contribution_schematic2.png")
        axd["E"].imshow(nontarget_contribution_schematic)
        axd["E"].axis("off")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["minimize_noncognate_binding"],
                                 ax=[axd["B"],axd["C"]],fontsize=fntsz,
                                 subtitles=["",""],suppress_leg=True,
                                 markerdict={0:"o",1:"P"},
                                 colorbar_leg=False,gray_first_level=True,
                                 varnames_dict=varnames_dict)
        box1 = axd["B"].get_position()
        box2 = axd["C"].get_position()
        axd["B"].xaxis.set_label_coords(1.1,-0.15)
        axd["C"].set_ylabel("")
        axd["C"].set_xlabel("")
        axd["B"].set_xticks([0,1])
        axd["B"].set_yticks([0,1])
        axd["C"].set_xticks([0,1])
        axd["C"].set_yticks([0,1])
        plt.setp(axd["C"].get_yticklabels(),visible=False)

        color_dict = plot_db.get_varname_to_color_dict()
        legend_elements = [Line2D([0],[0],marker='P',color='w',
                                  markerfacecolor=color_dict["chromatin"],markersize=20,
                                  label="optimize binding (chromatin)"),
                           Line2D([0],[0],marker='P',color='w',
                                  markerfacecolor=color_dict["free DNA"],markersize=20,
                                  label="optimize binding (free DNA)"),
                           Line2D([0],[0],marker='o',color='w',
                                  markerfacecolor=[0.6,0.6,0.6],markersize=20,
                                  label="optimize expression")]
        customleg = axd["B"].legend(handles=legend_elements,bbox_to_anchor=(1.07,-0.65),loc="center")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.scatter_error_fraction_groupby,
                                 ["tf_first_layer","ratio_KNS_KS"],
                                 subtitles=[""],
                                 ax=[axd["A"]],fontsize=fntsz,
                                 colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        #axd["A"].plot(0.5,0.2,'X',color='k')
        #axd["A"].plot(0.5,0.14,'X',color='k')

        plot_db.subplots_groupby(df_normal.loc[(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                               (df_normal["M_GENE"] == m_gene)],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["D"]],fontsize=fntsz,draw_lines=True,markeralpha=1,
                                 size_lims=[500,500],
                                 subtitles=[""],ylabel="GEE",
                                 varnames_dict=varnames_dict)
        xticks = [1e2,1e3,1e4]
        axd["D"].set_yscale("log")
        axd["D"].set_xticks(xticks)
        axd["D"].plot([1e2,1e4],[0.018,0.018],"gray",linewidth=2,linestyle="dashed")
        axd["D"].set_xlim(xticks[0],xticks[-1])
        #axd["D"].set_box_aspect(1)

        plt.gcf().text(0.01,0.94,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.005,0.534,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.50,0.94,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.485,0.529,"D",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig3.png")
        plt.close()


    # ----- FIGURE 4 ----- #
    if GEN_FIGURE_4:
        fig = plt.figure(figsize=(30,20),layout="tight")

        outer = gs.GridSpec(1,2,width_ratios=[1.1,1])
        normal = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[0],height_ratios=[1.5,1])
        normal_scatter = gs.GridSpecFromSubplotSpec(2,2,subplot_spec=normal[0],height_ratios=[2.5,1],
                                                    wspace=0.05,hspace=0.05)
        normal_metrics = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=normal[1],wspace=0.3)

        extended = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[1],height_ratios=[1,1.5])
        extended_scatter = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=extended[0],
                                                    wspace=0.05,hspace=0.05)
        axd = {"A":plt.subplot(normal_scatter[0]),
               "B":plt.subplot(normal_scatter[1]),
               "C":plt.subplot(normal_metrics[0]),
               "D":plt.subplot(normal_scatter[2]),
               "E":plt.subplot(normal_scatter[3]),
               "F":plt.subplot(normal_metrics[1]),
               "G":plt.subplot(extended_scatter[0]),
               "H":plt.subplot(extended_scatter[1]),
               "I":plt.subplot(extended[1])}

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["layer2_repressors"],
                                 ax=[axd["A"],axd["B"]],fontsize=fntsz,
                                 colorbar_leg=False,gray_first_level=True,
                                 varnames_dict=varnames_dict)
        plt.setp(axd["A"].get_xticklabels(),visible=False)
        axd["A"].set_xlabel("")
        axd["A"].set_yticks([1])
        axd["B"].set_xlabel("")
        axd["B"].set_ylabel("")
        plt.setp(axd["B"].get_xticklabels(),visible=False)
        plt.setp(axd["B"].get_yticklabels(),visible=False)

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_repressor_activator,
                                 ["ratio_KNS_KS"],
                                 subtitles=["",""],
                                 ax=[axd["D"],axd["E"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        axd["E"].set_ylabel("")
        axd["E"].get_legend().remove()
        plt.setp(axd["E"].get_yticklabels(),visible=False)
        axd["D"].set_ylim([0,300])
        axd["E"].set_ylim([0,300])
        axd["D"].set_yticks([0,150,300])
        axd["E"].set_yticks([0,150,300])
        xover_coord1 = 0.36
        axd["D"].set_xticks([0,xover_coord1,1])
        axd["D"].set_xticklabels(["0",f"{xover_coord1}","1"])
        xover_coord = 0.28
        axd["E"].annotate("approx. leaky\nexpression level",xy=(xover_coord,0),xytext=(xover_coord,100),
                          arrowprops=dict(arrowstyle="-",linewidth=2,edgecolor="k"),ha="center",
                          fontsize=round(0.75*fntsz))
        axd["E"].set_xticks([0,xover_coord,1])
        axd["E"].set_xticklabels(["0",f"{xover_coord}","1"])

        plot_db.subplots_groupby(df_normal.loc[(df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["minimize_noncognate_binding"] == 0)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 subtitles=[""],
                                 ax=[axd["F"]],fontsize=fntsz,
                                 ylabel="dynamic range",
                                 suppress_leg=True,color=np.array([0.8,0.8,0.8]),force_color=True,
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range,
                                 subtitles=[""],
                                 ax=[axd["F"]],fontsize=fntsz,
                                 ylabel="dynamic range",
                                 legloc="lower right",
                                 varnames_dict=varnames_dict)
        axd["F"].plot([1e2,1e4],[0.81,0.81],linewidth=2,color="gray",linestyle="dashed")

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["C"]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] > 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["C"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["C"].set_ylim(0,0.06)
        axd["C"].set_yticks([0,0.03,0.06])

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] < 0.01)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["layer2_repressors"],subtitles=["",""],
                                 ax=[axd["G"],axd["H"]],fontsize=fntsz,
                                 colorbar_leg=False,gray_first_level=True,
                                 suppress_leg=True,
                                 varnames_dict=varnames_dict)
        plt.setp(axd["H"].get_yticklabels(),visible=False)
        axd["H"].set_ylabel("")
        axd["G"].set_xticks([0,1])
        axd["H"].set_xticks([0,1])

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["I"]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["I"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["I"].set_ylim(0,0.06)
        axd["I"].set_yticks([0,0.03,0.06])

        plt.setp(axd["A"].get_xticklabels(),visible=False)
        axd["A"].set_xlabel("")
        axd["A"].set_yticks([1])
        axd["B"].set_xlabel("")
        axd["B"].set_ylabel("")
        plt.setp(axd["B"].get_xticklabels(),visible=False)
        plt.setp(axd["B"].get_yticklabels(),visible=False)

        plt.gcf().text(0.012,0.950,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.012,0.604,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.012,0.384,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.268,0.384,"E",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.520,0.942,"F",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.527,0.550,"G",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig4.png")
        plt.close()


    """
    # ----- FIGURE 5 ----- #
    if GEN_FIGURE_5:
        maxclust_to_check = [50,60,70,80,100]
        ngenes = 1000
        for ii in maxclust_to_check:
            nentries = len(df_normal.loc[(df_normal['MAX_CLUSTERS_ACTIVE'] == ii)])
            print(f"{ii}: {nentries}")
        sys.exit()

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
                                 ylabel="GEE",
                                 varnames_dict=varnames_dict)
        def plot_guess(ax,N0_set,a,p,q):
            M = np.linspace(100,500,100)
            for N0 in N0_set:
                #ax.plot(M,a*(np.power(N0,p)*np.power(M-N0+1,q)),linewidth=5,color="k")
                ax.plot(M,a*(np.power(N0,p)*np.power(M,q)),linewidth=5,color="k")
        N0_set = [30,50,80,100]
        plot_guess(ax[0][0],N0_set=N0_set,a=6.6e-5,p=1.3,q=0.08)
        plot_guess(ax[0][0],N0_set=N0_set,a=4.8e-8,p=2.5,q=0.03)

        collapse_exponent_free_DNA = 1.3
        collapse_exponent_chromatin = 2.5

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["tf_first_layer"] == 1)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_free_DNA),
                                 ax=[ax[0][1]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,normalize=False,logfit=True,
                                 ylabel=f"RMSE / (fraction ON genes)^{collapse_exponent_free_DNA}",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["tf_first_layer"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_chromatin),
                                 ax=[ax[0][1]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,normalize=False,logfit=True,
                                 ylabel=f"RMSE / (fraction ON genes)^{collapse_exponent_chromatin}",
                                 varnames_dict=varnames_dict)
        ax[0][1].set_ylabel("RMSE / (fraction ON genes)^p")

        N = np.linspace(100,500,100)
        exponent = 1.4
        ax[0][1].plot(N,(0.035/np.power(100,exponent))*np.power(N,exponent),linewidth=5,color="k")
        exponent = 2.6
        ax[0][1].plot(N,(0.005/np.power(100,exponent))*np.power(N,exponent),linewidth=5,color="k")
        ax[0][1].set_yscale("log")

        def fraction_on(vals,cols):
            vals[cols[3]] = 10*vals[cols[2]].div(vals[cols[3]])
            return vals

        def numonoff(vals,cols):
            vals[cols[2]] = 10*vals[cols[2]]
            vals[cols[3]] = vals[cols[3]] - vals[cols[2]]
            return vals

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["tf_first_layer"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_chromatin),
                                 ax=[ax[0][2]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=False,
                                 transform_columns=numonoff,
                                 ylabel="RMSE / (fraction ON genes)^p",
                                 varnames_dict=varnames_dict)
        ax[0][2].set_xscale("linear")
        ax[0][2].set_xlabel("number OFF genes")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["tf_first_layer"] == 1)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.curve_collapse(x,collapse_exponent_chromatin/2),#collapse_exponent_free_DNA),
                                 ax=[ax[0][2]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=False,
                                 transform_columns=numonoff,
                                 ylabel="RMSE / (fraction ON genes)^p",
                                 varnames_dict=varnames_dict)
        ax[0][2].set_xscale("linear")
        ax[0][2].set_xlabel("number OFF genes")
    """

    # ----- FIGURE 5 ----- #
    if GEN_FIGURE_5:
        #fig, ax = plt.subplots(2,3,figsize=(42,42),layout="tight")
        fig = plt.figure(figsize=(20,10),layout="tight")

        outer = gs.GridSpec(1,2)

        axd = {"A":plt.subplot(outer[0]),
               "B":plt.subplot(outer[1])}

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["MAX_CLUSTERS_ACTIVE"] <= 10)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["A"]],fontsize=fntsz,
                                 subtitles=[""],
                                 suppress_leg=True,draw_lines=True,
                                 ylabel="GEE",
                                 varnames_dict=varnames_dict)
        #axd["A"].set_yscale("log")
        #axd["A"].yaxis.set_minor_formatter(ticker.NullFormatter())
        axd["A"].set_xticks([100,150,250,500,1000])
        axd["A"].xaxis.set_major_formatter(ticker.ScalarFormatter())
        axd["A"].xaxis.set_minor_formatter(ticker.NullFormatter())
        axd["A"].set_yticks([0,0.02,0.04])

        def numon(vals,cols):
            N_ON = 10*vals[cols[2]]
            M = vals[cols[3]]
            vals[cols[3]] = N_ON # x-axis
            vals[cols[2]] = N_ON.div(M) # marker sizing
            return vals

        def hundred_percent_on(df):
            d = plot_db.rms_patterning_error(df)
            fraction_on = df["MAX_CLUSTERS_ACTIVE"].div(df["N_CLUSTERS"],axis=0)
            d.loc[fraction_on < 1] = None
            return d

        def rel_to_hundred_percent_on(df,refs):
            d = plot_db.rms_patterning_error(df)
            return d.div(df["M_GENE"].map(refs))

        def get_refs(df):
            fraction_on = df["MAX_CLUSTERS_ACTIVE"].div(df["N_CLUSTERS"],axis=0)
            new_df = df.loc[fraction_on == 1]
            gb = new_df.groupby("M_GENE",group_keys=True)
            gbn = gb.apply(plot_db.rms_patterning_error)
            gbn = gbn.groupby("M_GENE",group_keys=True)
            return gbn.mean().to_dict()

        dfoi = df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                             (df_normal["minimize_noncognate_binding"] == 0)]
        #refs = get_refs(dfoi)
        
        plot_db.subplots_groupby(dfoi,
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,#lambda x: rel_to_hundred_percent_on(x,refs),
                                 ax=[axd["B"]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=False,
                                 subtitles=[""],
                                 transform_columns=numon,
                                 ylabel="GEE",# / max RMSE for M genes",
                                 varnames_dict=varnames_dict)
        axd["B"].set_xscale("log")
        axd["B"].set_yscale("log")
        axd["B"].set_xlabel("number of ON genes")

        plt.gcf().text(0.012,0.920,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.512,0.920,"B",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig5.png")
        plt.close()



    if GEN_SUPPLEMENTAL:
        #plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 1) &
                                        #(df_normal["M_GENE"] == m_gene) &
                                        #(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 #["tf_first_layer"],
                                 #[],[],
                                 #plot_db.scatter_error_fraction_groupby,
                                 #["ratio_KNS_KS"],
                                 #ax=[axd["F"],axd["G"]],fontsize=fntsz,
                                 #varnames_dict=varnames_dict)

        fig, ax = plt.subplots(4,4,figsize=(60,60),layout="tight")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                               (df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["MAX_CLUSTERS_ACTIVE"] <= 10) &
                                               (df_normal["M_GENE"] <= 500)],
                                 ["MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["tf_first_layer"],
                                 fontsize=fntsz,ax=ax,
                                 varnames_dict=varnames_dict)

        plt.savefig("../plots/fig/supp.png")
        plt.close()


    if GEN_TEST:
        df_filter = pandas.read_parquet("../fluctuation_res.pq")

        fig, ax = plt.subplots(2,3,figsize=(40,20),layout="tight")

        #plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0)],
                                 #["ratio_KNS_KS"],
                                 #[],[],
                                 #plot_db.hist_fluctuations_groupby,
                                 #["tf_first_layer"],
                                 #fontsize=fntsz,ax=ax,
                                 #varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_filter,
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.fluctuation_barchart_groupby,
                                 ["tf_first_layer"],
                                 ax=ax,
                                 fontsize=fntsz,ylabel="GEE",
                                 legloc="best",axlabel=" ",
                                 varnames_dict=varnames_dict)

        plt.savefig("../plots/fig/test3.png")
        plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
