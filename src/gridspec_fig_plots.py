import shelve
import dill
import manage_db
import plot_db
import sys, argparse
import os
import pandas
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.image as mpimg
import matplotlib.patches as patches
from matplotlib import ticker
from matplotlib.lines import Line2D

matplotlib.use("agg")

HIGHLY_EXPRESSING_THRESHOLD = 0.8
RATIO_FOR_SINGLE_EXAMPLES = 1000
GEN_FIGURE_2 = False
GEN_FIGURE_3 = False
GEN_FIGURE_4 = True
GEN_FIGURE_5 = False
GEN_FIGURE_5_FORMER = False
GEN_SUPPLEMENTAL = False
GEN_NOISE = False
GEN_DIST_TEST = False
GEN_TEST = False

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
                (df["MIN_EXPRESSION"] > 0.01) &
                (df["target_distribution"] == "uni")]

    df_filter_0p1 = pandas.read_parquet(f"../fluctuation_res_sigma0.1.pq")
    df_filter_0p05 = pandas.read_parquet(f"../fluctuation_res_sigma0.05.pq")
    df_filter_0p2 = pandas.read_parquet(f"../fluctuation_res_sigma0.2.pq")


    varnames_dict = plot_db.get_varname_to_value_dict(df)

    fntsz = 36
    insetfntsz = 28
    insetsz = 0.4
    biginsetsz = 0.5
    plt.rcParams["font.size"] = f"{fntsz}"
    pale_yellow = np.array([254,252,158])/255


    # ----- FIGURE 2 ----- #
    if GEN_FIGURE_2:
        fig = plt.figure(figsize=(30,20),layout="tight")

        outer = gs.GridSpec(2,1,height_ratios=[1,0.66])
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0],width_ratios=[2,1])
        scattertarget = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=inner0[0],wspace=0.1)
        inner1 = gs.GridSpecFromSubplotSpec(1,4,subplot_spec = outer[1],wspace=0.3)
        scattermod = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=inner1[2],hspace=0.05)

        axd = {"A":plt.subplot(scattertarget[0]),
               "B":plt.subplot(scattertarget[1]),
               "C":plt.subplot(inner0[1]),
               "E":plt.subplot(inner1[0]),
               "G":plt.subplot(inner1[1]),
               "H":plt.subplot(scattermod[0]),
               "I":plt.subplot(scattermod[1]),
               "J":plt.subplot(inner1[3])}

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
                                 markers=["h"],
                                 varnames_dict=varnames_dict)
        ax_inset.set_box_aspect(1)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range_fold_change,
                                 ax=[axd["E"]],legloc="lower right",
                                 subtitles=[""],fontsize=fntsz,
                                 ylabel="dynamic range\n(fold-change)",
                                 varnames_dict=varnames_dict)
        axd["E"].plot([1e2,1e4],[10,10],linewidth=2,color="gray",linestyle="dashed")
        axd["E"].set_yticks([0,5,10])

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 0) &
                                        (df["MIN_EXPRESSION"] > 0.01) &
                                        (df["target_distribution"] == "uni")],
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
                                        (df["MIN_EXPRESSION"] > 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["G"].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        #axd["G"].text(0.3,0.56,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      #transform=axd["G"].transAxes,va="center",ha="center")
        axd["G"].set_yticks([0,0.1])

        plot_db.subplots_groupby(df_normal.loc[(df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_modulating_concentrations,
                                 subtitles=["",""],
                                 ax=[axd["H"],axd["I"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        axd["H"].set_ylim([0.5,0.91])
        axd["H"].set_xlim([8,1000])
        axd["H"].set_xticks([1e1,1e2,1e3])
        axd["H"].set_xlabel("")
        axd["H"].get_legend().remove()
        axd["H"].set_ylabel("target expression")
        axd["H"].yaxis.set_label_coords(-0.15,-0.02)
        plt.setp(axd["H"].get_xticklabels(),visible=False)


        axd["I"].set_xlim([8,1000])
        axd["I"].set_ylim([0.5,0.91])
        axd["I"].set_ylabel("")
        axd["I"].set_xticks([1e1,1e2,1e3])
        axd["I"].get_legend().remove()

        #axd["I"].text(0.76,0.32,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      #transform=axd["I"].transAxes,va="center",ha="center")

        legend_elements = [Line2D([0],[0],marker='o',color='none',markersize=15,markeredgecolor="none",
                                  markerfacecolor=plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                  label="global"),
                           Line2D([0],[0],marker='o',color='none',markersize=15,markeredgecolor="none",
                                  markerfacecolor=0.5*plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                  label="selfish TF"),
                           Line2D([0],[0],color='k',linewidth=2,label="induction\ncurve")]
        customleg = axd["H"].legend(handles=legend_elements,handlelength=0.7,
                                    bbox_to_anchor=(0.76,0.33),loc="center",frameon=False,
                                    fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))


        def numon(vals,cols):
            N_ON = 10*vals[cols[2]]
            M = vals[cols[3]]
            vals[cols[3]] = N_ON # x-axis
            vals[cols[2]] = N_ON.div(M) # marker sizing
            return vals
        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 plot_db.rms_patterning_error,#lambda x: rel_to_hundred_percent_on(x,refs),
                                 ax=[axd["J"]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=False,
                                 subtitles=[""],
                                 transform_columns=numon,
                                 ylabel="GEE",# / max RMSE for M genes",
                                 varnames_dict=varnames_dict)
        axd["J"].set_xscale("log")
        axd["J"].set_yscale("log")
        ylims = axd["J"].get_ylim()
        axd["J"].plot([maxclust*10,maxclust*10],ylims,linewidth=2,color="gray",linestyle="dashed",
                      zorder=0)
        axd["J"].set_ylim(ylims)
        axd["J"].set_xlabel("number of ON genes")

        legend_elements = [Line2D([0],[0],marker='o',color='none',markersize=10,
                                  markeredgecolor="k",
                                  markerfacecolor=plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                  label=" "),
                           Line2D([0],[0],marker='o',color='none',markersize=((np.sqrt(500)-10)/2)+10,
                                  markeredgecolor="k",
                                  markerfacecolor=plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                  label=" "),
                           Line2D([0],[0],marker='o',color='none',markersize=np.sqrt(500),
                                  markeredgecolor="k",
                                  markerfacecolor=plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                  label=" ")]
        customleg = axd["J"].legend(handles=legend_elements,handlelength=0,handletextpad=0,
                                    bbox_to_anchor=(0.69,0.18),loc="center",frameon=False,
                                    fontsize=round(plot_db.LEG_FONT_RATIO*fntsz),
                                    title="ON genes/\ntotal genes",ncol=3)
        customleg.get_title().set_multialignment("center")
        
        txt = matplotlib.offsetbox.TextArea("0            1")
        box = customleg._legend_box
        box.get_children().append(txt)
        box.set_figure(box.figure)
        

        # yellow highlight
        rectwidth = 1
        rectx = 2.125-rectwidth/2
        rect = patches.Rectangle((rectx,0),rectwidth,0.038, \
                linewidth=3,edgecolor="none",facecolor=pale_yellow,zorder=0)
        axd["C"].add_patch(rect)

        axd["G"].set_facecolor("none")
        axd["H"].set_facecolor("none")
        axd["I"].set_facecolor("none")
        axd["J"].set_facecolor("none")

        plt.gcf().text(0.014,0.930,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.652,0.950,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.014,0.390,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.268,0.390,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.51,0.390,"E",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.748,0.390,"F",fontsize=fntsz,fontweight="bold")

        polypts = ((rectx,0),(-0.24,1.1),(-0.24,-0.15),(1.08,-0.15),(1.08,1.1),(rectx+rectwidth,0))
        polyax = (axd["C"],axd["G"],axd["G"],axd["J"],axd["J"],axd["C"])
        polycoords = ("data","axes fraction","axes fraction","axes fraction","axes fraction","data")

        l = np.zeros((3*(len(polypts)-1),2))
        for ii in range(1,len(polypts)):
            p = matplotlib.patches.ConnectionPatch(polypts[ii-1],polypts[ii], \
                    coordsA=polycoords[ii-1],coordsB=polycoords[ii], \
                    axesA=polyax[ii-1],axesB=polyax[ii], \
                    color="none")
            axd["C"].add_artist(p)
            l[3*(ii-1):(3*(ii-1)+3),:] = p.get_path().vertices
        fig.patches.extend([plt.Polygon(l,ec="none",fc=pale_yellow,zorder=-100, \
                transform=axd["C"].transData,clip_on=False)])

        """
        p1 = matplotlib.patches.ConnectionPatch((rectx,0),(0,1), \
                coordsA="data",coordsB="axes fraction", \
                axesA=axd["C"],axesB=axd["G"], \
                color="k",linewidth=3,facecolor=pale_yellow)
        p2 = matplotlib.patches.ConnectionPatch((rectx+rectwidth,0),(1,1), \
                coordsA="data",coordsB="axes fraction", \
                axesA=axd["C"],axesB=axd["J"], \
                color="k",linewidth=3,facecolor=pale_yellow)
        axd["C"].add_artist(p1)
        axd["C"].add_artist(p2)
        """


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

        legend_elements = [Line2D([0],[0],marker='P',color='w',
                                  markerfacecolor=plot_db.color_dict["chromatin"],markersize=20,
                                  label="optimize binding (chromatin)"),
                           Line2D([0],[0],marker='P',color='w',
                                  markerfacecolor=plot_db.color_dict["free DNA"],markersize=20,
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
        fig = plt.figure(figsize=(30,16),layout="tight")

        outer = gs.GridSpec(2,1,height_ratios=[5,1])
        top = gs.GridSpecFromSubplotSpec(1,2,subplot_spec=outer[0],width_ratios=[1,0.55],wspace=0.15)
        left = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=top[0],height_ratios=[1.2,1],hspace=0.15)
        metrics = gs.GridSpecFromSubplotSpec(1,3,subplot_spec=left[1],wspace=0.55)
        extended_scatter = gs.GridSpecFromSubplotSpec(2,2,subplot_spec=top[1],height_ratios=[1.5,1],
                                                    wspace=0.07,hspace=0.15)
        """
        on_distributions = gs.GridSpecFromSubplotSpec(1,4,subplot_spec=outer[1],width_ratios=[3.2,1,3.2,1],
                                                      wspace=0.15)
        A_histograms = gs.GridSpecFromSubplotSpec(1,3,subplot_spec=on_distributions[0],
                                                   wspace=0.05)
        AR_histograms = gs.GridSpecFromSubplotSpec(1,3,subplot_spec=on_distributions[2],
                                                   wspace=0.05)
        """
        histograms = gs.GridSpecFromSubplotSpec(1,8,subplot_spec=outer[1],wspace=0.5, \
                width_ratios=[1,1,1,0.001,1,1,1,0.001])

        extended = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=outer[1],height_ratios=[1,1.5])
        axd = {"schematic":plt.subplot(left[0]),
               "A":plt.subplot(extended_scatter[0]),
               "B":plt.subplot(extended_scatter[1]),
               "C":plt.subplot(metrics[0]),
               "Ca":plt.subplot(metrics[1]),
               "D":plt.subplot(extended_scatter[2]),
               "E":plt.subplot(extended_scatter[3]),
               "F":plt.subplot(metrics[2]),
               "G":plt.subplot(histograms[0]),
               "H":plt.subplot(histograms[1]),
               "I":plt.subplot(histograms[2]),
               "K":plt.subplot(histograms[4]),
               "L":plt.subplot(histograms[5]),
               "M":plt.subplot(histograms[6])}
        """
               "G":plt.subplot(A_histograms[0]),
               "H":plt.subplot(A_histograms[1]),
               "I":plt.subplot(A_histograms[2]),
               "J":plt.subplot(on_distributions[1]),
               "K":plt.subplot(AR_histograms[0]),
               "L":plt.subplot(AR_histograms[1]),
               "M":plt.subplot(AR_histograms[2]),
               "N":plt.subplot(on_distributions[3])}
        """

        repressor_schematic = mpimg.imread("../plots/fig/repressor_schematic2.png")
        axd["schematic"].imshow(repressor_schematic)
        axd["schematic"].axis("off")

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["layer2_repressors"],
                                 ax=[axd["A"],axd["B"]],fontsize=fntsz,
                                 colorbar_leg=False,suppress_leg=True,
                                 gray_first_level=True,set_box_aspect=False,
                                 #color_list=[plot_db.color_dict["activators only"],
                                             #plot_db.color_dict["with repressors"]],
                                 legloc="best",
                                 varnames_dict=varnames_dict)
        axlim = 0.4
        xover_coord_chromatin = 0.30
        xover_coord_free_DNA = 0.27

        plt.setp(axd["B"].get_yticklabels(),visible=False)
        axd["B"].set_ylabel("")

        axd["A"].set_xlim([0,axlim+0.01])
        axd["A"].set_ylim([0,axlim])
        axd["A"].set_xticks([0,axlim])
        axd["A"].set_xticklabels(["0",f"{axlim}"])
        axd["A"].set_yticks([0,axlim])
        axd["A"].set_yticklabels(["0",f"{axlim}"])
        axd["A"].set_xlabel("")

        axd["B"].set_xlim([0,axlim+0.01])
        axd["B"].set_ylim([0,axlim])
        axd["B"].set_xticks([0,axlim])
        axd["B"].set_xticklabels(["0",f"{axlim}"])
        axd["B"].set_yticks([0,axlim])
        axd["B"].set_yticklabels(["0",f"{axlim}"])
        axd["B"].set_xlabel("")

        """
        ax_inset_a = axd["A"].inset_axes((0.57,0.07,insetsz,insetsz))
        ax_inset_b = axd["B"].inset_axes((0.57,0.07,insetsz,insetsz))
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["layer2_repressors"],subtitles=["",""],
                                 ax=[ax_inset_a,ax_inset_b],fontsize=insetfntsz,
                                 colorbar_leg=False,
                                 gray_first_level=True,
                                 suppress_leg=True,
                                 varnames_dict=varnames_dict)
        def adjust_inset(ax,add_box=True):
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([0,1])
            ax.set_yticks([0,1])
            if add_box:
                ax.add_patch(patches.Rectangle((0,0),axlim,axlim,linewidth=2, \
                        edgecolor=[0.5,0.5,0.5],facecolor='none',zorder=20))
        adjust_inset(ax_inset_a)
        adjust_inset(ax_inset_b)
        """

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
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
        axd["D"].set_xticks([0,1])
        axd["E"].set_xticks([0,1])

        axd["A"].text(0.21,0.20,"A+R",va="top",ha="left",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        axd["A"].text(0.045,0.166,"A",va="top",ha="left",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        axd["B"].text(0.24,0.23,"A+R",va="top",ha="left",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        axd["B"].text(0.055,0.205,"A",va="top",ha="left",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))

        axd["A"].plot(xover_coord_chromatin,xover_coord_chromatin,"*",markersize=25, \
                color=np.array([252,247,150])/255,markeredgecolor="k",zorder=100,clip_on=False)
        legend_elements = axd["B"].plot(xover_coord_free_DNA,xover_coord_free_DNA,"*",markersize=25, \
                color=np.array([252,247,150])/255,markeredgecolor="k",zorder=100,clip_on=False, \
                label="baseline\nexpression")
        axd["D"].plot(xover_coord_chromatin,0,"*",markersize=25, \
                color=np.array([252,247,150])/255,markeredgecolor="k",zorder=100,clip_on=False)
        axd["E"].plot(xover_coord_free_DNA,0,"*",markersize=25, \
                color=np.array([252,247,150])/255,markeredgecolor="k",zorder=100,clip_on=False, \
                label="baseline\nexpression")
        customleg = axd["E"].legend(handles=legend_elements,handletextpad=0.2,
                                    fontsize=round(plot_db.LEG_FONT_RATIO*fntsz),
                                    frameon=False)

        #axd["E"].annotate("baseline\nexpression",xy=(xover_coord_free_DNA,0),xytext=(xover_coord,100),
                          #arrowprops=dict(arrowstyle="-",linewidth=2,edgecolor="k"),ha="center",
                          #fontsize=round(0.75*fntsz))

        """
        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["tf_first_layer","layer2_repressors"],
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
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["tf_first_layer","layer2_repressors"],
                                 ax=[axd["C"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["C"].set_ylim(0,0.06)
        axd["C"].set_yticks([0,0.03,0.06])
        """
        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["layer2_repressors"] == 0) &
                                        (df["target_distribution"] == "uni")],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["C"]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,
                                 take_ratio=True,ylabel="GEE fold-reduction",logyax=True,
                                 linestyle="dashed",
                                 markers=["o"],
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["target_distribution"] == "uni")],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["C"]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="GEE fold-reduction\nchromatin / free DNA",logyax=True,
                                 #force_color=True,color=plot_db.color_dict["with repressors"],
                                 markers=["v"],
                                 varnames_dict=varnames_dict)
        axd["C"].text(550,8,"A+R",va="top",ha="right",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        axd["C"].text(2000,3.1,"A",va="top",ha="right",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))

        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["tf_first_layer"] == 0) &
                                        (df["target_distribution"] == "uni")],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","layer2_repressors"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["Ca"]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,logyax=True,
                                 force_color=True,color=plot_db.color_dict["chromatin"],
                                 markers=["o"],reverse_ratio=True,
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["tf_first_layer"] == 1) &
                                        (df["target_distribution"] == "uni")],
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","layer2_repressors"],
                                 plot_db.rms_patterning_error,
                                 ax=[axd["Ca"]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,#linewidth=2,markersize=10,
                                 take_ratio=True,ylabel="GEE fold-reduction\nA+R / A",logyax=True,
                                 force_color=True,color=plot_db.color_dict["free DNA"],
                                 markers=["D"],reverse_ratio=True,
                                 varnames_dict=varnames_dict)
        axd["C"].set_ylim([pow(10,0),pow(10,1.5)])
        axd["Ca"].set_ylim([pow(10,-0.25),10])
        axd["Ca"].plot([1e2,1e4],[1,1],linewidth=1,color="black",linestyle="solid",zorder=0)
        #labels=["f.D./c. (a.o.)","f.D./c. (w.r.)","a.o./w.r. (c.)","a.o/w.r. (f.D.)"]
        #axd["C"].legend(labels=labels,handlelength=1,ncol=2,columnspacing=0.8,
                        #fontsize=round(plot_db.LEG_FONT_RATIO*fntsz),
                        #bbox_to_anchor=(-0.15,1.02,1,0.1),loc=3)


        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["layer2_repressors"] == 0) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range_fold_change,
                                 ax=[axd["F"]],legloc="lower right",
                                 subtitles=[""],fontsize=fntsz,suppress_leg=True,
                                 ylabel="dynamic range",
                                 linestyle="dashed",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df.loc[(df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["MIN_EXPRESSION"] < 0.01) &
                                        (df["target_distribution"] == "uni")],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.effective_dynamic_range_fold_change,
                                 ax=[axd["F"]],legloc="lower right",
                                 subtitles=[""],fontsize=fntsz,suppress_leg=True,
                                 ylabel="dynamic range\n(fold-change)",
                                 varnames_dict=varnames_dict)
        axd["F"].set_yscale("log")
        axd["F"].set_yticks([1,10,100])
        axd["F"].yaxis.set_minor_formatter(ticker.NullFormatter())

        axd["F"].text(550,24,"A+R",va="top",ha="right",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        axd["F"].text(2000,6.5,"A",va="top",ha="right",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz))
        """
        legend_elements = [Line2D([0],[0],color=plot_db.color_dict["activators only"],linestyle="dashed",
                                  label="A",linewidth=3),
                           Line2D([0],[0],color=plot_db.color_dict["with repressors"],linestyle="solid",
                                  label="A+R",linewidth=3),
                           Line2D([0],[0],marker='o',ls="none",
                                    color=plot_db.to_grayscale(plot_db.color_dict["chromatin"]),
                                    label="c."),
                           Line2D([0],[0],marker='D',ls="none",
                                    color=plot_db.to_grayscale(plot_db.color_dict["free DNA"]),
                                    label="f.D.")]
        customleg = axd["F"].legend(handles=legend_elements,handlelength=1,ncol=2,
                                    fontsize=round(plot_db.LEG_FONT_RATIO*fntsz),markerscale=2,
                                    bbox_to_anchor=(-0.15,1.02,1,0.1),loc=3)
        """

        df_A_dist = df.loc[(df["M_GENE"] == m_gene) &
                         (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                         (df["minimize_noncognate_binding"] == 0) &
                         (df["target_independent_of_clusters"] == 0) &
                         (df["ignore_off_during_optimization"] == 0) &
                         (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                         (df["layer2_repressors"] == 0) &
                         (df["MIN_EXPRESSION"] < 0.01)]
        df_AR_dist = df.loc[(df["M_GENE"] == m_gene) &
                         (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                         (df["minimize_noncognate_binding"] == 0) &
                         (df["target_independent_of_clusters"] == 0) &
                         (df["ignore_off_during_optimization"] == 0) &
                         (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                         (df["layer2_repressors"] == 1) &
                         (df["MIN_EXPRESSION"] < 0.01)]
        plot_db.subplots_groupby(df_A_dist,
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.expression_distribution_groupby,
                                 ["M_GENE"],
                                 ax=[axd["I"],axd["H"],axd["G"]],fontsize=fntsz,
                                 subtitles=["","",""],
                                 varnames_dict=varnames_dict)
        axd["H"].set_ylabel("")
        axd["I"].set_ylabel("")
        plot_db.subplots_groupby(df_AR_dist,
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.expression_distribution_groupby,
                                 ["M_GENE"],
                                 ax=[axd["M"],axd["L"],axd["K"]],fontsize=fntsz,
                                 subtitles=["","",""],
                                 varnames_dict=varnames_dict)
        axd["L"].set_ylabel("")
        axd["M"].set_ylabel("")
        """
        ax_inset_g = axd["G"].inset_axes((0.03,0.4,biginsetsz,biginsetsz))
        ax_inset_h = axd["H"].inset_axes((0.5,0.4,biginsetsz,biginsetsz))
        plot_db.subplots_groupby(df_AR_dist.loc[(df_AR_dist["tf_first_layer"] == True) &
                                             (df_AR_dist["target_distribution"] != "uni")],
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["tf_first_layer"],
                                 ax=[ax_inset_g,ax_inset_h],fontsize=fntsz,
                                 mastercolor=plot_db.color_dict["free DNA"],
                                 colorbar_leg = False,subtitles=["",""],
                                 suppress_leg=True,
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_AR_dist.loc[(df_AR_dist["tf_first_layer"] == False) &
                                             (df_AR_dist["target_distribution"] != "uni")],
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["tf_first_layer"],
                                 mastercolor=plot_db.color_dict["chromatin"],
                                 colorbar_leg = False,subtitles=["",""],
                                 suppress_leg=True,
                                 ax=[ax_inset_g,ax_inset_h],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        adjust_inset(ax_inset_g,add_box=False)
        adjust_inset(ax_inset_h,add_box=False)
        """
        axd["G"].get_legend().remove()
        axd["H"].get_legend().remove()
        axd["I"].get_legend().remove()
        axd["K"].get_legend().remove()
        axd["L"].get_legend().remove()
        axd["M"].get_legend().remove()

        axd["G"].set_xlabel("")
        axd["H"].set_xlabel("target expression")
        axd["I"].set_xlabel("")
        axd["K"].set_xlabel("")
        axd["L"].set_xlabel("target expression")
        axd["M"].set_xlabel("")

        def label_axis(ax,text):
            ax.text(0.1,0.9,text,va="top",ha="left",fontsize=round(plot_db.LEG_FONT_RATIO*fntsz),transform=ax.transAxes)
        label_axis(axd["G"],"A")
        label_axis(axd["H"],"A")
        label_axis(axd["I"],"A")
        label_axis(axd["K"],"A+R")
        label_axis(axd["L"],"A+R")
        label_axis(axd["M"],"A+R")

        axd["G"].annotate("target",xy=(0.895,0.04),xytext=(0.7,0.4),xycoords="axes fraction",ha="center",
                          arrowprops=dict(arrowstyle="-",linewidth=1.5,mutation_scale=40,edgecolor="k"))

        """
        plot_db.subplots_groupby(df_A_dist,
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["target_distribution","tf_first_layer"],
                                 ax=[axd["J"]],fontsize=fntsz,
                                 subtitles=[""],axlabel=" ",
                                 ylabel="GEE",
                                 colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        axd["J"].set_yticks([0,0.04,0.08])

        plot_db.subplots_groupby(df_AR_dist,
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["target_distribution","tf_first_layer"],
                                 ax=[axd["N"]],fontsize=fntsz,
                                 subtitles=[""],axlabel=" ",
                                 ylabel="GEE",
                                 colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        axd["N"].set_yticks([0,0.04,0.08])
        """

        plt.gcf().text(0.055,0.949,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.630,0.956,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.055,0.590,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.630,0.558,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.055,0.212,"E",fontsize=fntsz,fontweight="bold")

        plt.savefig("../plots/fig/fig4.png")
        plt.close()


    # ----- FIGURE 5 ----- #
    if GEN_FIGURE_5:
        def fluctuation_plot_fn(x):
            fluctuation_all = plot_db.get_mean_fluctuation_rmse(x)
            return np.divide(fluctuation_all,x["actual_patterning_error"])

        fig = plt.figure(figsize=(30,18),layout="tight")

        outer = gs.GridSpec(2,1,height_ratios=[1.4,1])
        top = gs.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[0],width_ratios=[1,0.8,0.8])
        #middle = gs.GridSpecFromSubplotSpec(2,1,subplot_spec=top[1],height_ratios=[1,0.8],hspace=0.05)
        bottom = gs.GridSpecFromSubplotSpec(1,4,subplot_spec=outer[1],wspace=0.3)

        S_xticks = [1e2,1e3,1e4]
        repressor_markerdict = {0:"o",1:"v"}

        axd = {"B":plt.subplot(top[0]),
               "A":plt.subplot(top[1]),
               #"B":plt.subplot(middle[0]),
               #"C":plt.subplot(middle[1]),
               "D":plt.subplot(top[2]),
               "E":plt.subplot(bottom[0]),
               "F":plt.subplot(bottom[1]),
               "G":plt.subplot(bottom[2]),
               "H":plt.subplot(bottom[3])}

        fluc_colors = [0.8*np.ones((1,3)),0.5*np.ones((1,3)),np.zeros((1,3))]
        def plot_fold_reduction_fluctuation(df,ax,color):
            plot_db.subplots_groupby(df.loc[(df["layer2_repressors"] == 0) &
                                            (df["target_distribution"] == "uni")],
                                     "M_GENE",
                                     [],[],
                                     plot_db.symbolscatter_groupby,
                                     ["ratio_KNS_KS","tf_first_layer"],
                                     plot_db.get_mean_fluctuation_rmse,
                                     ax=ax,suppress_leg=True,color=color,
                                     subtitles=[""],fontsize=fntsz,
                                     take_ratio=True,ylabel="fold-reduction",logyax=True,
                                     varnames_dict=varnames_dict)
            plot_db.subplots_groupby(df.loc[(df["layer2_repressors"] == 1) &
                                            (df["target_distribution"] == "uni")],
                                     "M_GENE",
                                     [],[],
                                     plot_db.symbolscatter_groupby,
                                     ["ratio_KNS_KS","tf_first_layer"],
                                     plot_db.get_mean_fluctuation_rmse,
                                     ax=ax,suppress_leg=True,color=color,
                                     subtitles=[""],fontsize=fntsz,
                                     take_ratio=True,ylabel="fold-reduction",logyax=True,
                                     markers=repressor_markerdict[1],
                                     varnames_dict=varnames_dict)
        plot_fold_reduction_fluctuation(df_filter_0p05,[axd["A"]],fluc_colors[0])
        plot_fold_reduction_fluctuation(df_filter_0p1,[axd["A"]],fluc_colors[1])
        plot_fold_reduction_fluctuation(df_filter_0p2,[axd["A"]],fluc_colors[2])
        axd["A"].set_ylabel("fold-reduction in GEE")
        axd["A"].set_ylim(0.5,15)
        axd["A"].set_yticks([1,10])
        axd["A"].set_xticks(S_xticks)
        # legend
        handles = [Line2D([0],[0],color=x,linewidth=2) for x in fluc_colors]
        axd["A"].legend(handles,["$\sigma$ = 0.05","$\sigma$ = 0.1","$\sigma$ = 0.2"],
                        frameon=False,handlelength=1,loc="lower left")

        plot_db.subplots_groupby(df_filter_0p1,
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","layer2_repressors","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 lambda x: x["actual_patterning_error"],
                                 ax=[axd["B"]],fontsize=fntsz,draw_lines=True,
                                 markeralpha=1,markerdict=repressor_markerdict,
                                 darken_color=True,
                                 size_lims=[500,500],suppress_leg=True,
                                 subtitles=[""],ylabel="GEE",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_filter_0p1,
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","layer2_repressors","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 plot_db.get_mean_fluctuation_rmse,
                                 ax=[axd["B"]],fontsize=fntsz,draw_lines=True,
                                 markeralpha=1,markerdict=repressor_markerdict,
                                 size_lims=[500,500],#suppress_leg=True,
                                 subtitles=[""],ylabel="GEE",
                                 varnames_dict=varnames_dict)
        axd["B"].set_yscale("log")
        axd["B"].set_xticks(S_xticks)
        plt.setp(axd["B"].get_xticklabels(),visible=False)
        axd["B"].set_xlabel("")

        plot_db.subplots_groupby(df_filter_0p1.loc[(df_filter_0p1["layer2_repressors"] == 0) &
                                               (df_filter_0p1["tf_first_layer"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=[""],gray_cb=True,
                                 fontsize=fntsz,ax=[axd["D"]],
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_filter_0p1.loc[(df_filter_0p1["layer2_repressors"] == 0) &
                                               (df_filter_0p1["tf_first_layer"] == 1)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=[""],
                                 suppress_leg=True,colorbar_leg=False,
                                 fontsize=fntsz,ax=[axd["D"]],
                                 varnames_dict=varnames_dict)
        axd["D"].set_xticks([1e-1,1e-2])
        ax_inset = axd["D"].inset_axes((0.55,0.08,insetsz,insetsz))
        plot_db.subplots_groupby(df_filter_0p1.loc[(df_filter_0p1["layer2_repressors"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],normalize=True,
                                 subtitles=["",""],
                                 fontsize=insetfntsz,ax=[ax_inset,ax_inset],
                                 suppress_leg=True,colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        ax_inset.set_xticks([0.9,1])
        ax_inset.set_yticks([0,0.5,1])
        ax_inset.set_xlabel("")
        ax_inset.set_ylabel("")
        ax_inset.set_title("normalized\nGEE")


        plot_db.subplots_groupby(df_filter_0p1.loc[(df["layer2_repressors"] == 0) &
                                                    (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_pr_on_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=["","free DNA"],
                                 fontsize=fntsz,ax=[axd["E"],axd["F"]],
                                 colorbar_leg=False,
                                 suppress_leg=True,
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_filter_0p1.loc[(df["layer2_repressors"] == 0) &
                                                    (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_pr_on_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 fontsize=fntsz,ax=[axd["G"],axd["H"]],colorbar_leg=False,
                                 suppress_leg=True,factors="tf",
                                 varnames_dict=varnames_dict)


        plt.gcf().text(0.014,0.965,"A",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.375,0.965,"B",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.685,0.965,"C",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.010,0.430,"D",fontsize=fntsz,fontweight="bold")
        plt.gcf().text(0.506,0.418,"E",fontsize=fntsz,fontweight="bold")

        plt.savefig(f"../plots/fig/fig5.png")
        plt.close()


    # ----- FIGURE 5 (FORMER) ----- #
    if GEN_FIGURE_5_FORMER:
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

        plt.savefig("../plots/fig/fig5_former.png")
        plt.close()



    if GEN_SUPPLEMENTAL:
        fig, ax = plt.subplots(2,2,figsize=(20,20),layout="tight")
        fig.delaxes(ax[1][0])
        fig.delaxes(ax[1][1])

        plot_db.subplots_groupby(df_filter_0p1.loc[(df["layer2_repressors"] == 0) &
                                                    (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_pr_on_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=["",""],
                                 fontsize=fntsz,ax=ax,colorbar_leg=False,
                                 suppress_leg=True,factors="tf",
                                 varnames_dict=varnames_dict)

        plt.savefig(f"../plots/fig/supp_fluctuation.png")
        plt.close()


    if GEN_NOISE:
        sigma = 0.1
        df_filter = pandas.read_parquet(f"../fluctuation_res_sigma{sigma}.pq")

        fig, ax = plt.subplots(2,3,figsize=(45,30),layout="tight")
        plot_db.subplots_groupby(df_filter,
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","layer2_repressors","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 lambda x: x["actual_patterning_error"],
                                 ax=[ax[0][0]],fontsize=fntsz,draw_lines=True,markeralpha=1,
                                 force_color=True,
                                 size_lims=[500,500],
                                 subtitles=[""],ylabel="GEE",
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_filter,
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","layer2_repressors","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 plot_db.get_mean_fluctuation,
                                 ax=[ax[0][0]],fontsize=fntsz,draw_lines=True,markeralpha=1,
                                 size_lims=[500,500],
                                 subtitles=[""],ylabel="GEE",
                                 varnames_dict=varnames_dict)
        ax[0][0].set_yscale("log")

        def fluctuation_plot_fn(x):
            fluctuation_all = plot_db.get_mean_fluctuation(x)
            return np.divide(fluctuation_all,x["actual_patterning_error"])

        plot_db.subplots_groupby(df_filter,
                                 ["M_GENE"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","layer2_repressors","MAX_CLUSTERS_ACTIVE","ratio_KNS_KS"],
                                 fluctuation_plot_fn,
                                 ax=[ax[0][1]],fontsize=fntsz,draw_lines=True,markeralpha=1,
                                 size_lims=[500,500],legloc="upper left",
                                 subtitles=[""],ylabel="fold-change in GEE",
                                 varnames_dict=varnames_dict)
        ax[0][1].set_yscale("log")

        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.get_mean_fluctuation,
                                 ax=[ax[1][0]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,
                                 take_ratio=True,ylabel="fold-reduction",logyax=True,
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 1)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer"],
                                 plot_db.get_mean_fluctuation,
                                 ax=[ax[1][0]],suppress_leg=True,
                                 subtitles=[""],fontsize=fntsz,
                                 take_ratio=True,ylabel="fold-reduction",logyax=True,
                                 markers=["P"],
                                 varnames_dict=varnames_dict)
        ax[1][0].set_ylim(-1,1)

        #--OVERLAY--#
        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0) &
                                               (df_filter["tf_first_layer"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=[""],gray_cb=True,
                                 fontsize=fntsz,ax=[ax[0][2]],
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0) &
                                               (df_filter["tf_first_layer"] == 1)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=[""],
                                 suppress_leg=True,colorbar_leg=False,
                                 fontsize=fntsz,ax=[ax[0][2]],
                                 varnames_dict=varnames_dict)
        ax_inset = ax[0][2].inset_axes((0.61,0.09,0.9*insetsz,0.9*insetsz))
        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],normalize=True,
                                 subtitles=["",""],
                                 fontsize=insetfntsz,ax=[ax_inset,ax_inset],
                                 suppress_leg=True,colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        ax_inset.set_xticks([0.9,1])
        ax_inset.set_yticks([0,0.5,1])
        ax_inset.set_xlabel("single-target")
        ax_inset.set_ylabel("multi-target")

        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 fontsize=fntsz,ax=[ax[1][1],ax[1][2]],
                                 varnames_dict=varnames_dict)

        ax1_inset = ax[1][1].inset_axes((0.61,0.09,0.9*insetsz,0.9*insetsz))
        ax2_inset = ax[1][2].inset_axes((0.61,0.09,0.9*insetsz,0.9*insetsz))
        plot_db.subplots_groupby(df_filter.loc[(df_filter["layer2_repressors"] == 0)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_fluctuation_groupby,
                                 ["ratio_KNS_KS"],normalize=True,
                                 subtitles=["",""],
                                 fontsize=insetfntsz,ax=[ax1_inset,ax2_inset],
                                 suppress_leg=True,colorbar_leg=False,
                                 varnames_dict=varnames_dict)
        ax1_inset.set_xticks([0.9,1])
        ax1_inset.set_yticks([0,0.5,1])
        ax2_inset.set_xticks([0.9,1])
        ax2_inset.set_yticks([0,0.5,1])

        ax1_inset.set_xlabel("single-target")
        ax1_inset.set_ylabel("multi-target")
        ax2_inset.set_xlabel("single-target")
        ax2_inset.set_ylabel("multi-target")

        plt.savefig(f"../plots/fig/test_sigma{sigma}.png")
        plt.close()

    if GEN_DIST_TEST:
        df_test = df.loc[(df["M_GENE"] == m_gene) &
                         (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                         (df["minimize_noncognate_binding"] == 0) &
                         (df["target_independent_of_clusters"] == 0) &
                         (df["ignore_off_during_optimization"] == 0) &
                         (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                         (df["layer2_repressors"] == 1) &
                         (df["MIN_EXPRESSION"] < 0.01)]
        fig, ax = plt.subplots(3,3,figsize=(30,30))

        print(len(df_test.loc[(df_test["target_distribution"] == "uni")]))
        print(len(df_test.loc[(df_test["target_distribution"] == "loguni")]))
        print(len(df_test.loc[(df_test["target_distribution"] == "invloguni")]))

        plot_db.subplots_groupby(df_test,
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.expression_distribution_groupby,
                                 ["M_GENE"],
                                 ax=ax[0,:],fontsize=fntsz,
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_test.loc[df_test["tf_first_layer"] == True],
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["tf_first_layer"],
                                 ax=ax[1,:],fontsize=fntsz,
                                 mastercolor=plot_db.color_dict["free DNA"],
                                 colorbar_leg = False,
                                 varnames_dict=varnames_dict)
        plot_db.subplots_groupby(df_test.loc[df_test["tf_first_layer"] == False],
                                 ["target_distribution"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["tf_first_layer"],
                                 mastercolor=plot_db.color_dict["chromatin"],
                                 colorbar_leg = False,
                                 ax=ax[1,:],fontsize=fntsz,
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df_test,
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["target_distribution","tf_first_layer"],
                                 ax=ax[2,:],fontsize=fntsz,
                                 subtitles=[""],axlabel=" ",
                                 colorbar_leg=False,
                                 varnames_dict=varnames_dict)

        plt.savefig(f"../plots/fig/test_distributions.png")
        plt.close()

    if GEN_TEST:
        fig, ax = plt.subplots(2,2,figsize=(20,26),layout="tight")
        fig.delaxes(ax[1][0])
        fig.delaxes(ax[1][1])

        plot_db.subplots_groupby(df_filter_0p1.loc[(df["layer2_repressors"] == 0) &
                                                    (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_pr_on_fluctuation_groupby,
                                 ["ratio_KNS_KS"],
                                 subtitles=["",""],
                                 fontsize=fntsz,ax=ax,colorbar_leg=False,
                                 suppress_leg=True,factors="tf",
                                 varnames_dict=varnames_dict)

        plt.savefig(f"../plots/fig/test_fluctuation_scatter.png")
        plt.close()
        

if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
