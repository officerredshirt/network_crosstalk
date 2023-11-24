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
GEN_FIGURE_4 = False
GEN_FIGURE_5 = True
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
                (df["target_independent_of_clusters"] == False) &
                (df["layer2_repressors"] == False)]

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
                                 fontsize=fntsz,ylabel="global expression error",
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
                                        (df["layer2_repressors"] == 0)],
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
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["target_independent_of_clusters","tf_first_layer"],
                                 ax=[axd["G"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        axd["G"].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
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
        fig = plt.figure(figsize=(20,28),layout="tight")

        outer = gs.GridSpec(3,1,height_ratios=[1.3,1,1])
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0],wspace=0.1)
        inner1 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[1],wspace=0.1)
        inner2 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[2],width_ratios=[0.5,1])

        axd = {"A":plt.subplot(inner0[0]),
               "B":plt.subplot(inner0[1]),
               "C":plt.subplot(inner1[0]),
               "D":plt.subplot(inner1[1]),
               "E":plt.subplot(inner2[0]),
               "F":plt.subplot(inner2[1])}


        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["minimize_noncognate_binding"],
                                 ax=[axd["C"],axd["D"]],fontsize=fntsz,
                                 colorbar_leg=False,gray_first_level=True,
                                 varnames_dict=varnames_dict)
        box1 = axd["C"].get_position()
        box2 = axd["D"].get_position()
        #axd["C"].text(1.1,1.2,f"intrinsic specificity = {RATIO_FOR_SINGLE_EXAMPLES}",fontsize=fntsz,
                      #ha="center",va="center")
        axd["D"].set_ylabel("")
        plt.setp(axd["D"].get_yticklabels(),visible=False)

        plot_db.subplots_groupby(df_normal.loc[(df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df_normal["M_GENE"] == m_gene)],
                                 "M_GENE",
                                 [],[],
                                 plot_db.symbolscatter_groupby,
                                 ["ratio_KNS_KS","tf_first_layer","minimize_noncognate_binding"],
                                 plot_db.rms_patterning_error,
                                 subtitles=[""],
                                 ax=[axd["E"]],fontsize=fntsz,take_ratio=True,
                                 ylabel="fold-reduction in error",
                                 legloc="upper left",
                                 varnames_dict=varnames_dict)
        axd["E"].set_yscale("log")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["minimize_noncognate_binding"] == 0) &
                                        (df_normal["M_GENE"] == m_gene) &
                                        (df_normal["MAX_CLUSTERS_ACTIVE"] == maxclust)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_error_fraction_groupby,
                                 ["ratio_KNS_KS"],
                                 ax=[axd["A"],axd["B"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        plt.setp(axd["B"].get_yticklabels(),visible=False)

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
        """
        N = np.linspace(100,500,100)
        exponent = 1.4
        ax[0][1].plot(N,(0.035/np.power(100,exponent))*np.power(N,exponent),linewidth=5,color="k")
        exponent = 2.6
        ax[0][1].plot(N,(0.005/np.power(100,exponent))*np.power(N,exponent),linewidth=5,color="k")
        """
        ax[0][1].set_yscale("log")

        plot_db.subplots_groupby(df_normal.loc[(df_normal["ratio_KNS_KS"] == 1000) &
                                               (df_normal["minimize_noncognate_binding"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.colorscatter_2d_groupby,
                                 ["tf_first_layer","minimize_noncognate_binding","MAX_CLUSTERS_ACTIVE","M_GENE"],
                                 lambda x: plot_db.patterning_error(x),
                                 ax=[ax[0][2]],fontsize=fntsz,
                                 suppress_leg=True,draw_lines=True,
                                 ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        ax[0][2].set_ylabel("M*(global expression error)^2")
        ax[0][2].set_xscale("linear")

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


    # ----- FIGURE 5 ----- #
    if GEN_FIGURE_5:
        fig = plt.figure(figsize=(28,20),layout="tight")

        outer = gs.GridSpec(2,1)
        inner0 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[0],width_ratios=[2,1])
        scattertarget = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = inner0[0],wspace=0.1)
        inner1 = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = outer[1],width_ratios=[2,1])
        scatterreg = gs.GridSpecFromSubplotSpec(1,2,subplot_spec = inner1[0],wspace=0.1)

        axd = {"A":plt.subplot(scattertarget[0]),
               "B":plt.subplot(scattertarget[1]),
               "C":plt.subplot(inner0[1]),
               "D":plt.subplot(scatterreg[0]),
               "E":plt.subplot(scatterreg[1]),
               "F":plt.subplot(inner1[1])}


        RATIO_FOR_SINGLE_EXAMPLES = 1000
        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 ["tf_first_layer"],
                                 [],[],
                                 plot_db.scatter_target_expression_groupby,
                                 ["layer2_repressors"],
                                 ax=[axd["A"],axd["B"]],fontsize=fntsz,
                                 colorbar_leg=False,gray_first_level=True,
                                 varnames_dict=varnames_dict)
        axd["B"].set_ylabel("")
        plt.setp(axd["B"].get_yticklabels(),visible=False)

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
                                        ((df["ratio_KNS_KS"] == 1000) | (df["ratio_KNS_KS"] == 500) |
                                        (df["ratio_KNS_KS"] == 200)) &
                                        (df["layer2_repressors"] == 1)],
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

        plot_db.subplots_groupby(df.loc[(df["M_GENE"] == m_gene) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["target_independent_of_clusters"] == 0) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["layer2_repressors"] == 1) &
                                        (df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES)],
                                 "tf_first_layer",
                                 [],[],
                                 plot_db.scatter_repressor_activator,
                                 ["ratio_KNS_KS"],
                                 subtitles=["",""],
                                 ax=[axd["D"],axd["E"]],fontsize=fntsz,
                                 varnames_dict=varnames_dict)
        axd["E"].set_ylabel("")
        plt.setp(axd["E"].get_yticklabels(),visible=False)
        axd["D"].set_ylim([0,300])
        axd["E"].set_ylim([0,300])

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) &
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_barchart_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["C"]],axlabel=" ",
                                 legloc="upper left",subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)

        plot_db.subplots_groupby(df.loc[(df["ratio_KNS_KS"] == RATIO_FOR_SINGLE_EXAMPLES) & 
                                        (df["minimize_noncognate_binding"] == 0) &
                                        (df["MAX_CLUSTERS_ACTIVE"] == maxclust) &
                                        (df["M_GENE"] == m_gene) &
                                        (df["ignore_off_during_optimization"] == 0) &
                                        (df["target_independent_of_clusters"] == 0)],
                                 ["ratio_KNS_KS"],
                                 [],[],
                                 plot_db.rms_scatter_groupby,
                                 ["layer2_repressors","tf_first_layer"],
                                 ax=[axd["C"]],
                                 legloc="upper left",#subtitles=[""],
                                 fontsize=fntsz,ylabel="global expression error",
                                 varnames_dict=varnames_dict)
        axd["C"].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        #axd["C"].text(0.3,0.56,f"intrinsic\nspecificity\n= {RATIO_FOR_SINGLE_EXAMPLES}",
                      #transform=axd["C"].transAxes,va="center",ha="center")

        plt.savefig("../plots/fig/fig5.png")
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
