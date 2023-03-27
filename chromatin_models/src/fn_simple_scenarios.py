#!/usr/bin/env python
# coding: utf-8

import sys, argparse, os, shutil
import numpy as np
from scipy import optimize
import math
import itertools

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

import shelve
import dill

from params import *
from ss_params import *

def prob_expressing(pr):
    return [np.multiply(x,y) for x,y in pr]

# from ali_m on Stack Overflow
def get_contour_verts(cn):
    contours = []

    for cc in cn.collections:
        paths = []
        for pp in cc.get_paths():
            xy = []
            for vv in pp.iter_segments():
                xy.append(vv[0])
            paths.append(np.vstack(xy))
        contours.append(paths)

    return contours

def get_all_contour_verts(cn):
    contour_verts = []
    for cc in cn.collections:
        for pp in cc.get_paths():
            for vv in pp.iter_segments():
                contour_verts.append(vv[0])

    return np.vstack(contour_verts)


# Generates shaded contour plots of crosstalk objective function as a function of
# concentrations of first-layer vs. second-layer regulatory factors for three
# simple scenarios.
def main(argv):
    
    # parse input arguments
    parser = argparse.ArgumentParser(
            prog = "fn_simple_scenarios",
            description = "",
            epilog = "")
    parser.add_argument("folder")
    parser.add_argument("first_layer_file")
    parser.add_argument("second_layer_file")
    parser.add_argument("prefix")
    parser.add_argument("-s","--scenarios",type=int,nargs='+',default=range(1,7)) # ignore scenario 7 for now
    parser.add_argument("-o","--optimize",dest="optimize_xtalk",action="store_true",default=False)
    parser.add_argument("-x","--crosslayer_crosstalk",action="store_true",default=False)
    parser.add_argument("-m","--max_expression_lock",action="store_true",default=False)
    parser.add_argument("-f","--error_rate_first_layer_file")
    parser.add_argument("-e","--error_rate_second_layer_file")

    args = parser.parse_args()
    folder = args.folder
    first_layer_file = args.first_layer_file
    second_layer_file = args.second_layer_file
    prefix = args.prefix
    scenarios = args.scenarios
    optimize_xtalk = args.optimize_xtalk
    crosslayer_crosstalk = args.crosslayer_crosstalk
    max_expression_lock = args.max_expression_lock
    error_rate_first_layer_file = args.error_rate_first_layer_file
    error_rate_second_layer_file = args.error_rate_second_layer_file

    #mplstyle.use('fast')

    # check file structure
    assert os.path.exists(folder), f"folder {folder} does not exist"

    # copy parameter file into output folder
    param_filepath = os.path.join(folder,f"ss_params.py")
    if not(os.path.exists(param_filepath)):
        print(f"Copying ss_params.py into {folder}...")
        shutil.copy("src/ss_params.py",param_filepath)

    print(f"Analyzing crosstalk objective function for {prefix}...")


    # load functions describing probability open/on for each of the regulatory layers
    pr_layer1 = dill.load(open(os.path.join(folder,first_layer_file),"rb"))
    pr_layer2 = dill.load(open(os.path.join(folder,second_layer_file),"rb"))

    if error_rate_first_layer_file is not None:
        err_layer1 = dill.load(open(os.path.join(folder,error_rate_first_layer_file),"rb"))
    else:
        def err_layer1(C,c_S):
            return None

    if error_rate_second_layer_file is not None:
        err_layer2 = dill.load(open(os.path.join(folder,error_rate_second_layer_file),"rb"))
    else:
        def err_layer2(C,c_S):
            return None

    # define maximum expression level
    if max_expression_lock:
        max_expression = 1.0    # float for consistency (if integer 1, get different results for crosstalk calculations)
        print(f"***max expression locked at 1.0")
    else:
        max_concentration = 1e20
        max_expression = pr_layer1(max_concentration,max_concentration)*pr_layer2(max_concentration,max_concentration)
        if max_expression > 1.0:
            print("max expression returned > 0; locking at 1.0")
            max_expression = 1.0
        print(f"***max_expression = {max_expression}")
    

    # generate range of concentrations to plot
    Cl1_set, Cl2_set = np.meshgrid(np.logspace(-1,8,npts), np.logspace(-1,8,npts), indexing='xy')
    

    # functions for generating plots
    def gen_heatmap(xtalk,filename):
        fig, ax = plt.subplots(figsize = (15,12))
        plt.pcolor(Cl1_set,Cl2_set,xtalk,cmap="binary")
        plt.colorbar()
        ax.set_xlabel("layer 1 concentration")
        ax.set_ylabel("layer 2 concentration")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.savefig(os.path.join(folder,f"{filename}.png"))


    
    def gen_contourf(xtalk,filename,xtalk_vmax=1,optimal_C=[None,None],optimal_xtalk=None):
        plt.rcParams.update({'font.size': 36})
        lvs = np.array([0,0.05,0.1,0.15,0.25,0.35,0.55,0.75,1])*xtalk_vmax
        fig, ax = plt.subplots(figsize = (24,24))
        cp = plt.contourf(Cl1_set,Cl2_set,xtalk,cmap="RdBu",levels=lvs)
        plt.colorbar()
        plt.scatter(optimal_C[0],optimal_C[1],marker=".",facecolors="black",s=1000)
        ax.set_xlabel("layer 1 concentration")
        ax.set_ylabel("layer 2 concentration")
        ax.set_xscale("log")
        ax.set_yscale("log")
        title_str = "crosstalk"
        if optimal_xtalk is not None:
            title_str = title_str + f" (minimum = {optimal_xtalk:.5f})"
        if crosslayer_crosstalk:
            title_str = title_str + ", cross-layer allowed"
        ax.set_title(title_str)
        plt.savefig(os.path.join(folder,f"{filename}.png"))

    def gen_bar_expression_level(num,C,filename):
        plt.rcParams.update({'font.size':24})
        txtoffset = 0.01
        entries_on, coeffs, pr, err, labs = xtalk_on_off_exp(num)(C[0],C[1])
        pr_exp = [p/max_expression for p in prob_expressing(pr)]

        fig, ax = plt.subplots(figsize = (24,24))

        if err[0] is not None:
            pr_noncognate = np.array(pr_exp)*np.array(err)
            pr_cognate = np.array(pr_exp)*(1 - np.array(err))
            plt.bar(labs,pr_noncognate,color="r")
            plt.bar(labs,pr_cognate,bottom=pr_noncognate,color="b")
            for ii in range(len(labs)):
                plt.text(ii,pr_exp[ii]+4*txtoffset,f"2nd err rate: {err[ii]:.3f}",ha="center",color="red")
        else:
            plt.bar(labs,pr_exp)

        plt.ylim([0,1.1])
        ax.set_xlabel("class of gene")
        ax.set_ylabel("probability expressing / max expression")
        ax.set_title(f"Cl1 = {C[0]:.2f}, Cl2 = {C[1]:.2f}, max_expression = {max_expression:.5f}")
        for ii in range(len(labs)):
            plt.text(ii,pr_exp[ii]+txtoffset,f"{pr_exp[ii]:.3f}",ha="center")
        plt.savefig(os.path.join(folder,f"{filename}.png"))


    # returns probability on/off for relevant groups of genes
    def xtalk_on_off_exp(num):
        if num == 1:            # N_ON genes ON, N_OFF genes OFF, unique TF and PF for each gene
            def fn(Cl1,Cl2):
                cl1 = Cl1/N_ON
                cl2 = Cl2/N_ON
                entries_on = [1,0]
                coeffs = [N_ON,N_OFF]
                labs = [f"ON ({N_ON})",f"OFF ({M-N_ON})"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,cl1),pr_layer2(C,cl2)),
                                                (pr_layer1(C,0),pr_layer2(C,0))], \
                                                        [err_layer2(C,cl2),err_layer2(C,0)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,cl1),pr_layer2(Cl2,cl2)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,0))], \
                                                        [err_layer2(Cl2,cl2),err_layer2(Cl2,0)], labs
        elif num == 2:          # N_ON genes ON that share same PF, N_OFF genes OFF, unique TF for each gene
            def fn(Cl1,Cl2):
                cl2 = Cl2/N_ON
                entries_on = [1,0]
                coeffs = [N_ON,N_OFF]
                labs = [f"ON ({N_ON})",f"OFF ({M-N_ON})"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1),pr_layer2(C,cl2)),
                                                (pr_layer1(C,0),pr_layer2(C,0))], \
                                                        [err_layer2(C,cl2),err_layer2(C,0)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1),pr_layer2(Cl2,cl2)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,0))], \
                                                        [err_layer2(Cl2,cl2),err_layer2(Cl2,0)], labs
        elif num == 3:          # N_ON genes ON that share same PF, N_OFF_shared genes OFF that also share this PF
            def fn(Cl1,Cl2):
                cl2 = Cl2/N_ON
                entries_on = [1,0,0]
                coeffs = [N_ON, N_OFF - N_OFF_shared, N_OFF_shared]
                labs = [f"ON ({N_ON})",f"OFF ({M-N_ON-N_OFF_shared})",f"OFF, shared PF w/ ON ({N_OFF_shared})"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1),pr_layer2(C,cl2)),
                                                 (pr_layer1(C,0),pr_layer2(C,0)),
                                                 (pr_layer1(C,Cl1),pr_layer2(C,0))], \
                                                         [err_layer2(C,cl2), err_layer2(C,0), err_layer2(C,0)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1),pr_layer2(Cl2,cl2)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,0)),
                                                (pr_layer1(Cl1,Cl1),pr_layer2(Cl2,0))], \
                                                        [err_layer2(Cl2,cl2), err_layer2(Cl2,0), err_layer2(Cl2,0)], labs
        elif num == 4:          # 4-gene uniquely addressed network, one gene ON
            def fn(Cl1,Cl2):
                entries_on = [1,0,0,0]
                coeffs = [1,1,1,1]
                labs = ["ON","OFF, shared PF w/ ON","OFF, shared TF w/ ON","OFF, no overlap w/ ON"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1),pr_layer2(C,Cl2)),
                                                (pr_layer1(C,Cl1),pr_layer2(C,0)),
                                                (pr_layer1(C,0),pr_layer2(C,Cl2)),
                                                (pr_layer1(C,0),pr_layer2(C,0))], \
                                                        [err_layer2(C,Cl2), err_layer2(C,0),
                                                         err_layer2(C,Cl2), err_layer2(C,0)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1),pr_layer2(Cl2,Cl2)),
                                                (pr_layer1(Cl1,Cl1),pr_layer2(Cl2,0)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,Cl2)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,0))], \
                                                        [err_layer2(Cl2,Cl2), err_layer2(Cl2,0),
                                                         err_layer2(Cl2,Cl2), err_layer2(Cl2,0)], labs
        elif num == 5:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that share PF
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, shared PF","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1),pr_layer2(C,Cl2/2)),
                                                (pr_layer1(C,0),pr_layer2(C,Cl2/2))], \
                                                        [err_layer2(C,Cl2/2), err_layer2(C,Cl2/2)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1),pr_layer2(Cl2,Cl2/2)),
                                                (pr_layer1(Cl1,0),pr_layer2(Cl2,Cl2/2))], \
                                                        [err_layer2(Cl2,Cl2/2), err_layer2(Cl2,Cl2/2)], labs
        elif num == 6:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that share TF
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, shared TF","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1/2),pr_layer2(C,Cl2)),
                                                (pr_layer1(C,Cl1/2),pr_layer2(C,0))], \
                                                        [err_layer2(C,Cl2), err_layer2(C,0)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1/2),pr_layer2(Cl2,Cl2)),
                                                (pr_layer1(Cl1,Cl1/2),pr_layer2(Cl2,0))], \
                                                        [err_layer2(Cl2,Cl2), err_layer2(Cl2,0)], labs
        elif num == 7:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that do not share PF or TF (not an achievable pattern)
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, no overlap","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, [(pr_layer1(C,Cl1/2),pr_layer2(C,Cl2/2)),
                                                (pr_layer1(C,Cl1/2),pr_layer2(C,Cl2/2))], \
                                                        [err_layer2(C,Cl2/2), err_layer2(C,Cl2/2)], labs
                else:
                    return entries_on, coeffs, [(pr_layer1(Cl1,Cl1/2),pr_layer2(Cl2,Cl2/2)),
                                                (pr_layer1(Cl1,Cl1/2),pr_layer2(Cl2,Cl2/2))], \
                                                        [err_layer2(Cl2,Cl2/2), err_layer2(Cl2,Cl2/2)], labs
        else:
            print(f"no scenario number {num}")
            sys.exit(2)

        return fn

    # wrapper function for calculating mean square error for xtalk
    def mse_wrapper(entry_on,coeff,pr_exp):
        return coeff*((entry_on - np.array(pr_exp)/max_expression)**2)

    # objective function for crosstalk for scenario num at concentrations Cl1, Cl2
    def xtalk_objective_fn_scenario(num,Cl1,Cl2):
        entries_on, coeffs, pr, err, labs = xtalk_on_off_exp(num)(Cl1,Cl2)
        pr_exp = prob_expressing(pr)
        return sum(list(map(mse_wrapper,entries_on,coeffs,pr_exp)))


    # pick index and expression level for contour, x axis (Cl1 or Cl2), generate plots for all
    # len(labs) gene categories along the contour
    def get_pr_at_contour(num,contour_ix,contour_level,x_axis,optimal_C):
        plt.rcParams.update({'font.size': 24})
        markersz = 500

        entries_on, coeffs, pr_opt, err, labs = xtalk_on_off_exp(num)(optimal_C[0],optimal_C[1])
        pr_expressing_opt = prob_expressing(pr_opt)
        xtalk_opt = xtalk_objective_fn_scenario(num,optimal_C[0],optimal_C[1])

        entries_on, coeffs, pr, err, labs = xtalk_on_off_exp(num)(Cl1_set,Cl2_set)
        pr_exp = prob_expressing(pr)

        try:
            pr_contour_genes = pr_exp[contour_ix]
        except:
            print(f"contour_ix should be between 0 and {len(labs)-1}")

        # trace contour
        cn = plt.contour(Cl1_set,Cl2_set,pr_contour_genes,[contour_level])
        contour_verts = get_all_contour_verts(cn)

        entries_on, coeffs, pr, err, labs = xtalk_on_off_exp(num)(contour_verts[:,0],contour_verts[:,1])
        pr_expressing = prob_expressing(pr)

        if x_axis == "Cl1":
            x_ax_ix = 0
        elif x_axis == "Cl2":
            x_ax_ix = 1
        else:
            print(f"unrecognized x axis variable name {x_axis}")
            sys.exit()

        xtalk = xtalk_objective_fn_scenario(num,contour_verts[:,0],contour_verts[:,1])
        fig, ax = plt.subplots(figsize = (24,24))
        plt.plot(contour_verts[:,x_ax_ix],xtalk)
        plt.plot([optimal_C[x_ax_ix],optimal_C[x_ax_ix]],[0,1e10],label=f"optimal {x_axis}",color="gray",linewidth=2)
        plt.scatter(optimal_C[x_ax_ix],xtalk_opt,marker="o",s=markersz,edgecolors="gray",facecolors="none",label="optimum")
        ax.set_xlabel(x_axis)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(min(1e-20,min(xtalk)),max(xtalk))
        ax.set_ylabel("crosstalk")
        ax.set_title(f"crosstalk along contour pr({labs[contour_ix]}) = {contour_level:.3f}")
        plt.savefig(os.path.join(folder,f"{prefix}_scenario{num}_crosstalk_contourslice_{contour_ix}_{contour_level:.3f}.png"))

        for p in range(len(labs)):
            pr_layer1_at_contour = pr[p][0]
            pr_layer2_at_contour = pr[p][1]

            fig, ax = plt.subplots(figsize = (24,24))
            p1 = plt.scatter(contour_verts[:,x_ax_ix],pr_layer1_at_contour,label="probability layer 1 on")
            p2 = plt.scatter(contour_verts[:,x_ax_ix],pr_layer2_at_contour,label="probability layer 2 on")
            p3 = plt.scatter(contour_verts[:,x_ax_ix],pr_expressing[p],label="probability expressing (p1*p2)")
            plt.plot([optimal_C[x_ax_ix],optimal_C[x_ax_ix]],[0,1],label=f"optimal {x_axis}",color="gray",linewidth=2)

            plt.scatter(optimal_C[x_ax_ix],pr_opt[p][0],marker="o",s=markersz,edgecolors=p1.get_edgecolor(),facecolors="none")
            plt.scatter(optimal_C[x_ax_ix],pr_opt[p][1],marker="o",s=markersz,edgecolors=p2.get_edgecolor(),facecolors="none")
            plt.scatter(optimal_C[x_ax_ix],pr_expressing_opt[p],marker="o",s=markersz,edgecolors=p3.get_edgecolor(),facecolors="none")
            ax.set_xlabel(x_axis)
            ax.set_xscale("log")

            pr_appended = list(itertools.chain(pr_layer1_at_contour,pr_layer2_at_contour,pr_expressing[p]))
            ax.set_ylim(min(pr_appended),max(pr_appended))
            ax.set_ylabel("probability on")
            ax.set_title(f"probability of layers on for {labs[p]} along contour pr({labs[contour_ix]}) = {contour_level:.3f}")
            ax.legend()
            plt.savefig(os.path.join(folder,f"{prefix}_scenario{num}_genes{p}_contourslice_{contour_ix}_{contour_level:.3f}.png"))


    for ii, sc in enumerate(scenarios):
        print(f"  Analyzing scenario {sc}...")
        if sc > 3:
            xtalk_vmax = 4
        else:
            xtalk_vmax = M
        xtalk_scenario = xtalk_objective_fn_scenario(sc,Cl1_set,Cl2_set)

        if optimize_xtalk:
            print(f"    Optimizing crosstalk...")
            def objfun(c):
                return xtalk_objective_fn_scenario(sc,c[0],c[1])
            optres = optimize.minimize(objfun,C0,tol=eps,bounds=[(0,np.inf)]*len(C0))
            optimal_C = optres.x
            optimal_xtalk = optres.fun
            gen_bar_expression_level(sc,optimal_C,f"{prefix}_scenario{sc}_expression_at_min_xtalk")
        else:
            optimal_C = [None,None]
            optimal_xtalk = None

        gen_contourf(xtalk_scenario,f"{prefix}_scenario{sc}_xtalk_contour",xtalk_vmax,optimal_C,optimal_xtalk)

        """
        entries_on, coeffs, pr, err, labs = xtalk_on_off_exp(sc)(optimal_C[0],optimal_C[1])
        pr_exp = prob_expressing(pr)
        get_pr_at_contour(sc,0,pr_exp[0],"Cl1",optimal_C)
        get_pr_at_contour(sc,1,pr_exp[1],"Cl2",optimal_C)
        """

        plt.close("all")
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
