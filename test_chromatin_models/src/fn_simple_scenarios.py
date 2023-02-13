#!/usr/bin/env python
# coding: utf-8

import sys, argparse, os, shutil
import numpy as np
from scipy import optimize
import math

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

import shelve
import dill

from params import *
from ss_params import *


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
    parser.add_argument("-s","--scenarios",type=int,nargs='+',default=range(1,8))
    parser.add_argument("-o","--optimize",dest="optimize_xtalk",action="store_true",default=False)
    parser.add_argument("-x","--crosslayer_crosstalk",dest="crosslayer_crosstalk",action="store_true",default=False)

    args = parser.parse_args()
    folder = args.folder
    first_layer_file = args.first_layer_file
    second_layer_file = args.second_layer_file
    prefix = args.prefix
    scenarios = args.scenarios
    optimize_xtalk = args.optimize_xtalk
    crosslayer_crosstalk = args.crosslayer_crosstalk

    mplstyle.use('fast')

    # check file structure
    assert os.path.exists(folder), f"folder {folder} does not exist"

    # copy parameter file into output folder
    param_filepath = os.path.join(folder,f"ss_params.py")
    if not(os.path.exists(param_filepath)):
        print(f"Copying ss_params.py into {folder}...")
        shutil.copy("src/ss_params.py",param_filepath)

    print(f"Analyzing crosstalk objective function for {prefix}...")


    # load functions describing probability open/on for each of the regulatory layers
    pr_layer1 = dill.load(open(os.path.join(folder,first_layer_file), "rb"))
    pr_layer2 = dill.load(open(os.path.join(folder,second_layer_file), "rb"))
    

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
        entries_on, coeffs, pr, labs = xtalk_on_off_exp(num)(C[0],C[1])
        fig, ax = plt.subplots(figsize = (24,24))
        plt.bar(labs,pr)
        plt.ylim([0,1.1])
        ax.set_xlabel("class of gene")
        ax.set_ylabel("probability expressing")
        ax.set_title(f"Cl1 = {C[0]:.2f}, Cl2 = {C[1]:.2f}")
        for ii in range(len(labs)):
            plt.text(ii,pr[ii]+txtoffset,f"{pr[ii]:.3f}",ha="center")
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
                    return entries_on, coeffs, (pr_layer1(C,cl1)*pr_layer2(C,cl2),pr_layer1(C,0)*pr_layer2(C,0)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,cl1)*pr_layer2(Cl2,cl2),pr_layer1(Cl1,0)*pr_layer2(Cl2,0)), labs
        elif num == 2:          # N_ON genes ON that share same PF, N_OFF genes OFF, unique TF for each gene
            def fn(Cl1,Cl2):
                cl2 = Cl2/N_ON
                entries_on = [1,0]
                coeffs = [N_ON,N_OFF]
                labs = [f"ON ({N_ON})",f"OFF ({M-N_ON})"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1)*pr_layer2(C,cl2),pr_layer1(C,0)*pr_layer2(C,0)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,cl2),pr_layer1(Cl1,0)*pr_layer2(Cl2,0)), labs
        elif num == 3:          # N_ON genes ON that share same PF, N_OFF_shared genes OFF that also share this PF
            def fn(Cl1,Cl2):
                cl2 = Cl2/N_ON
                entries_on = [1,0,0]
                coeffs = [N_ON, N_OFF - N_OFF_shared, N_OFF_shared]
                labs = [f"ON ({N_ON})",f"OFF ({M-N_ON-N_OFF_shared})",f"OFF, shared PF w/ ON ({N_OFF_shared})"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1)*pr_layer2(C,cl2),pr_layer1(C,0)*pr_layer2(C,0),pr_layer1(C,Cl1)*pr_layer2(C,0)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,cl2),pr_layer1(Cl1,0)*pr_layer2(Cl2,0),pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,0)), labs
        elif num == 4:          # 4-gene uniquely addressed network, one gene ON
            def fn(Cl1,Cl2):
                entries_on = [1,0,0,0]
                coeffs = [1,1,1,1]
                labs = ["ON","OFF, shared PF w/ ON","OFF, shared TF w/ ON","OFF, no overlap w/ ON"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1)*pr_layer2(C,Cl2),pr_layer1(C,Cl1)*pr_layer2(C,0),pr_layer1(C,0)*pr_layer2(C,Cl2),pr_layer1(C,0)*pr_layer2(C,0)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,Cl2),pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,0),pr_layer1(Cl1,0)*pr_layer2(Cl2,Cl2),pr_layer1(Cl1,0)*pr_layer2(Cl2,0)), labs
        elif num == 5:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that share PF
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, shared PF","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1)*pr_layer2(C,Cl2/2),pr_layer1(C,0)*pr_layer2(C,Cl2/2)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1)*pr_layer2(Cl2,Cl2/2),pr_layer1(Cl1,0)*pr_layer2(Cl2,Cl2/2)), labs
        elif num == 6:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that share TF
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, shared TF","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1/2)*pr_layer2(C,Cl2),pr_layer1(C,Cl1/2)*pr_layer2(C,0)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1/2)*pr_layer2(Cl2,Cl2),pr_layer1(Cl1,Cl1/2)*pr_layer2(Cl2,0)), labs
        elif num == 7:
            def fn(Cl1,Cl2):    # 4-gene uniquely addressed network, 2 genes ON that do not share PF or TF (not an achievable pattern)
                entries_on = [1,0]
                coeffs = [2,2]
                labs = ["ON, no overlap","OFF"]
                if crosslayer_crosstalk:
                    C = Cl1+Cl2
                    return entries_on, coeffs, (pr_layer1(C,Cl1/2)*pr_layer2(C,Cl2/2),pr_layer1(C,Cl1/2)*pr_layer2(C,Cl2/2)), labs
                else:
                    return entries_on, coeffs, (pr_layer1(Cl1,Cl1/2)*pr_layer2(Cl2,Cl2/2),pr_layer1(Cl1,Cl1/2)*pr_layer2(Cl2,Cl2/2)), labs
        else:
            printf(f"no scenario number {num}")
            sys.exit(2)

        return fn

    # wrapper function for calculating mean square error for xtalk
    def mse_wrapper(entry_on,coeff,pr):
        return coeff*(entry_on - np.array(pr))**2

    # objective function for crosstalk for scenario num at concentrations Cl1, Cl2
    def xtalk_objective_fn_scenario(num,Cl1,Cl2):
        entries_on, coeffs, pr, labs = xtalk_on_off_exp(num)(Cl1,Cl2)
        return sum(list(map(mse_wrapper,entries_on,coeffs,pr)))


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
            optres = optimize.minimize(objfun,C0,tol=eps,bounds=[(0,np.inf)]*2)
            optimal_C = optres.x
            optimal_xtalk = optres.fun
            gen_bar_expression_level(sc,optimal_C,f"{prefix}_expression_at_min_xtalk_scenario{sc}")
        else:
            optimal_C = [None,None]
            optimal_xtalk = None

        gen_contourf(xtalk_scenario,f"{prefix}_xtalk_scenario{sc}_contour",xtalk_vmax,optimal_C,optimal_xtalk)
        # gen_heatmap(xtalk_scenario,"xtalk_scenario1_heatmap")
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
