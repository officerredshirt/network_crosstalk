#!/usr/bin/env python
# coding: utf-8

from numpy import *
from scipy import optimize
from math import *
from chromatin_params import *
import sympy as sym
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import matplotlib.style as mplstyle
mplstyle.use('fast')
import shelve
import dill


load_from_saved = False

## CHROMATIN PROOFREADING ##
pr_open = dill.load(open("kpr_pr_open.out", "rb"))

## TF BINDING ##
K_S = kh_Sm/kh_Sp
K_NS = 100#kh_NSm/kh_NSp
n = 1

# assume single cluster with single site (to match single PF site for chromatin)
def pr_bound(C,c_s):
    return (1 - 1/(c_s/K_S + (C-c_s)/K_NS + 1))


def pr_on_symmetric(Cpf,cpf,Ctf,ctf):
    return pr_open(Cpf,cpf)*pr_bound(Ctf,ctf)


# "pattern" specification
M = 100             # total number genes
N_ON = 10           # number ON genes
N_OFF = M - N_ON    # number OFF genes

# parameters for xtalk heatmap plots
npts = 500          # number sample points
Cpf_set, Ctf_set = meshgrid(logspace(-1,6,npts), logspace(-1,6,npts), indexing='xy')

def gen_heatmap(xtalk,filename):
    fig, ax = plt.subplots(figsize = (15,12))
    plt.pcolor(Cpf_set,Ctf_set,xtalk,cmap="binary")
    plt.colorbar()
    ax.set_xlabel("C_PF")
    ax.set_ylabel("C_TF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.savefig(f"{filename}.png")

def gen_contourf(xtalk,filename):
    lvs = [0,5,10,15,30,45,60,90]
    fig, ax = plt.subplots(figsize = (24,24))
    cp = plt.contourf(Cpf_set,Ctf_set,xtalk,cmap="RdBu",levels=lvs)
    ax.set_xlabel("C_PF")
    ax.set_ylabel("C_TF")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.colorbar()
    plt.savefig(f"{filename}.png")
    
# parameters for optimization
C0 = [10]*2         # all optimizations are over total PF and TF concentrations (in that order)
eps = 1e-6          # tolerance
bnds = [(0,inf)]*2  # force concentrations positive


## SCENARIO 1 ##
def xtalk_objective_fn_scenario1(Cpf,Ctf):
    cpf = Cpf/N_ON
    ctf = Ctf/N_ON
    return N_ON*(1 - pr_on_symmetric(Cpf,cpf,Ctf,ctf))**2 + N_OFF*pr_on_symmetric(Cpf,0,Ctf,0)**2

## SCENARIO 2 ##
def xtalk_objective_fn_scenario2(Cpf,Ctf):
    ctf = Ctf/N_ON
    return N_ON*(1 - pr_open(Cpf,Cpf)*pr_bound(Ctf,ctf))**2 + N_OFF*pr_on_symmetric(Cpf,0,Ctf,0)**2

def xtalk_objective_fn_for_opt(C):
    Cpf = C[0]
    Ctf = C[1]
    return xtalk_objective_fn_scenario1(Cpf,Ctf)

## SCENARIO 3 ##
N_OFF_shared = floor(0.2*N_OFF)
def xtalk_objective_fn_scenario3(Cpf,Ctf):
    return xtalk_objective_fn_scenario2(Cpf,Ctf) + N_OFF_shared*(pr_open(Cpf,Cpf)**2 - pr_open(Cpf,0)**2)*pr_bound(Ctf,0)**2


if load_from_saved:
    with shelve.open("ss_xtalk_results.out") as ms:
        for key in ms:
            globals()[key] = ms[key]
else:
    # xtalk_scenario1 = optimize.minimize(xtalk_objective_fn_for_opt, C0, tol = eps, bounds = bnds)
    # print(f"C_PF = {xtalk_scenario1.x[0]}, C_TF = {xtalk_scenario1.x[1]}")

    xtalk_scenario1 = xtalk_objective_fn_scenario1(Cpf_set,Ctf_set)
    # gen_heatmap(xtalk_scenario1,"xtalk_scenario1_heatmap")
    gen_contourf(xtalk_scenario1,"xtalk_scenario1_contour")

    xtalk_scenario2 = xtalk_objective_fn_scenario2(Cpf_set,Ctf_set)
    # gen_heatmap(xtalk_scenario2,"xtalk_scenario2_heatmap")
    gen_contourf(xtalk_scenario2,"xtalk_scenario2_contour")

    xtalk_scenario3 = xtalk_objective_fn_scenario3(Cpf_set,Ctf_set)
    # gen_heatmap(xtalk_scenario3,"xtalk_scenario3_heatmap")
    gen_contourf(xtalk_scenario3,"xtalk_scenario3_contour")


## CONTOUR PLOT ##
#lvs = [1,4,15,30,60,75]
lvs = [2,4,6,9,12]
fig, ax = plt.subplots(figsize = (24,24))
cp1 = plt.contour(Cpf_set,Ctf_set,xtalk_scenario1,levels=lvs,cmap="RdBu")
plt.clabel(cp1, inline=1, fontsize=24)
cp2 = plt.contour(Cpf_set,Ctf_set,xtalk_scenario2,levels=lvs,linestyles='dashed',cmap="RdBu")
plt.clabel(cp2, inline=1, fontsize=24)
cp3 = plt.contour(Cpf_set,Ctf_set,xtalk_scenario3,levels=lvs,linestyles='dotted',cmap="RdBu")
plt.clabel(cp3, inline=1, fontsize=24)
ax.set_xlabel("C_PF")
ax.set_ylabel("C_TF")
ax.set_xscale("log")
ax.set_yscale("log")

xt1 = plt.Line2D((0,0),(0,0),color='black')
xt2 = plt.Line2D((0,0),(0,0),color='black',linestyle='dashed')
xt3 = plt.Line2D((0,0),(0,0),color='black',linestyle='dotted')
plt.legend([xt1,xt2,xt3],["scenario 1","scenario 2","scenario 3"],loc="lower right")
plt.savefig("contour.png")


"""
# -- SAVE TO FILE -- #
with shelve.open("ss_xtalk_results.out",'n') as ms:
    ms['xtalk_scenario1'] = xtalk_scenario1
    ms['xtalk_scenario2'] = xtalk_scenario2
    ms['xtalk_scenario3'] = xtalk_scenario3
"""
