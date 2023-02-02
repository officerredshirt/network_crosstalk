#!/usr/bin/env python
# coding: utf-8

from numpy import *
from math import *
from chromatin_params import *
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import sympy as sym
import matplotlib.pyplot as plt
import shelve
import dill
plt.rcParams.update({'font.size': 20})

kpr_pr_on = dill.load(open("VARPARAM_kpr_pr_open.out", "rb"))
kpr_error_rate = dill.load(open("VARPARAM_kpr_opening_error_rate.out", "rb"))


## TF BINDING ##
K_S = kh_Sm/kh_Sp
n = 1

# assume single cluster with single site (to match single PF site for chromatin)
def pr_gene_on(kh_NSm_var,C,c_s):
    K_NS = kh_NSm_var/kh_NSp
    return (1 - 1/(c_s/K_S + (C-c_s)/K_NS + 1))

def gene_on_error_rate(kh_NSm_var,C,c_s):
    K_NS = kh_NSm_var/kh_NSp
    return ((C-c_s)/K_NS)/((C-c_s)/K_NS + c_s/K_S)


## PARAMETER SCAN ##
c_ns = 1000
c_s_set = logspace(-5,5,100)
C_set = c_ns + c_s_set
frac_induction = 0.9
npts = 100

#kh_NSm_set = logspace(-2,2,20)
#k_neq_set = linspace(0.01,0.5,20)

kh_NSm_set, k_neq_set = meshgrid(logspace(-2,2,npts), linspace(0.01,0.5,npts), indexing='xy')

tf_err = full(kh_NSm_set.shape[0],nan)
kpr_err = full(kh_NSm_set.shape,nan)
for jj in range(npts):
    cur_kh_NSm = kh_NSm_set[0,jj]

    tf_frac_on = pr_gene_on(cur_kh_NSm,C_set,c_s_set)
    tf_error_rates = gene_on_error_rate(cur_kh_NSm,C_set,c_s_set)
    tf_interp = interp1d(tf_error_rates,tf_frac_on)#,fill_value='extrapolate')

    def tf_f(x):
        return tf_interp(x) - frac_induction

    try:
        tf_err[jj] = fsolve(tf_f,0.5)
    except:
        pass

    for ii in range(npts):
        #cur_kh_NSm = kh_NSm_set[ii,jj]
        cur_k_neq = k_neq_set[ii,jj]

        kpr_frac_on = kpr_pr_on(cur_kh_NSm,cur_k_neq,C_set-c_s_set,c_s_set)
        kpr_error_rates = kpr_error_rate(cur_kh_NSm,cur_k_neq,C_set-c_s_set,c_s_set)
        kpr_interp = interp1d(kpr_error_rates,kpr_frac_on)#,fill_value='extrapolate')

        def kpr_f(x):
            return kpr_interp(x) - frac_induction

        try:
            kpr_err[ii,jj] = fsolve(kpr_f,0.5)
        except:
            pass

def gen_heatmap(err,filename):
    fig, ax = plt.subplots(figsize = (15,12))
    plt.pcolor(kh_NSm_set,k_neq_set,err,cmap="coolwarm",vmin=0,vmax=1)
    plt.colorbar()
    ax.set_xlabel("k_NS-")
    ax.set_ylabel("k_neq")
    ax.set_xscale("log")
    plt.savefig(f"{filename}.png")

gen_heatmap(kpr_err,"kpr_err_heatmap")


fig, ax = plt.subplots(figsize = (15,12))
plt.plot(kh_NSm_set[0,:],tf_err)
ax.set_xlabel("k_NS-")
ax.set_ylabel("TF error rate")
ax.set_xscale("log")
ax.set_xlim((min(kh_NSm_set[0,:]),max(kh_NSm_set[0,:])))
plt.savefig("tf_err.png")
