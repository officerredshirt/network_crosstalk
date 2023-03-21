from numpy import *

import shelve
import dill

import matplotlib.pyplot as plt
import sys
import os
import importlib

folder_in = "cluster_2022-11-08"
prefix = "tf-"
suffix = ".xtalk"

sys.path.append(folder_in)
sys.path.append(folder_in + "/src")
sys.path.append(folder_in + "/res")

from boolarr import *
from tf_binding_equilibrium import *

p = importlib.import_module(prefix + "params")
pr_chromatin_open = dill.load(open(folder_in + "/src/chromatin_kpr_pr_open.out","rb"))

files = os.listdir(folder_in + "/res/")
filenames = [fn for fn in files if fn.startswith(prefix) & fn.endswith(suffix)]
filenames = [fn[:-len(suffix)] for fn in filenames]


xtalk_off_by_ngenes = [[] for x in range(p.M_GENE+1)]
xtalk_on_by_ngenes = [[] for x in range(p.M_GENE+1)]

"""
pgtol = zeros(len(filenames))
pgtol_ngenes = []
"""

for ii, cur_filename in enumerate(filenames):
    #print(cur_filename)

    filename_in = folder_in + "/res/" + cur_filename
    
    with shelve.open(filename_in + ".arch") as ms:
        for key in ms:
            globals()[key] = ms[key]

    is_pathological = any(sum(T,axis=0) < 1) or ((len(R) > 0) and any(sum(R,axis=0) < 1))

    if is_pathological:
        print(cur_filename + " is pathological")
    else:
    
        with shelve.open(filename_in + ".achieved") as ms:
            for key in ms:
                globals()[key] = ms[key]

        R_bool = (R != 0)
        T_bool = (T != 0)

        if p.N_PF == 0:   # network is TFs only
            def get_gene_exp(c_PF,c_TF):
                C_TF = sum(c_TF)
                pr_wrapper = lambda t: pr_gene_on(C_TF,c_TF[t])
                return list(map(pr_wrapper,T_bool))
        else:
            def get_gene_exp(c_PF,c_TF):
                C_PF = sum(c_PF)
                C_TF = sum(c_TF)
                    
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF,c_PF[r])*pr_gene_on(C_TF,c_TF[t])
                return concatenate(list(map(pr_wrapper,R_bool,T_bool)))
        
        # crosstalk metric decomposed
        def crosstalk_metric_decomp(x,c_PF,c_TF):
            d = x - get_gene_exp(c_PF,c_TF)
            erroneously_on = d[d < 0]
            erroneously_off = d[d >= 0]
            xtalk_eon = transpose(erroneously_on)@erroneously_on
            xtalk_eoff = transpose(erroneously_off)@erroneously_off
            return xtalk_eon, xtalk_eoff
        
        # CALCULATE DECOMPOSED XTALK #
        optres = dill.load(open(filename_in + ".xtalk", "rb"))
        
        for achieved_pattern in optres.keys():
            achieved_pattern_bool = int2bool(achieved_pattern,p.M_GENE)
            ngenes = sum(achieved_pattern_bool)

            target_pattern = zeros(p.M_GENE)
            target_pattern[achieved_pattern_bool] = 1
    
            c = optres[achieved_pattern][0].x
            xtalk_on, xtalk_off = crosstalk_metric_decomp(target_pattern,c[0:p.N_PF],c[p.N_PF:])
            xtalk_on_by_ngenes[ngenes].append(xtalk_on)
            xtalk_off_by_ngenes[ngenes].append(xtalk_off)
    
            """
            if 'PGTOL' in optres[achieved_pattern][0].message:
                pgtol[ii] = pgtol[ii] + 1
                pgtol_ngenes.append(ngenes)
            """

dill.dump(xtalk_on_by_ngenes, open(folder_in + "/" + prefix + "ensemble_xtalk_on.dat", "wb"))
dill.dump(xtalk_off_by_ngenes, open(folder_in + "/" + prefix + "ensemble_xtalk_off.dat", "wb"))

fig, ax = plt.subplots(figsize = (24,12))
ax.boxplot(xtalk_on_by_ngenes)
ax.set_xticklabels(map(str,range(p.M_GENE+1)))
ax.set_xlabel('number genes in achieved pattern')
ax.set_ylabel('crosstalk (on genes)')
ax.set_ylim([0,0.3])
ax.set_title(prefix)
plt.savefig(folder_in + "/plots/" + prefix + "xtalk_by_genes_on_boxplot.png")

fig, ax = plt.subplots(figsize = (24,12))
ax.boxplot(xtalk_off_by_ngenes)
ax.set_xticklabels(map(str,range(p.M_GENE+1)))
ax.set_xlabel('number genes in achieved pattern')
ax.set_ylabel('crosstalk (off genes)')
ax.set_ylim([0,0.3])
ax.set_title(prefix)
plt.savefig(folder_in + "/plots/" + prefix + "xtalk_by_genes_off_boxplot.png")

"""
fig, ax = plt.subplots(figsize = (24,12))
ax.hist(pgtol)
ax.set_title(prefix)
plt.savefig(prefix + "pgtol_hist.png")

fig, ax = plt.subplots(figsize = (24,12))
ax.hist(pgtol_ngenes)
ax.set_title(prefix)
plt.savefig(prefix + "pgtol_ngenes_hist.png")
"""
