from numpy import *

import shelve
import dill

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
import sys
import os
import importlib

folder_in = "cluster_2022-11-25"

sys.path.append(folder_in)
sys.path.append(folder_in + "/src")
sys.path.append(folder_in + "/res")

if not(os.path.exists(folder_in + "/plots")):
    os.mkdir(folder_in + "/plots")

from boolarr import *

prefixes = ["tf"]#"kpr","tf"]

for prefix in prefixes:
    p = importlib.import_module(prefix + "-params")
    
    xtalk_by_ngenes = dill.load(open(folder_in + "/" + prefix + "-ensemble_xtalk.dat","rb"))
    
    sum_total = 0
    for ls in xtalk_by_ngenes:
        sum_total = sum_total + len(ls)
    print(f"{sum_total}")
    
    fig, ax = plt.subplots(figsize = (24,12))
    # ax.violinplot([val or [nan,nan] for val in kpr_xtalk_by_ngenes])
    ax.boxplot(xtalk_by_ngenes)
    ax.set_xticklabels(map(str,range(p.M_GENE+1)))
    ax.set_xlabel('number genes in achieved pattern')
    ax.set_ylabel('crosstalk')
    ax.set_ylim([0,0.3])
    ax.set_title(prefix)
    # plt.savefig(folder_in + "/plots/" + "kpr-xtalk_by_ngenes_violin.png")
    plt.savefig(folder_in + "/plots/" + prefix + "-xtalk_by_ngenes_boxplot-2.png")

"""
kpr_entropy = dill.load(open(folder_in + "/kpr-entropy_achieved.dat","rb"))
tf_entropy = dill.load(open(folder_in + "/tf-entropy_achieved.dat","rb"))

fig, ax = plt.subplots(figsize = (24,12))
ax.boxplot([kpr_entropy,tf_entropy])
ax.set_xticklabels(['kpr','tf'])
ax.set_ylabel('entropy')
ax.set_title('entropy over achieved patterns assuming uniform distribution over inputs')

plt.savefig(folder_in + "/plots/" + "entropy_boxplot.png")

kpr_nachievable = dill.load(open(folder_in + "/kpr-nachievable.dat","rb"))
tf_nachievable = dill.load(open(folder_in + "/tf-nachievable.dat","rb"))

fig, ax = plt.subplots(figsize = (24,12))
ax.boxplot([kpr_nachievable,tf_nachievable])
ax.set_xticklabels(['kpr','tf'])
ax.set_ylabel('number achievable patterns')

plt.savefig(folder_in + "/plots/" + "nachievable_boxplot.png")
"""
