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

p = importlib.import_module(prefix + "params")

files = os.listdir(folder_in + "/res/")
filenames = [fn for fn in files if fn.startswith(prefix) & fn.endswith(suffix)]
filenames = [fn[:-len(suffix)] for fn in filenames]

fig, ax = plt.subplots(figsize = (24,12))
for cur_filename in filenames[0:5]:
    print("Now processing " + cur_filename + "...")

    xtalk_by_ngenes = [[] for x in range(p.M_GENE+1)]

    filename_in = folder_in + "/res/" + cur_filename
    
    with shelve.open(filename_in + ".arch") as ms:
        for key in ms:
            globals()[key] = ms[key]

    with shelve.open(filename_in + ".achieved") as ms:
        for key in ms:
            globals()[key] = ms[key]

    ##---POOL XTALK RESULTS---##
    optres = dill.load(open(filename_in + ".xtalk", "rb"))
   
    for achieved_pattern in optres.keys():
        ngenes = sum(int2bool(achieved_pattern,p.M_GENE))
        xtalk_by_ngenes[ngenes].append(optres[achieved_pattern][0].fun)

    dill.dump(xtalk_by_ngenes, open(filename_in + "_xtalk.dat","wb"))

    val_xtalk =([(x,y) for x,y in enumerate(xtalk_by_ngenes) if not(len(y) == 0)])

    # print([x[0] for x in val_xtalk])
    # print(len(list(map(mean,[x[1] for x in val_xtalk]))))

    ax.scatter([x[0] for x in val_xtalk],list(map(mean,[x[1] for x in val_xtalk])))

# ax.set_xticklabels(map(str,range(p.M_GENE+1)))
ax.set_xlabel('number genes in achieved pattern')
ax.set_ylabel('crosstalk')
# ax.set_ylim([0,0.3])
ax.set_title(prefix)

plt.savefig(folder_in + "/plots/" + prefix + "xtalk_partial_scatter.png")
