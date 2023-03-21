from numpy import *

import shelve
import dill

import matplotlib.pyplot as plt
import sys
import os
import importlib

folder_in = "cluster_2022-11-25"
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


xtalk_by_ngenes = [[] for x in range(p.M_GENE+1)]
entropy_achieved = []
nachievable = []
for cur_filename in filenames:
    #print(cur_filename)

    filename_in = folder_in + "/res/" + cur_filename
    
    with shelve.open(filename_in + ".arch") as ms:
        for key in ms:
            globals()[key] = ms[key]

    """
    is_pathological = any(sum(T,axis=0) < 1) or ((len(R) > 0) and any(sum(R,axis=0) < 1))

    if is_pathological:
        print(cur_filename + " is pathological")
    else:
    """
    
    with shelve.open(filename_in + ".achieved") as ms:
        for key in ms:
            globals()[key] = ms[key]

        ##---ENTROPY MEASURE PER NETWORK---##
        cur_nachievable = len(mappings)
        nachievable.append(cur_nachievable)
        ndegen = zeros(cur_nachievable)
        for ii, (key,value) in enumerate(mappings.items()):
            ndegen[ii] = len(value)

        # assuming distribution over inputs is uniform, calculate entropy over realized patterns
        prob_achieved_pattern = ndegen / p.NUM_RANDINPUTS
        entropy_achieved.append(-sum(multiply(prob_achieved_pattern,log2(prob_achieved_pattern))))
            
        ##---POOL XTALK RESULTS---##
        optres = dill.load(open(filename_in + ".xtalk", "rb"))
        
        for achieved_pattern in optres.keys():
        # if not(mappings.get(achieved_pattern) == None):
        # if mappings.get(achieved_pattern) == None:
            ngenes = sum(int2bool(achieved_pattern,p.M_GENE))
            xtalk_by_ngenes[ngenes].append(optres[achieved_pattern][0].fun)

dill.dump(xtalk_by_ngenes, open(folder_in + "/" + prefix + "ensemble_xtalk.dat","wb"))
dill.dump(entropy_achieved, open(folder_in + "/" + prefix + "entropy_achieved.dat","wb"))
dill.dump(nachievable, open(folder_in + "/" + prefix + "nachievable.dat","wb"))
