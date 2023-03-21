from numpy import *

import shelve
import dill

import matplotlib.pyplot as plt
import sys
import os
import importlib

folder_in = "cluster_2022-11-21"
prefix = "kpr-"
suffix = ".xtalk"

sys.path.append(folder_in)
sys.path.append(folder_in + "/src")
sys.path.append(folder_in + "/res")

from boolarr import *
from tf_binding_equilibrium import *

p = importlib.import_module(prefix + "params")
pr_chromatin_open = dill.load(open(folder_in + "/src/chromatin_kpr_pr_open.out","rb"))

with shelve.open(folder_in + "/res/kpr-000001.achieved") as ms:
    all_achieved = ms.get("mappings")
    
with shelve.open(folder_in + "/res/kpr-000001.achieved.max") as ms:
    max_achieved = ms.get("mappings")

not_shared = 0
for key in all_achieved:
    if max_achieved.get(key) == None:
        not_shared = not_shared + 1

print(f"{not_shared} out of {len(all_achieved)} not shared")
