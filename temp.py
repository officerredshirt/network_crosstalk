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

files = os.listdir(folder_in + "/res/")
filenames = [fn for fn in files if fn.startswith(prefix) & fn.endswith(suffix)]
filenames = [fn[:-len(suffix)] for fn in filenames]

for fn in filenames:
    disp("Processing " + fn + "...")
    if os.system("python3 " + "src/get_achievable_patterns.py -m -i " + folder_in + "/res/" + fn):
        break
