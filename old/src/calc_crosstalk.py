#!/usr/bin/env python
# coding: utf-8

from numpy import *
from scipy import optimize
from multiprocess import Pool
import dill
# import time
import matplotlib.pyplot as plt
import sys, argparse
from os.path import exists
# from memory_profiler import profile, memory_usage

from params import *
from tf_binding_equilibrium import *
from boolarr import *

import manage_db


# @profile
def main(argv):
    # mem_usage = memory_usage(-1, interval=.2, timeout=1, max_usage=True, include_children=True)
    filename_in = ""
    npatterns = inf
    database = "temp.db"

    parser = argparse.ArgumentParser(
            prog = "calc_crosstalk",
            description = "",
            epilog = "")
    parser.add_argument("-i","--filename_in",required=True)
    parser.add_argument("-n","--npatterns",type=int,default=inf)
    parser.add_argument("-d","--database",required=True)
    parser.add_argument("-x","--crosslayer_crosstalk",action="store_true",default=False)
    parser.add_argument("-t","--tf_first_layer",action="store_true",default=False)

    args = parser.parse_args()
    filename_in = args.filename_in
    npatterns = args.npatterns
    database = args.database
    crosslayer_crosstalk = args.crosslayer_crosstalk
    tf_first_layer = args.tf_first_layer

    local_id = manage_db.extract_local_id(filename_in)

    # load architecture
    R, T, G = manage_db.get_network(database,local_id)

    # load achievable patterns
    achievable_patterns = manage_db.get_achieved_patterns(database,local_id)

    # pr_gene_on is imported with tf_binding_equilibrium
    if tf_first_layer:
        pr_chromatin_open = dill.load(open("./src/tf_chrom_equiv_pr_bound.out","rb"))
    else:
        pr_chromatin_open = dill.load(open("./src/chromatin_kpr_pr_open.out", "rb"))


    R_bool = (R != 0)
    T_bool = (T != 0)
    # TODO: support multiple enhancers per gene
    if N_PF == 0:   # network is TFs only
        def get_gene_exp(c_PF,c_TF):
            C_TF = sum(c_TF)

            pr_wrapper = lambda t: pr_gene_on(C_TF,c_TF[t])

            return list(map(pr_wrapper,T_bool))

    else:
        def get_gene_exp(c_PF,c_TF):
            C_PF = sum(c_PF)
            C_TF = sum(c_TF)
            
            if crosslayer_crosstalk:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF+C_TF,c_PF[r])*pr_gene_on(C_TF+C_PF,c_TF[t])
            else:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF,c_PF[r])*pr_gene_on(C_TF,c_TF[t])
        
            return concatenate(list(map(pr_wrapper,R_bool,T_bool)))

    # crosstalk metric
    def crosstalk_metric(x,c_PF,c_TF):
        d = x - get_gene_exp(c_PF,c_TF)
        return transpose(d)@d


    if npatterns > len(achievable_patterns):
        npatterns = len(achievable_patterns)
    elif npatterns < 1:
        print("npatterns must be strictly positive")
        sys.exit(2)
        
    eps = 1e-6   # tolerance for optimization

    # tstart = time.perf_counter()
    for ii, target_pattern in enumerate(achievable_patterns):
        if ii >= npatterns:
            break

        # optimize concentration of PFs and TFs in the input to reduce crosstalk metric
        def crosstalk_objective_fn(c):
            return crosstalk_metric(target_pattern,c[0:N_PF],c[N_PF:])

        bnds = [(0,inf)]*(N_PF + N_TF)   # force concentrations positive

        if not(manage_db.xtalk_result_found(database,local_id,target_pattern)):
            # starting point
            c_0 = ones(N_PF + N_TF)
            optres = optimize.minimize(crosstalk_objective_fn, c_0, tol = eps, bounds = bnds)
            output_expression = get_gene_exp(optres.x[0:N_PF],optres.x[N_PF:])
            manage_db.add_xtalk(database,local_id,target_pattern,optres,output_expression)
    

    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_in + ".xtalk","w") as file:
        pass

    # tend = time.perf_counter()
    # print(f"elapsed time = {tend - tstart}")
    # print(f"memory usage = {mem_usage}")


if __name__ == "__main__":
    main(sys.argv[1:])
