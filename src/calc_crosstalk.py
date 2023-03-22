#!/usr/bin/env python
# coding: utf-8

from numpy import *
from scipy import optimize
from multiprocess import Pool
import dill
# import time
import matplotlib.pyplot as plt
import os, sys, argparse
# from memory_profiler import profile, memory_usage

from params import *
# from tf_binding_equilibrium import *
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
    parser.add_argument("-m","--model_folder",required=True)
    parser.add_argument("-n","--npatterns",type=int,default=inf)
    parser.add_argument("-d","--database",required=True)
    parser.add_argument("-x","--crosslayer_crosstalk",action="store_true",default=False)
    parser.add_argument("-t","--tf_first_layer",action="store_true",default=False)
    parser.add_argument("-c","--minimize_noncognate_binding",action="store_true",default=False)

    args = parser.parse_args()
    filename_in = args.filename_in
    npatterns = args.npatterns
    database = args.database
    crosslayer_crosstalk = args.crosslayer_crosstalk
    tf_first_layer = args.tf_first_layer
    minimize_noncognate_binding = args.minimize_noncognate_binding
    model_folder = args.model_folder

    local_id = manage_db.extract_local_id(filename_in)

    # load architecture
    R, T, G = manage_db.get_network(database,local_id)

    # load target patterns
    target_patterns = manage_db.get_target_patterns(database,local_id)

    # pr_tf_bound is imported with tf_binding_equilibrium
    if tf_first_layer:
        pr_chromatin_open = dill.load(open(os.path.join(model_folder,"tf_chrom_equiv_pr_tf_bound.out"),"rb"))
        pr_chromatin_error = dill.load(open(os.path.join(model_folder,"tf_chrom_equiv_error_rate.out"),"rb"))
    else:
        pr_chromatin_open = dill.load(open(os.path.join(model_folder,"kpr_pr_open.out"), "rb"))
        pr_chromatin_error = dill.load(open(os.path.join(model_folder,"kpr_opening_error_rate.out"),"rb"))

    pr_tf_bound = dill.load(open(os.path.join(model_folder,"tf_pr_bound.out"),"rb"))
    pr_tf_error = dill.load(open(os.path.join(model_folder,"tf_error_rate.out"),"rb"))


    R_bool = (R != 0)
    T_bool = (T != 0)
    # TODO: support multiple enhancers per gene
    if N_PF == 0:   # network is layer 2 TFs only
        def get_gene_exp(c_PF,c_TF):
            C_TF = sum(c_TF)

            pr_wrapper = lambda t: pr_tf_bound(C_TF,c_TF[t])

            return list(map(pr_wrapper,T_bool))

    else:
        def get_gene_exp(c_PF,c_TF):
            C_PF = sum(c_PF)
            C_TF = sum(c_TF)
            
            if crosslayer_crosstalk:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF+C_TF,c_PF[r])*pr_tf_bound(C_TF+C_PF,c_TF[t])
            else:
                pr_wrapper = lambda r,t: pr_chromatin_open(C_PF,c_PF[r])*pr_tf_bound(C_TF,c_TF[t])
        
            return concatenate(list(map(pr_wrapper,R_bool,T_bool)))

        def get_error_frac(c_PF,c_TF):
            C_PF = sum(c_PF)
            C_TF = sum(c_TF)

            if crosslayer_crosstalk:
                E1 = lambda r: pr_chromatin_error(C_PF+C_TF,c_PF[r])
                E2 = lambda t: pr_tf_error(C_TF+C_PF,c_TF[t])
            else:
                E1 = lambda r: pr_chromatin_error(C_PF,c_PF[r])
                E2 = lambda t: pr_tf_error(C_TF,c_TF[t])

            pr_err_wrapper = lambda r,t: E1(r) + E2(t) - E1(r)*E2(t)
            return concatenate(list(map(pr_err_wrapper,R_bool,T_bool)))

    # crosstalk metric
    if minimize_noncognate_binding:
        def crosstalk_metric(x,c_PF,c_TF):
            gene_exp = get_gene_exp(c_PF,c_TF)
            err_frac = get_error_frac(c_PF,c_TF)
            d1 = x - gene_exp*(1-err_frac)
            d2 = gene_exp*err_frac
            return transpose(d1)@d1 + transpose(d2)@d2
    else:   # patterning error
        def crosstalk_metric(x,c_PF,c_TF):
            d = x - get_gene_exp(c_PF,c_TF)
            return transpose(d)@d


    if npatterns > len(target_patterns):
        npatterns = len(target_patterns)
    elif npatterns < 1:
        print("npatterns must be strictly positive")
        sys.exit(2)
        

    # tstart = time.perf_counter()
    for ii, target_pattern in enumerate(target_patterns):
        if ii >= npatterns:
            break

        # optimize concentration of PFs and TFs in the input to reduce crosstalk metric
        def crosstalk_objective_fn(c):
            return crosstalk_metric(target_pattern,c[0:N_PF],c[N_PF:])

        bnds = [(0,inf)]*(N_PF + N_TF)   # force concentrations positive

        if not(manage_db.xtalk_result_found(database,local_id,target_pattern)):
            # starting point
            c_0 = [10]*(N_PF + N_TF)
            optres = optimize.minimize(crosstalk_objective_fn, c_0, tol = eps, bounds = bnds)
            output_expression = get_gene_exp(optres.x[0:N_PF],optres.x[N_PF:])
            manage_db.add_xtalk(database,local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optres,output_expression)
    

    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_in + ".xtalk","w") as file:
        pass

    # tend = time.perf_counter()
    # print(f"elapsed time = {tend - tstart}")
    # print(f"memory usage = {mem_usage}")


if __name__ == "__main__":
    main(sys.argv[1:])
