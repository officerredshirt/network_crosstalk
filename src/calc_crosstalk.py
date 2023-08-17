#!/usr/bin/env python
# coding: utf-8

from numpy import *
from scipy import optimize
from multiprocess import Pool
import itertools
import dill
import time
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
    parser.add_argument("-s","--suppress_filesave",action="store_true",default=False)

    args = parser.parse_args()
    filename_in = args.filename_in
    npatterns = args.npatterns
    database = args.database
    crosslayer_crosstalk = args.crosslayer_crosstalk
    tf_first_layer = args.tf_first_layer
    minimize_noncognate_binding = args.minimize_noncognate_binding
    model_folder = args.model_folder
    suppress_filesave = args.suppress_filesave
    
    local_id = manage_db.extract_local_id(filename_in)

    # load target patterns
    input_for_target, target_patterns = manage_db.get_target_patterns(database,local_id)

    # load crosstalk metric
    crosstalk_metric = manage_db.get_crosstalk_metric_from_file(filename_in,database,N_PF,N_TF,crosslayer_crosstalk,tf_first_layer,minimize_noncognate_binding,model_folder)

    if npatterns > len(target_patterns):
        npatterns = len(target_patterns)
    elif npatterns < 1:
        print("npatterns must be strictly positive")
        sys.exit(2)
        

    tstart = time.perf_counter()
    print(f"Calculating crosstalk (minimize_noncognate_binding = {minimize_noncognate_binding}, tf_first_layer = {tf_first_layer})")
    print("  ",end="")

    def optim(target_pattern):
        # optimize concentration of PFs and TFs in the input to reduce crosstalk metric

        # decide whether to redo other optimizations with this setting...
        if not target_independent_of_clusters:
            N_TF_to_use = sum(target_pattern > 0)
            N_PF_to_use = int(N_TF_to_use / GENES_PER_CLUSTER)

            def crosstalk_objective_fn(c):
                cp = concatenate((c[0:N_PF_to_use],[0]*(N_PF - N_PF_to_use)))
                ct = concatenate((c[N_PF_to_use:],[0]*(N_TF - N_TF_to_use)))
                return crosstalk_metric(target_pattern,cp,ct,
                                        ignore_off_for_opt = ignore_off_during_optimization,
                                        off_ixs = (target_pattern == 0))
        else:
            N_TF_to_use = N_TF
            N_PF_to_use = N_PF

            def crosstalk_objective_fn(c):
                return crosstalk_metric(target_pattern,c[0:N_PF_to_use],c[N_PF_to_use:],
                                        ignore_off_for_opt = ignore_off_during_optimization,
                                        off_ixs = (target_pattern == 0))

        bnds = [(0,inf)]*(N_PF_to_use + N_TF_to_use)    # force concentrations positive

        if not(manage_db.xtalk_result_found(database,local_id,int(minimize_noncognate_binding),int(tf_first_layer),target_pattern)):
            print(".",end="",flush=True)
            # starting point
            c_0 = array([10]*(N_PF_to_use + N_TF_to_use))
            #c_0[cur_input] = 10 

            try:
                optres = optimize.minimize(crosstalk_objective_fn, c_0, tol = eps, bounds = bnds,
                                           method = "L-BFGS-B", options = {"maxfun":1000000})

                if not target_independent_of_clusters:
                    optres.x = concatenate((optres.x[0:N_PF_to_use],zeros(N_PF - N_PF_to_use),
                                           optres.x[N_PF_to_use:],zeros(N_TF - N_TF_to_use)))

                if ignore_off_during_optimization:
                    # store actual error even though optimization itself ignores OFF genes
                    optres.f = crosstalk_metric([], \
                            optres.x[0:N_PF],optres.x[N_PF:])
                output_expression = crosstalk_metric([], \
                        optres.x[0:N_PF],optres.x[N_PF:], \
                        return_var="gene_exp")#get_gene_exp(optres.x[0:N_PF],optres.x[N_PF:])
                output_error = crosstalk_metric([], \
                        optres.x[0:N_PF],optres.x[N_PF:], \
                        return_var="error_frac")#get_error_frac(optres.x[0:N_PF],optres.x[N_PF:])
                max_expression = crosstalk_metric([],[],[],return_var="max_expression")
                manage_db.add_xtalk(database,local_id,minimize_noncognate_binding,crosslayer_crosstalk,tf_first_layer,target_pattern,optres,output_expression,output_error,max_expression)
                print("! ",end="",flush=True)
            except Exception as e:
                print(f"optimization error \"{e}\"; skipping...")
                #pass

    with Pool() as pool:
        pool.map(optim,target_patterns[0:npatterns])
    
    print("")

    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #

    print("Saving proof of completion...")
    with open(filename_in + ".xtalk","w") as file:
        pass

    tend = time.perf_counter()
    print(f"elapsed time = {tend - tstart}")
    # print(f"memory usage = {mem_usage}")


if __name__ == "__main__":
    main(sys.argv[1:])
