#!/usr/bin/env python
# coding: utf-8

from numpy import *
from scipy import optimize
from multiprocess import Pool
import shelve
import dill
# import time
import matplotlib.pyplot as plt
import sys, getopt
# from memory_profiler import profile, memory_usage

from params import *
from tf_binding_equilibrium import *
from boolarr import *


def print_usage():
    print("usage is: calc_crosstalk -i <filename_in> -n <ndict_entries>")

# @profile
def main(argv):
    # mem_usage = memory_usage(-1, interval=.2, timeout=1, max_usage=True, include_children=True)
    filename_in = ""
    ndict_entries = inf

    try:
        opts, args = getopt.getopt(argv,"hi:n:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if len(argv) < 1:
        print("calc_crosstalk.py: input -i is required")
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-i":
            filename_in = arg
        elif opt == "-n":
            ndict_entries = int(arg)


    # load architecture
    with shelve.open(filename_in + ".arch") as ms:
        for key in ms:
            globals()[key] = ms[key]
            # print(key)

    # load achievable patterns
    with shelve.open(filename_in + ".achieved") as ms:
        for key in ms:
            globals()[key] = ms[key]
            # print(key)

    # pr_chromatin_open = dill.load(open("./src/chromatin_6state_pr_open.out", "rb"))
    # pr_gene_on is imported with tf_binding_equilibrium
    pr_chromatin_open = dill.load(open("./src/chromatin_kpr_pr_open.out", "rb"))


    # note: for loop and map implementations seem equally efficient
    # TODO: support multiple enhancers per gene

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
            
            pr_wrapper = lambda r,t: pr_chromatin_open(C_PF,c_PF[r])*pr_gene_on(C_TF,c_TF[t])
        
            return concatenate(list(map(pr_wrapper,R_bool,T_bool)))

    # crosstalk metric
    def crosstalk_metric(x,c_PF,c_TF):
        d = x - get_gene_exp(c_PF,c_TF)
        return transpose(d)@d


    if ndict_entries > len(mappings):
        ndict_entries = len(mappings)
    elif ndict_entries < 1:
        print("ndict_entries must be strictly positive")
        sys.exit(2)
        
    eps = 1e-6   # tolerance for optimization
    optim_results = {}

    # tstart = time.perf_counter()
    for ii, (key, value) in enumerate(mappings.items()):
        # print(str(f"key: {key}, value: {value}"))
        if ii >= ndict_entries:
            break

        print(ii,end="")
        sys.stdout.flush()
        achieved_pattern = int2bool(key,M_GENE)
        target_pattern = zeros(M_GENE)
        target_pattern[achieved_pattern] = 1

        # optimize concentration of PFs and TFs in the input to reduce crosstalk metric
        def crosstalk_objective_fn(c):
            return crosstalk_metric(target_pattern,c[0:N_PF],c[N_PF:])
        bnds = [(0,inf)]*(N_PF + N_TF)   # force concentrations positive

        optim_results.setdefault(key,[])

        def optim(inp):
            print("", end=".")
            sys.stdout.flush()
            
            # starting point
            c_0_bool = int2bool(inp,N_PF + N_TF)
            c_0 = zeros(N_PF + N_TF)
            c_0[c_0_bool] = 1

            A = identity(N_PF + N_TF)
            A = delete(A, c_0_bool, axis = 0)
            
            if len(A) != 0:
                cons = {optimize.LinearConstraint(A,0,0)}   # force solution to follow form of given input
                optres = optimize.minimize(crosstalk_objective_fn, c_0, tol = eps, bounds = bnds, constraints = cons)
            else:
                optres = optimize.minimize(crosstalk_objective_fn, c_0, tol = eps, bounds = bnds)

            print("", end="!")
            sys.stdout.flush()
                
            # optim_results[key].append(optres)
            return optres

        if __name__ == "__main__":
            with Pool() as pool:
                all_optres = pool.map(optim,value)

            print("")

            for res in all_optres:
                optim_results[key].append(res)

    # tend = time.perf_counter()
    # print(f"elapsed time = {tend - tstart}")
    # print(f"memory usage = {mem_usage}")


    dill.dump(optim_results, open(filename_in + ".xtalk", "wb"))


if __name__ == "__main__":
    main(sys.argv[1:])
