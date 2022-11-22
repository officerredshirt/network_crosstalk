#!/usr/bin/env python
# coding: utf-8
from numpy import *
from multiprocess import Pool
import random as ran
import shelve
import sys, getopt
# import time

from params import *
from boolarr import *

def print_usage():
    print("usage is: get_achievable_patterns.py -m -i <filename_in>")


# here logical vectors are often treated as binary strings and represented as integers up to 2^VEC_LENGTH
def main(argv):
    filename_in = ""
    thresh_gene_on = 0
    suffix = ""

    try:
        opts, args = getopt.getopt(argv,"hmi:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    if len(argv) < 1:
        print("get_achievable_patterns.py: input -i is required")
        print_usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-i":
            filename_in = arg
        elif opt in ("-m", "--max"):
            # require all TFs to be present to activate gene
            thresh_gene_on = THETA-1 # strictly greater than this
            suffix = ".max"

    # tic = time.perf_counter()

    # load architecture
    with shelve.open(filename_in + ".arch") as ms:
        for key in ms:
            globals()[key] = ms[key]
            # print(key)

    # generate random inputs (as integers)
    inputs_to_test = ran.sample(range(pow(2,N_PF + N_TF)),NUM_RANDINPUTS)

    # calculate patterns achieved by each input
    mappings = {}
    for u_int in inputs_to_test:
        # convert u as integer into u as boolean vector
        u = int2bool(u_int,N_PF + N_TF)
        
        # R: M_enh x N_PF
        # T: M_enh x N_TF
        # G: M_gene x M_enh
        if N_PF == 0:
             enhancer_exp_level = T@u
             gene_is_on = G@enhancer_exp_level > thresh_gene_on 
        else:
             enhancer_exp_level = (R@u[0:N_PF]) * (T@u[N_PF:])
             gene_is_on = G@enhancer_exp_level > thresh_gene_on
        # enhancer_exp_level = (matmul(R,u[0:N_PF])) * (matmul(T,u[N_PF:]))
        # gene_is_on = matmul(G,enhancer_exp_level) > 0
        
        # convert achieved pattern to int
        gene_on_int = bool2int(gene_is_on)
        
        # dictionary of patterns to inputs (pattern is key since input -> pattern is surjective but not necessarily injective)
        mappings.setdefault(gene_on_int,[])
        mappings[gene_on_int].append(u_int)

    with shelve.open(filename_in + ".achieved" + suffix,'n') as ms_out:
        ms_out['inputs_to_test'] = inputs_to_test
        ms_out['mappings'] = mappings

    # toc = time.perf_counter()
    # print(f"elapsed time: {toc-tic} s, {len(inputs_to_test)} inputs")


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
