#!/usr/bin/env python
# coding: utf-8
from numpy import *
from multiprocess import Pool
import random as ran
import shelve
import sys, getopt
import manage_db
# import time

from params import *
from boolarr import *

def print_usage():
    print("usage is: get_achievable_patterns.py -i <filename_in> -d <database>")


# here logical vectors are often treated as binary strings and represented as integers up to 2^VEC_LENGTH
def main(argv):
    filename_in = ""
    suffix = ""
    database = "temp.db"

    try:
        opts, args = getopt.getopt(argv,"hi:d:")
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
        elif opt == "-d":
            database = arg

    # tic = time.perf_counter()

    # load architecture
    local_id = manage_db.extract_local_id(filename_in)
    R, T, G = manage_db.get_network(database,local_id)

    # generate random inputs (as integers)
    # DO WE NEED TO CHANGE THIS to account for limited size of integer type?
    inputs_to_test = ran.sample(range(pow(2,N_PF + N_TF)),NUM_RANDINPUTS)

    # calculate patterns achieved by each input
    for u_int in inputs_to_test:
        # convert u as integer into u as boolean vector
        u = int2bool(u_int,N_PF + N_TF)
        
        # R: M_enh x N_PF
        # T: M_enh x N_TF
        # G: M_gene x M_enh
        if N_PF == 0:
            enhancer_exp_level = T@u
            gene_is_on = G@enhancer_exp_level > thresh_gene_on 

            if sum(gene_is_on) > 0:
                achieved_pattern = gene_is_on.astype(int)
            else:
                achieved_pattern = zeros(len(gene_is_on))
        else:
            enhancer_exp_level = (R@u[0:N_PF]) * (T@u[N_PF:])
            gene_exp_level = G@enhancer_exp_level
            ix_nonzero = gene_exp_level.astype(bool)

            if sum(ix_nonzero.astype(int)) > 0:
                new_entries = ones(sum(ix_nonzero))/gene_exp_level[ix_nonzero]
                achieved_pattern = gene_exp_level
                put(achieved_pattern,where(ix_nonzero),new_entries)
            else:
                achieved_pattern = zeros(len(ix_nonzero))

        # -- UPLOAD ACHIEVED PATTERN TO DATABASE -- #
        local_id = manage_db.extract_local_id(filename_in)
        manage_db.add_pattern(database,local_id,u,achieved_pattern)
        
    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_in + ".achieved","w") as file:
        pass

    # toc = time.perf_counter()
    # print(f"elapsed time: {toc-tic} s, {len(inputs_to_test)} inputs")


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
