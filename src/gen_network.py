#!/usr/bin/env python
# coding: utf-8
from numpy import *
import sys, getopt
import manage_db

from params import *

def print_usage():
    print("usage is: gen_network.py -o <filename_out> -d <database>")


# TODO:
# [ ] assign connections according to particular distributions
# [ ] allow more than one gene per enhancer

def main(argv):
    filename_out = "out.arch"
    database = "temp.db"

    try:
        opts, args = getopt.getopt(argv,"ho:d:")
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print_usage()
            sys.exit()
        elif opt == "-o":
            filename_out = arg
        elif opt == "-d":
            database = arg

    print(filename_out)
    assert not(database == "temp.db"), "database not passed correctly"

    is_pathological = True

    while is_pathological:

        # generate connections from PF to enhancers
        if N_PF != 0:
            R = zeros([M_ENH,N_PF])
            for ii in range(M_ENH):
                R[ii,random.randint(0,N_PF)] = 1
        else:
            R = array([])
        
        # generate connections from TFs to enhancers
        T = zeros([M_ENH,N_TF])
        temp = concatenate((ones(THETA),zeros(N_TF - THETA)))
        for ii in range(M_ENH):
            T[ii,] = random.permutation(temp)
        
        # generate connections from enhancers to genes
        # trivial for now (one gene per enhancer)
        G = identity(M_GENE)

        is_pathological = any(sum(T,axis=0) < 1) or ((len(R) > 0) and any(sum(R,axis=0) < 1))
        

    # -- UPLOAD ARCHITECTURE TO DATABASE -- #
    local_id = manage_db.extract_local_id(filename_out)
    manage_db.add_network(database,local_id,R,G,T)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_out + ".arch","w") as file:
        pass
    

if __name__ == "__main__":
    main(sys.argv[1:])
