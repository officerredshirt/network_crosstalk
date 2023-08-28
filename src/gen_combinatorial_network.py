#!/usr/bin/env python
# coding: utf-8
from numpy import *
from numpy import matlib
import sys, getopt
import manage_db
import itertools

from params import *

def print_usage():
    print("usage is: gen_combinatorial_network.py -o <filename_out> -d <database>")


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

    if database == "temp.db":
        disp("no database provided---proceeding with temp.db...")

    # generate database
    manage_db.init_tables(database)

    # populate parameters
    manage_db.add_parameters(database)


    # -- GENERATE ARCHITECTURE -- #
    # combinatorially addressed network with N_PF, N_TF as in params.py

    # generate connections from PFs to enhancers
    R = matlib.repmat(identity(N_PF),N_TF,1)
    
    # generate connections from TFs to enhancers
    T = matlib.repeat(identity(N_TF),N_PF,axis=0)
    
    # generate connections from enhancers to genes
    # trivial for now (one gene per enhancer)
    G = identity(M_GENE)


    # -- UPLOAD ARCHITECTURE TO DATABASE -- #
    local_id = manage_db.extract_local_id(filename_out)
    manage_db.add_network(database,local_id,R,G,T)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_out + ".arch","w") as file:
        pass
    

if __name__ == "__main__":
    main(sys.argv[1:])
