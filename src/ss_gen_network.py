#!/usr/bin/env python
# coding: utf-8
from numpy import *
import sys, argparse
import manage_db

from params import *

# TODO:
# [ ] assign connections according to particular distributions
# [ ] allow more than one gene per enhancer

def main(argv):

    parser = argparse.ArgumentParser(
            prog = "ss_gen_network",
            description = "",
            epilog = "")
    parser.add_argument("-o","--filename_out",required=True,default="out.arch")
    parser.add_argument("-d","--database",required=True)

    args = parser.parse_args()
    filename_out = args.filename_out
    database = args.database

    # generate database
    manage_db.init_tables(database)

    # populate parameters
    manage_db.add_parameters(database)

    # R is M_GENE x N_PF
    if sc == 1:
        R = identity(M_GENE)
    elif sc == 2:
        R_ul = ones((N_ON,1))
        R_ur = zeros((N_ON,N_PF-1))
        R_ll = zeros((N_PF-1,1))
        R_lr = identity(N_PF-1)
        R = block([[R_ul,R_ur],[R_ll,R_lr]])
    elif sc == 3:
        R_ul = ones((N_ON+N_OFF_shared,1))
        R_ur = zeros((N_ON+N_OFF_shared,N_PF-1))
        R_ll = zeros((N_PF-1,1))
        R_lr = identity(N_PF-1)
        R = block([[R_ul,R_ur],[R_ll,R_lr]])

    T = identity(M_GENE)
    G = identity(M_GENE)
    

    # -- UPLOAD ARCHITECTURE TO DATABASE -- #
    local_id = manage_db.extract_local_id(filename_out)
    manage_db.add_network(database,local_id,R,G,T)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_out + ".arch","w") as file:
        pass
    

if __name__ == "__main__":
    main(sys.argv[1:])
