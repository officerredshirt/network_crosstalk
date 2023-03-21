#!/usr/bin/env python
# coding: utf-8
from numpy import *
import sys, argparse
import manage_db

from params import *


def main(argv):

    parser = argparse.ArgumentParser(
            prog = "gen_custom_network",
            description = "",
            epilog = "")
    parser.add_argument("-o","--filename_out",required=True,default="out.arch")
    parser.add_argument("-d","--database",required=True)

    args = parser.parse_args()
    filename_out = args.filename_out
    database = args.database

    # generate database
    print("Generating database...")
    manage_db.init_tables(database)

    # populate parameters
    print("Adding parameters...")
    manage_db.add_parameters(database)

    # R (M_ENH x N_PF): connections from PFs to enhancers
    # one PF per cluster of GENES_PER_CLUSTER genes
    R = zeros([M_ENH,N_PF])
    for ii in range(0,N_PF):
        R[ii:ii+GENES_PER_CLUSTER,ii] = 1

    # T (M_ENH x N_TF): connections from TFs to enhancers
    T = identity(M_GENE)

    # G (M_GENE x M_ENH): connections from enhancers to genes
    G = identity(M_GENE)
    

    # -- UPLOAD ARCHITECTURE TO DATABASE -- #
    local_id = manage_db.extract_local_id(filename_out)
    manage_db.add_network(database,local_id,R,G,T)


    # -- SAVE PROOF OF COMPLETION FOR SNAKEMAKE FLOW -- #
    with open(filename_out + ".arch","w") as file:
        pass
    

if __name__ == "__main__":
    main(sys.argv[1:])
